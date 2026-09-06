package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"slices"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// maxStoredHistoryMessages bounds how much of a conversation GET
// /api/chat/history/{conversationID} returns — the reload-render path, which
// shows the user more than the agent ever needs (loadConversation, the
// WS-seed path, uses its own much smaller cap; see there).
const maxStoredHistoryMessages = 50

// conversationTurn is the wire shape of GET /api/chat/history — one entry per
// completed exchange. title_refs is deliberately not surfaced here: nothing
// yet re-renders title cards from a reload, only the text (see
// ARCHITECTURE.md's data model and DECISIONS.md's "One socket per session").
type conversationTurn struct {
	UserText      string `json:"user_text"`
	AssistantText string `json:"assistant_text"`
}

// fetchRecentMessages returns up to limit of one conversation's most recent
// messages, oldest-first. Shared by loadConversation (which only ever needs
// windowHistory's own small window) and loadConversationTurns (which shows
// the caller much more) — same query and scan, different caps.
func fetchRecentMessages(ctx context.Context, tx pgx.Tx, conversationID string, limit int) ([]historyTurn, error) {
	// Newest-first with a limit, then reversed below, so the cap keeps the
	// most recent messages rather than the oldest ones. Ordered by seq, not
	// created_at: saveMessages inserts a turn's user and assistant row in one
	// statement, so both get the exact same now() value, and two rows with a
	// tied timestamp have no guaranteed order — the identity column does.
	rows, err := tx.Query(ctx, `
		select role, content from messages
		where conversation_id = $1
		order by seq desc
		limit $2`,
		conversationID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var history []historyTurn
	for rows.Next() {
		var t historyTurn
		if err := rows.Scan(&t.Role, &t.Text); err != nil {
			return nil, err
		}
		history = append(history, t)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	slices.Reverse(history)
	return history, nil
}

// pairTurns groups a flat, chronological role/text list into completed
// exchanges — expects strict user/assistant alternation, an invariant only
// saveMessages can create since it's the sole writer and always inserts both
// rows of a turn together. Checked, not just assumed: fetchRecentMessages'
// caller-supplied limit landing on an odd boundary, or any future write path
// that doesn't preserve the pairing, would otherwise silently swap which
// text renders as the user's and which as the assistant's. Once alternation
// breaks, everything from that point is dropped rather than mispaired.
func pairTurns(flat []historyTurn) []conversationTurn {
	turns := make([]conversationTurn, 0, len(flat)/2)
	for i := 0; i+1 < len(flat); i += 2 {
		if flat[i].Role != "user" || flat[i+1].Role != "assistant" {
			slog.Warn("conversation history is misaligned, dropping the rest", "index", i, "role", flat[i].Role)
			return turns
		}
		turns = append(turns, conversationTurn{UserText: flat[i].Text, AssistantText: flat[i+1].Text})
	}
	if len(flat)%2 != 0 {
		slog.Warn("conversation history has an odd trailing message, dropping it", "role", flat[len(flat)-1].Role)
	}
	return turns
}

// maxTitleLength caps a derived conversation title at roughly one sidebar
// line's worth of text (runes, not bytes, so a multi-byte character is never
// sliced mid-codepoint).
const maxTitleLength = 60

// defaultConversationTitle is what a sidebar row shows for a conversation
// with no derived title yet — titleFromMessage's own fallback (an
// empty/whitespace-only first message) and loadConversationSummaries'
// coalesce fallback for a null title (a row from before T20.5 introduced
// titles) share this one constant rather than each hardcoding "New chat".
const defaultConversationTitle = "New chat"

// titleFromMessage derives a conversation's title from its opening message
// (TASKS.md T20.5: "generated from the first exchange, once, and stored") —
// no second model call, the same principle as this file's interpretingLine
// in chat.go. Whitespace is collapsed so a pasted multi-line question reads
// as one line. Only ever consulted by the winning side of saveMessages'
// on-conflict insert, so the caller never needs to know whether this is
// actually the conversation's first message — the constraint enforces
// "once" for free.
func titleFromMessage(text string) string {
	collapsed := strings.Join(strings.Fields(text), " ")
	if collapsed == "" {
		return defaultConversationTitle
	}
	runes := []rune(collapsed)
	if len(runes) <= maxTitleLength {
		return collapsed
	}
	return string(trimToGraphemeBoundary(runes[:maxTitleLength])) + "…"
}

// trimToGraphemeBoundary drops trailing rune(s) that titleFromMessage's
// plain rune-count cutoff could otherwise leave dangling. Suffix truncation
// (keep the first N runes) can only ever drop what comes *after* a kept
// rune — so a combining mark, variation selector, or skin-tone modifier is
// never stranded without its base: whichever one is last, the base right
// before it (lower index) is always still there, and the result is always a
// complete, if sometimes less-decorated, character. That leaves exactly two
// genuine bisection hazards:
//   - a trailing zero-width joiner, which joins *forward* to whatever
//     would-be next emoji got cut away, leaving a dangling join; and
//   - an odd-length trailing run of regional-indicator letters — each flag
//     is an unordered pair (e.g. "US" is 🇺 U+1F1FA + 🇸 U+1F1F8), so an odd
//     count means the last one is half a flag.
//
// Not a full UAX #29 grapheme-cluster implementation — just these two.
func trimToGraphemeBoundary(runes []rune) []rune {
	if len(runes) > 0 && runes[len(runes)-1] == 0x200D {
		runes = runes[:len(runes)-1]
	}
	trailingFlagHalves := 0
	for i := len(runes) - 1; i >= 0 && runes[i] >= 0x1F1E6 && runes[i] <= 0x1F1FF; i-- {
		trailingFlagHalves++
	}
	if trailingFlagHalves%2 == 1 {
		runes = runes[:len(runes)-1]
	}
	return runes
}

// loadConversation returns enough of conversationID's history to seed a
// fresh WS connection's in-memory context (TASKS.md T20), or nil for a
// conversation with nothing saved yet — a brand-new "new chat" id and a
// foreign id both resolve the same way, via RLS filtering fetchRecentMessages
// down to zero rows. Capped at windowHistory's own window, not
// maxStoredHistoryMessages: nothing else ever reads more of this than
// windowHistory(history) forwards to the agent, so loading further would be
// discarded work on every connect or conversation switch.
func loadConversation(db *pgxpool.Pool) func(ctx context.Context, userID, conversationID string) ([]historyTurn, error) {
	return func(ctx context.Context, userID, conversationID string) ([]historyTurn, error) {
		var history []historyTurn
		err := withUser(ctx, db, userID, func(tx pgx.Tx) error {
			var err error
			history, err = fetchRecentMessages(ctx, tx, conversationID, maxHistoryExchanges*2)
			return err
		})
		return history, err
	}
}

// loadConversationTurns pairs one conversation's stored messages into
// completed exchanges for GET /api/chat/history/{conversationID}. A foreign
// or nonexistent id resolves to an empty slice, not an error — RLS's
// message_isolation policy filters rows by conversation ownership before
// this ever sees them, so there's nothing here to distinguish "not yours"
// from "no messages yet," and no reason to.
func loadConversationTurns(db *pgxpool.Pool) func(ctx context.Context, userID, conversationID string) ([]conversationTurn, error) {
	return func(ctx context.Context, userID, conversationID string) ([]conversationTurn, error) {
		turns := []conversationTurn{}
		err := withUser(ctx, db, userID, func(tx pgx.Tx) error {
			flat, err := fetchRecentMessages(ctx, tx, conversationID, maxStoredHistoryMessages)
			if err != nil {
				return err
			}
			turns = pairTurns(flat)
			return nil
		})
		return turns, err
	}
}

// saveMessages persists one completed turn against conversationID, lazily
// creating that conversation row on first use rather than at WS-connect or
// "new chat" time — the same lazy-provisioning shape as upsertUser, so
// opening the app or clicking "New chat" without ever sending a message
// leaves no orphaned row for the sidebar to show. Reports whether this call
// is the one that actually created conversationID's row — RowsAffected() on
// the conflict-checked insert, the same outcome-bool idiom deleteConversation
// uses below — so chat.go's finishTurn can tell the browser exactly once,
// right after the row genuinely exists, instead of inferring "there's
// probably something new to show" from the WS event sequence. Only
// meaningful when err is nil: a later failure in this same transaction (the
// messages insert) rolls back the conversations insert too, so a caller must
// not trust created on a non-nil error.
//
// conversationID is always supplied by the caller (client-generated — see
// web/src/lib/chat-socket.ts) rather than discovered here, so — unlike T20's
// design — this never needs to hand an id back to its caller and never needs
// RETURNING. The create is `on conflict (id) do nothing`: a second, third,
// ... message against the same conversation simply no-ops the conversations
// insert (title is set once, from the first, and created is then false) and
// proceeds straight to the messages insert below.
//
// A conversationID belonging to another user fails closed, not silently: the
// conversations insert's DO NOTHING never touches that pre-existing row, but
// the messages insert immediately after is scoped by message_isolation's
// `with check` (conversation_id must resolve through a conversations row
// this session can see) and is rejected — the whole transaction rolls back
// and this returns an error, same handling as any other database failure. A
// random client-generated UUID isn't a guessable oracle, and this is a
// materially different trust profile than turn ids (never persisted, never a
// primary key elsewhere).
func saveMessages(db *pgxpool.Pool) func(ctx context.Context, userID, conversationID, userText, assistantText string, titleRefs []agentTitleRef) (bool, error) {
	return func(ctx context.Context, userID, conversationID, userText, assistantText string, titleRefs []agentTitleRef) (bool, error) {
		// jsonb null (via a nil parameter), not the JSON literal "null" — a
		// json.Marshal of an empty/nil slice would round-trip as content, not
		// absence, for the "no picks shown" case ARCHITECTURE.md documents.
		var refsJSON []byte
		if len(titleRefs) > 0 {
			var err error
			refsJSON, err = json.Marshal(titleRefs)
			if err != nil {
				return false, err
			}
		}

		var created bool
		err := withUser(ctx, db, userID, func(tx pgx.Tx) error {
			tag, err := tx.Exec(ctx, `
				insert into conversations (id, user_id, title) values ($1, $2, $3)
				on conflict (id) do nothing`,
				conversationID, userID, titleFromMessage(userText))
			if err != nil {
				return err
			}
			created = tag.RowsAffected() > 0
			_, err = tx.Exec(ctx, `
				insert into messages (conversation_id, role, content, title_refs)
				values ($1, 'user', $2, null), ($1, 'assistant', $3, $4)`,
				conversationID, userText, assistantText, refsJSON)
			return err
		})
		return created, err
	}
}

// conversationSummary is one entry of GET /api/conversations — enough for the
// sidebar to list and link to a conversation without its message history.
type conversationSummary struct {
	ID    string `json:"id"`
	Title string `json:"title"`
}

// loadConversationSummaries lists the caller's conversations, newest first
// (TASKS.md T20.5: "the sidebar lists past conversations, newest first").
// coalesce(title, ...): every row created after this task always has a
// title (saveMessages sets it at creation, once); only a row created before
// T20.5 shipped, under T20's original one-conversation-per-user model, could
// still hold a null one.
func loadConversationSummaries(db *pgxpool.Pool) func(ctx context.Context, userID string) ([]conversationSummary, error) {
	return func(ctx context.Context, userID string) ([]conversationSummary, error) {
		summaries := []conversationSummary{}
		err := withUser(ctx, db, userID, func(tx pgx.Tx) error {
			rows, err := tx.Query(ctx, `
				select id, coalesce(title, $2) from conversations
				where user_id = $1
				-- id desc is an arbitrary tiebreaker, not a meaningful second sort key:
				-- id is a random v4 uuid (gen_random_uuid()), so this recovers no real
				-- chronological order. Without it, two conversations sharing the exact
				-- same created_at (e.g. concurrent "new chat" clicks) would have no
				-- guaranteed order, and the sidebar could reorder them on an unrelated
				-- reload.
				order by created_at desc, id desc`, userID, defaultConversationTitle)
			if err != nil {
				return err
			}
			defer rows.Close()
			for rows.Next() {
				var s conversationSummary
				if err := rows.Scan(&s.ID, &s.Title); err != nil {
					return err
				}
				summaries = append(summaries, s)
			}
			return rows.Err()
		})
		return summaries, err
	}
}

// deleteConversation removes one of the caller's conversations; messages
// cascade via the FK's `on delete cascade`. Reports whether a row was
// actually removed — the DELETE itself stays idempotent (0 or 1 rows is
// still success, matching setSubscription/setVerdict's convention), but the
// caller (deleteConversationHandler) must know the difference: only a delete
// that actually removed something has an in-flight turn worth protecting
// against, and bumping conversationDeletions for a no-op (a double-click, a
// stale UI id, a foreign id) would advance that id's generation for anyone
// else's legitimate save, since conversationDeletions is a single,
// unscoped-by-user map.
func deleteConversation(db *pgxpool.Pool) func(ctx context.Context, userID, conversationID string) (bool, error) {
	return func(ctx context.Context, userID, conversationID string) (bool, error) {
		var deleted bool
		err := withUser(ctx, db, userID, func(tx pgx.Tx) error {
			tag, err := tx.Exec(ctx, `delete from conversations where id = $1 and user_id = $2`, conversationID, userID)
			if err != nil {
				return err
			}
			deleted = tag.RowsAffected() > 0
			return nil
		})
		return deleted, err
	}
}

// chatHistory is GET /api/chat/history/{conversationID} (TASKS.md T20/T20.5)
// — the web client's mount-time hydration so opening a conversation
// re-renders its last exchanges rather than starting blank. Degrades the
// same way watchlist() and verdicts() do: a load failure is a 503, never a
// silent empty conversation.
func (h *Handler) chatHistory(w http.ResponseWriter, r *http.Request) {
	userID, _ := r.Context().Value(ctxUserID).(string)
	conversationID := r.PathValue("conversationID")
	// uuid.Parse accepts forms (e.g. a urn:uuid: prefix) that Postgres's uuid
	// column doesn't — canonicalize so a Go-valid-but-Postgres-invalid string
	// never reaches the query as a raw type-cast error (503) instead of the
	// clean 400 this check exists to give. Same fix as
	// deleteConversationHandler's below.
	parsed, err := uuid.Parse(conversationID)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid conversation id"})
		return
	}
	conversationID = parsed.String()

	turns, err := h.loadConversationTurns(r.Context(), userID, conversationID)
	if err != nil {
		slog.ErrorContext(r.Context(), "chat history load failed", "error", dbError(err))
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "temporarily unavailable"})
		return
	}
	// [] on the wire, never null — same guard as watchlist() and verdicts(),
	// here because the loader's own nil-safety (loadConversationTurns always
	// initializes turns) isn't a contract this handler should trust blindly.
	if turns == nil {
		turns = []conversationTurn{}
	}
	writeJSON(w, http.StatusOK, turns)
}

// conversations is GET /api/conversations — the sidebar's list, newest first.
func (h *Handler) conversations(w http.ResponseWriter, r *http.Request) {
	userID, _ := r.Context().Value(ctxUserID).(string)

	list, err := h.loadConversationSummaries(r.Context(), userID)
	if err != nil {
		slog.ErrorContext(r.Context(), "conversation list load failed", "error", dbError(err))
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "temporarily unavailable"})
		return
	}
	if list == nil {
		list = []conversationSummary{}
	}
	writeJSON(w, http.StatusOK, list)
}

// deleteConversationHandler is DELETE /api/conversations/{id} (TASKS.md
// T20.5: "conversations can be deleted").
func (h *Handler) deleteConversationHandler(w http.ResponseWriter, r *http.Request) {
	userID, _ := r.Context().Value(ctxUserID).(string)
	id := r.PathValue("id")
	// conversations.id is a uuid column — an invalid literal is a Postgres
	// type-cast error (503), not "nothing to delete." Reject it as a 400
	// instead, the same shape setVerdict validates mediaType/tmdbID before
	// ever reaching the database.
	parsed, err := uuid.Parse(id)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid conversation id"})
		return
	}
	// Canonicalized before it's used as a conversationDeletions key — see
	// chat.go's "message" case for why a differently-cased id for the same
	// conversation must not be treated as a different key here.
	id = parsed.String()
	deleted, err := h.deleteConversation(r.Context(), userID, id)
	if err != nil {
		slog.ErrorContext(r.Context(), "conversation delete failed", "error", dbError(err))
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "temporarily unavailable"})
		return
	}
	// Only on a real delete: a no-op (already gone, or never this caller's —
	// RLS makes a foreign id look identical) must not advance the
	// conversation's generation, which would otherwise let any caller
	// suppress another user's still-legitimate straggling turn just by
	// attempting to delete their conversation id.
	//
	// Accepted, narrow race: between h.deleteConversation's commit and this
	// bump, a straggling finishTurn on this same id could check
	// conversationDeletions.generation and see the pre-bump value, then
	// resurrect the row via saveMessages' own on-conflict-do-nothing insert.
	// Closing it would mean bumping before knowing `deleted` (violating the
	// no-op-must-not-tombstone rule above) or a cross-process synchronization
	// this single-instance, in-memory design doesn't have. The window is one
	// Go statement wide and requires exact timing against a genuine
	// straggler; not worth the complexity at this app's scale.
	if deleted {
		h.conversationDeletions.bump(id)
	}
	writeJSON(w, http.StatusOK, map[string]bool{"deleted": deleted})
}
