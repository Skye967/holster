package main

import (
	"context"
	"encoding/json"
	"errors"
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
// completed exchange. title_refs is deliberately not surfaced *to the browser*
// here: nothing yet re-renders title cards from a reload, only the text (see
// ARCHITECTURE.md's data model and its "One socket per session"). The column
// is read by fetchShownRefs alone, for the agent rather than the browser.
type conversationTurn struct {
	UserText      string `json:"user_text"`
	AssistantText string `json:"assistant_text"`
}

// fetchRecentMessages returns up to limit of one conversation's most recent
// messages, oldest-first. Shared by loadConversation (the WS-seed path) and
// loadConversationTurns (the reload-render path) — same query and scan,
// different caps. Text only: the already-shown set is fetchShownRefs' job,
// and it wants the whole conversation rather than a window of it.
//
// Joined through conversations on user_id explicitly, not left to
// RLS's message_isolation policy alone: messages carries no user_id of its
// own, so this join is the only guard a role that bypasses RLS still meets.
func fetchRecentMessages(ctx context.Context, tx pgx.Tx, userID, conversationID string, limit int) ([]historyTurn, error) {
	// Newest-first with a limit, then reversed below, so the cap keeps the
	// most recent messages rather than the oldest ones. Ordered by seq, not
	// created_at: saveMessages inserts a turn's user and assistant row in one
	// statement, so both get the exact same now() value, and two rows with a
	// tied timestamp have no guaranteed order — the identity column does.
	rows, err := tx.Query(ctx, `
		select m.role, m.content
		from messages m
		join conversations c on c.id = m.conversation_id
		where m.conversation_id = $1 and c.user_id = $2
		order by m.seq desc
		limit $3`,
		conversationID, userID, limit)
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

// fetchShownRefs returns every title id this conversation has put on screen,
// oldest-first — the whole of it, not a window, because a title re-offered is
// a visible bug and a window ages titles back into eligibility. Nothing caps
// this scan or the body it feeds but the conversation's own length;
// agent/chat.py's MAX_SHOWN trims only what reaches the search, after the
// whole set has crossed the wire.
//
// Rows with a null title_refs are filtered in SQL rather than skipped in Go:
// a user row and an assistant row that showed no cards are most of a
// conversation, and neither carries anything to decode.
//
// Joined through conversations on user_id for the same reason
// fetchRecentMessages is — messages carries no user_id of its own.
//
// What the column cannot record: whether a turn was a recommendation or a
// title lookup. A guest's set keeps that distinction on the connection
// (chat.go's turnRecord.countsAsShown) and a lookup there feeds nothing, so
// for an account, asking "is Heat on Netflix" bars Heat from every later
// recommendation in that conversation, however long it runs. Accepted because
// it is still one title against many and the fix is a column on messages, not
// a heuristic on the stored text.
func fetchShownRefs(ctx context.Context, tx pgx.Tx, userID, conversationID string) ([]agentTitleRef, error) {
	rows, err := tx.Query(ctx, `
		select m.title_refs
		from messages m
		join conversations c on c.id = m.conversation_id
		where m.conversation_id = $1 and c.user_id = $2
			and m.title_refs is not null
		order by m.seq`,
		conversationID, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var refs []agentTitleRef
	for rows.Next() {
		var refsJSON []byte
		if err := rows.Scan(&refsJSON); err != nil {
			return nil, err
		}
		var turnRefs []agentTitleRef
		if err := json.Unmarshal(refsJSON, &turnRefs); err != nil {
			// Logged, not fatal: losing one turn's refs costs a repeated
			// title, where failing the load costs the window and every other
			// turn's refs with it.
			slog.WarnContext(ctx, "message title_refs decode failed",
				"conversation", conversationID, "error", err.Error())
			continue
		}
		refs = append(refs, turnRefs...)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return refs, nil
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
// coalesce fallback for a null title share this one constant rather than each
// hardcoding "New chat".
const defaultConversationTitle = "New chat"

// titleFromMessage derives a conversation's title from its opening message —
// generated from the first exchange, once, and stored, with no second model
// call, the same principle as chat.go's interpretingLine. Whitespace is
// collapsed so a pasted multi-line question reads
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
// plain rune-count cutoff could leave dangling. Suffix truncation (keep the
// first N runes) can only ever drop what comes *after* a kept rune, so a
// combining mark, variation selector or skin-tone modifier is never stranded
// without its base: whichever one is last, the base before it is still there.
// That argument holds only for a suffix cut. The two hazards handled here:
//   - a trailing zero-width joiner, which joins *forward* to the emoji that
//     got cut away; and
//   - an odd-length trailing run of regional-indicator letters — each flag is
//     a pair (🇺 U+1F1FA + 🇸 U+1F1F8), so an odd count is half a flag.
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

// loadConversation returns what a fresh WS connection needs of
// conversationID: the recent history to seed its context, and every title it
// has already put on screen. Both are nil for a conversation with nothing
// saved yet — a brand-new "new chat" id and a foreign id resolve the same
// way, via each query's explicit join and RLS independently filtering down to
// zero rows.
//
// Two reads with different appetites, in one transaction: the history is
// capped at maxHistoryMessages (rather than maxStoredHistoryMessages — the
// agent is sent a window, not a transcript), and the shown set is the whole
// conversation.
//
// History first, for which way the read-committed gap falls: each statement
// takes its own snapshot, so a turn committed between the two lands in the
// refs, where it is suppressed, rather than in the window alone, where its
// cards would be offered again.
//
// Either read failing loses both — withUser rolls the transaction back and
// returns the error — so the caller answers with no history and may repeat a
// title, the same trade the single query made before.
func loadConversation(db *pgxpool.Pool) func(ctx context.Context, userID, conversationID string) ([]historyTurn, []agentTitleRef, error) {
	return func(ctx context.Context, userID, conversationID string) ([]historyTurn, []agentTitleRef, error) {
		var history []historyTurn
		var shown []agentTitleRef
		err := withUser(ctx, db, userID, func(tx pgx.Tx) error {
			var err error
			history, err = fetchRecentMessages(ctx, tx, userID, conversationID, maxHistoryMessages)
			if err != nil {
				return err
			}
			shown, err = fetchShownRefs(ctx, tx, userID, conversationID)
			return err
		})
		if err != nil {
			return nil, nil, err
		}
		return history, shown, nil
	}
}

// loadConversationTurns pairs one conversation's stored messages into
// completed exchanges for GET /api/chat/history/{conversationID}, and reports
// whether the conversation exists from this caller's point of view. A foreign
// id and a nonexistent one both read as false, deliberately: the user_id
// filter below and conversation_isolation's RLS policy each make another
// user's row invisible rather than merely filtered, so chatHistory 404s both
// rather than pretending to tell them apart.
func loadConversationTurns(db *pgxpool.Pool) func(ctx context.Context, userID, conversationID string) ([]conversationTurn, bool, error) {
	return func(ctx context.Context, userID, conversationID string) ([]conversationTurn, bool, error) {
		turns := []conversationTurn{}
		var exists bool
		err := withUser(ctx, db, userID, func(tx pgx.Tx) error {
			if err := tx.QueryRow(ctx,
				`select exists(select 1 from conversations where id = $1 and user_id = $2)`,
				conversationID, userID,
			).Scan(&exists); err != nil {
				return err
			}
			if !exists {
				return nil
			}
			flat, err := fetchRecentMessages(ctx, tx, userID, conversationID, maxStoredHistoryMessages)
			if err != nil {
				return err
			}
			turns = pairTurns(flat)
			return nil
		})
		return turns, exists, err
	}
}

// saveMessages persists one completed turn against conversationID, creating
// that conversation row lazily on first use so opening the app or clicking
// "New chat" without sending a message leaves no orphaned row in the sidebar.
//
// Reports whether this call created the row, so finishTurn can tell the
// browser exactly once. Only meaningful when err is nil: the messages insert
// failing rolls the conversations insert back too.
//
// conversationID is minted by the browser before the socket frame is sent —
// by the /chat routes and the sidebar's "New chat" (web/src/app/(app)/chat/
// page.tsx, [id]/page.tsx, components/app-sidebar.tsx), never here — so no
// RETURNING is needed. A random client-chosen UUID is safe as the primary key
// because it is not a guessable oracle: the create is `on conflict (id) do
// nothing`, so a colliding id is absorbed rather than raising, and later
// messages against the same conversation no-op it — the title is set once,
// from the first. What rejects a *foreign* id is the messages insert below,
// which joins through conversations on user_id.
//
// A conversationID belonging to another user fails closed: DO NOTHING leaves
// the foreign row untouched, and the messages insert joins through
// conversations on user_id, so it matches nothing and the RowsAffected
// check below raises errConversationNotWritable. message_isolation's RLS
// policy is a second, independent layer over the same tables.
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

			msgTag, err := tx.Exec(ctx, `
				insert into messages (conversation_id, role, content, title_refs)
				select $1, v.role, v.content, v.title_refs
				from (values ('user', $2::text, null::jsonb), ('assistant', $3::text, $4::jsonb))
					as v(role, content, title_refs)
				where exists (select 1 from conversations where id = $1 and user_id = $5)`,
				conversationID, userText, assistantText, refsJSON, userID)
			if err != nil {
				return err
			}
			// The exists() guard doesn't vary per row, so this can only ever
			// insert both rows or neither.
			if msgTag.RowsAffected() == 0 {
				return errConversationNotWritable
			}
			return nil
		})
		return created, err
	}
}

// errConversationNotWritable is saveMessages' own signal that conversationID
// doesn't belong to this caller — the explicit join finding no matching row,
// independent of RLS. chat.go's finishTurn treats this the same as
// isRLSRejection (main.go): the one persist failure the browser must be told
// about rather than only logged, since a turn always ends in a terminal frame
// (ARCHITECTURE.md).
var errConversationNotWritable = errors.New("conversation does not belong to the caller")

// conversationSummary is one entry of GET /api/conversations — enough for the
// sidebar to list and link to a conversation without its message history.
type conversationSummary struct {
	ID    string `json:"id"`
	Title string `json:"title"`
}

// loadConversationSummaries lists the caller's conversations, newest first
// — what the sidebar lists. coalesce(title, ...) covers rows predating titles:
// saveMessages sets one at creation, so only older rows can hold a null.
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

// chatHistory is GET /api/chat/history/{conversationID} — the web client's
// mount-time hydration so opening a conversation
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

	turns, exists, err := h.loadConversationTurns(r.Context(), userID, conversationID)
	if err != nil {
		slog.ErrorContext(r.Context(), "chat history load failed", "error", dbError(err))
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "temporarily unavailable"})
		return
	}
	// Foreign and nonexistent are the same outcome here — see
	// loadConversationTurns — so both 404, not 200 []. Safe for a brand-new
	// "New chat" id's first hydration too: chat-history.ts's fetchChatHistory
	// throws on any non-2xx, and chat-panel.tsx already treats that failure
	// identically to an empty thread.
	if !exists {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "conversation not found"})
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

// deleteConversationHandler is DELETE /api/conversations/{id}.
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
	// Bumped after the commit, which leaves one statement in which a
	// straggling finishTurn can read the pre-bump generation and recreate the
	// row it just deleted. Closing it means either bumping before `deleted` is
	// known — which would tombstone a no-op delete, the case above — or
	// cross-process synchronisation this single-instance design has no way to
	// do. Accepted: the window needs exact timing against a real straggler.
	if deleted {
		h.conversationDeletions.bump(id)
	}
	writeJSON(w, http.StatusOK, map[string]bool{"deleted": deleted})
}
