package main

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"slices"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// maxStoredHistoryMessages bounds how much of a conversation GET
// /api/chat/history returns — the reload-render path, which shows the user
// more than the agent ever needs (loadConversation, the WS-seed path, uses
// its own much smaller cap; see there). Until T20.5 ships "new chat," one
// conversation is the whole account's history and grows without the implicit
// bound a browser tab's lifetime used to give it (T20.5's own premise: "one
// endless thread ... makes answers worse"), so this stands in as the bound
// in the meantime.
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

// conversationIDForUser returns the caller's conversation id, or "" if none
// exists yet. `unique (user_id)` (20260904191046_conversations.sql) makes
// this a singleton lookup, not a "pick the latest" query.
func conversationIDForUser(ctx context.Context, tx pgx.Tx, userID string) (string, error) {
	var id string
	err := tx.QueryRow(ctx, `select id from conversations where user_id = $1`, userID).Scan(&id)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", nil
	}
	return id, err
}

// loadConversation returns the caller's conversation id and enough of its
// history to seed a fresh WS connection's in-memory context (TASKS.md T20),
// or ("", nil, nil) for a caller with no conversation yet. Capped at
// windowHistory's own window, not maxStoredHistoryMessages: nothing else
// ever reads more of this than windowHistory(history) forwards to the agent,
// so loading further would be discarded work on every connect.
func loadConversation(db *pgxpool.Pool) func(ctx context.Context, userID string) (string, []historyTurn, error) {
	return func(ctx context.Context, userID string) (string, []historyTurn, error) {
		var conversationID string
		var history []historyTurn
		err := withUser(ctx, db, userID, func(tx pgx.Tx) error {
			id, err := conversationIDForUser(ctx, tx, userID)
			if err != nil || id == "" {
				return err
			}
			conversationID = id
			history, err = fetchRecentMessages(ctx, tx, id, maxHistoryExchanges*2)
			return err
		})
		return conversationID, history, err
	}
}

// loadConversationTurns pairs the caller's stored messages into completed
// exchanges for GET /api/chat/history.
func loadConversationTurns(db *pgxpool.Pool) func(ctx context.Context, userID string) ([]conversationTurn, error) {
	return func(ctx context.Context, userID string) ([]conversationTurn, error) {
		turns := []conversationTurn{}
		err := withUser(ctx, db, userID, func(tx pgx.Tx) error {
			id, err := conversationIDForUser(ctx, tx, userID)
			if err != nil || id == "" {
				return err
			}
			flat, err := fetchRecentMessages(ctx, tx, id, maxStoredHistoryMessages)
			if err != nil {
				return err
			}
			turns = pairTurns(flat)
			return nil
		})
		return turns, err
	}
}

// saveMessages persists one completed turn, lazily creating the caller's
// conversation on first use rather than at WS-connect time — the same
// lazy-provisioning shape as upsertUser and providers.go's cache refresh, so
// opening the app without sending a message leaves no orphaned conversation
// row for T20.5's future list to show.
//
// The create is an upsert (`on conflict (user_id) do update ... returning
// id`), not a plain insert: two callers racing to create the caller's first-
// ever conversation (two concurrent WS connections, or a genuine
// loadConversation failure that left conversationID empty for an existing
// user — see runChatConnection) both converge on the single row `unique
// (user_id)` allows, rather than one of them failing outright. Do not
// simplify this to `do nothing returning id` — on the losing side of a
// conflict that returns zero rows, not the existing row's id, and this
// function would then hand back an empty conversation id having done
// nothing.
//
// Returns the conversation id the pair was written to, unchanged from the
// input on error — the insert and any lazy create share one transaction via
// withUser, so a failure here rolls both back and callers must not adopt a
// rolled-back id.
func saveMessages(db *pgxpool.Pool) func(ctx context.Context, userID, conversationID, userText, assistantText string, titleRefs []agentTitleRef) (string, error) {
	return func(ctx context.Context, userID, conversationID, userText, assistantText string, titleRefs []agentTitleRef) (string, error) {
		// jsonb null (via a nil parameter), not the JSON literal "null" — a
		// json.Marshal of an empty/nil slice would round-trip as content, not
		// absence, for the "no picks shown" case ARCHITECTURE.md documents.
		var refsJSON []byte
		if len(titleRefs) > 0 {
			var err error
			refsJSON, err = json.Marshal(titleRefs)
			if err != nil {
				return conversationID, err
			}
		}

		id := conversationID
		err := withUser(ctx, db, userID, func(tx pgx.Tx) error {
			if id == "" {
				if err := tx.QueryRow(ctx, `
					insert into conversations (user_id) values ($1)
					on conflict (user_id) do update set user_id = excluded.user_id
					returning id`,
					userID).Scan(&id); err != nil {
					return err
				}
			}
			_, err := tx.Exec(ctx, `
				insert into messages (conversation_id, role, content, title_refs)
				values ($1, 'user', $2, null), ($1, 'assistant', $3, $4)`,
				id, userText, assistantText, refsJSON)
			return err
		})
		if err != nil {
			return conversationID, err
		}
		return id, nil
	}
}

// chatHistory is GET /api/chat/history (TASKS.md T20) — the web client's
// mount-time hydration so a page reload re-renders the last exchanges rather
// than starting blank. Degrades the same way watchlist() and verdicts() do:
// a load failure is a 503, never a silent empty conversation.
func (h *Handler) chatHistory(w http.ResponseWriter, r *http.Request) {
	userID, _ := r.Context().Value(ctxUserID).(string)

	turns, err := h.loadConversationTurns(r.Context(), userID)
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
