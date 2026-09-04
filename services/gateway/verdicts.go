package main

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"strconv"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// mediaTypeValues and verdictValues mirror the check constraints on
// title_verdicts (supabase/migrations/20260903002450_verdicts.sql) — validated
// here too so a bad value 400s before it ever reaches the database.
var (
	mediaTypeValues = map[string]bool{"movie": true, "tv": true}
	verdictValues   = map[string]bool{
		"liked": true, "disliked": true, "seen": true,
		"not_interested": true, "want_to_watch": true,
	}
)

// Verdict is one row of title_verdicts, used for two things: the response
// body of GET /api/verdicts — the caller's whole set, so the chat UI can
// hydrate every title card's saved/judged state in one round trip instead of
// one request per card — and, unchanged, the shape the gateway hands the
// agent with each message (chat.go's runTurn fills agentChatRequest.Verdicts
// straight from loadVerdicts below; deliberately not via chatContext — see
// that type's comment). One type rather than two: same table, same three
// columns, and the agent's TitleVerdict (agent/catalog_tool.py) decodes
// these exact JSON names.
type Verdict struct {
	TMDBID    int    `json:"tmdb_id"`
	MediaType string `json:"media_type"`
	Verdict   string `json:"verdict"`
}

// loadVerdicts mirrors loadProviders' shape (providers.go): a plain function
// closed over the pool, injected into Handler so it's fakeable in tests. One
// loader for both consumers — the browser's hydration and the agent's chat
// context (T19) — so the two can never disagree about what a user has judged.
//
// Newest-first because the agent keeps only the first few liked titles for
// its taste hint (catalog_tool.py's MAX_TASTE_TITLES) and carries no
// timestamp to re-sort by; /api/verdicts is indifferent. created_at is a
// proxy, not the truth — saveVerdict's conflict clause below leaves it
// alone, so a title bookmarked in January and liked today still sorts as
// January. Exact for a judgment set directly on a title, stale only for the
// want_to_watch -> liked lifecycle. saveVerdict writes one row per
// transaction, so production timestamps are distinct and the tie-break is
// only insurance - it keeps the order total if a batched write or a backfill
// ever makes them tie, which is what db_test.go exercises.
//
// Uncapped, because the agent needs every row to answer "has this user judged
// this title" - a LIMIT would quietly expire T19's guarantee once someone had
// judged enough titles. Note the sort is not index-covered: title_verdicts
// has only its primary key (user_id, tmdb_id, media_type), so this fetches
// the user's rows and sorts them. Fine at these sizes; a
// (user_id, created_at desc) index is the fix if it stops being.
func loadVerdicts(db *pgxpool.Pool) func(ctx context.Context, userID string) ([]Verdict, error) {
	return func(ctx context.Context, userID string) ([]Verdict, error) {
		var verdicts []Verdict
		err := withUser(ctx, db, userID, func(tx pgx.Tx) error {
			rows, err := tx.Query(ctx,
				`select tmdb_id, media_type, verdict from title_verdicts
				 where user_id = $1
				 order by created_at desc, media_type, tmdb_id`,
				userID)
			if err != nil {
				return err
			}
			defer rows.Close()
			for rows.Next() {
				var v Verdict
				if err := rows.Scan(&v.TMDBID, &v.MediaType, &v.Verdict); err != nil {
					return err
				}
				verdicts = append(verdicts, v)
			}
			return rows.Err()
		})
		return verdicts, err
	}
}

// errVerdictLocked is returned by saveVerdict when the write would change one
// of the four judgments after it's already set — title_verdicts' migration
// comment (20260903002450_verdicts.sql) states those "never change" once set
// and assigns enforcing that to "whichever task builds the write path." Only
// want_to_watch is exempt: TASKS.md's T17 entry describes it as "an intention
// with a lifecycle, expected to become seen later," never the reverse.
var errVerdictLocked = errors.New("verdict already set and cannot change")

// saveVerdict mirrors saveSubscription's shape (providers.go): one function
// branching on state rather than a separate upsert/delete pair. A nil verdict
// deletes the row; otherwise it upserts. Both are idempotent, run inside
// withUser so verdict_isolation's RLS policy
// (20260903002450_verdicts.sql) applies.
func saveVerdict(db *pgxpool.Pool) func(ctx context.Context, userID string, tmdbID int, mediaType string, verdict *string) error {
	return func(ctx context.Context, userID string, tmdbID int, mediaType string, verdict *string) error {
		return withUser(ctx, db, userID, func(tx pgx.Tx) error {
			if verdict == nil {
				// A locked judgment isn't deletable either — the only
				// legitimate caller (the bookmark's clear button) only ever
				// fires when the stored verdict is want_to_watch, so this
				// guard only closes the raw-API loophole. Idempotent either
				// way, matching setSubscription's DELETE convention: 0 rows
				// affected (nothing to delete, or a locked row) is success.
				_, err := tx.Exec(ctx,
					`delete from title_verdicts
					 where user_id = $1 and tmdb_id = $2 and media_type = $3
					   and verdict = 'want_to_watch'`,
					userID, tmdbID, mediaType)
				return err
			}

			// The update side of the upsert only applies when the existing
			// row is still want_to_watch (the one mutable value) or already
			// holds the requested verdict (an idempotent no-op) — once a row
			// holds a different judgment, this WHERE excludes it and
			// RowsAffected is 0, which the caller maps to a 409.
			tag, err := tx.Exec(ctx, `
				insert into title_verdicts (user_id, tmdb_id, media_type, verdict)
				values ($1, $2, $3, $4)
				on conflict (user_id, tmdb_id, media_type) do update
					set verdict = excluded.verdict
					where title_verdicts.verdict = 'want_to_watch'
					   or title_verdicts.verdict = excluded.verdict`,
				userID, tmdbID, mediaType, *verdict)
			if err != nil {
				return err
			}
			if tag.RowsAffected() == 0 {
				return errVerdictLocked
			}
			return nil
		})
	}
}

// verdicts is GET /api/verdicts — every verdict the caller has set, so the
// chat UI can hydrate title cards without a request per card.
func (h *Handler) verdicts(w http.ResponseWriter, r *http.Request) {
	userID, _ := r.Context().Value(ctxUserID).(string)

	vs, err := h.loadVerdicts(r.Context(), userID)
	if err != nil {
		slog.ErrorContext(r.Context(), "verdict load failed", "error", dbError(err))
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "temporarily unavailable"})
		return
	}
	if vs == nil {
		vs = []Verdict{}
	}
	writeJSON(w, http.StatusOK, vs)
}

// setVerdictBody is PUT /api/verdicts/{mediaType}/{tmdbID}'s request body.
type setVerdictBody struct {
	Verdict string `json:"verdict"`
}

// setVerdict is PUT/DELETE /api/verdicts/{mediaType}/{tmdbID}. PUT upserts the
// given verdict (body: {"verdict": "..."}); DELETE clears whatever verdict is
// stored, if any — both idempotent, mirroring setSubscription (providers.go).
// mediaType comes first in the path to match TMDB's own convention
// (agent/tmdb.py's /{media_type}/{id} calls), not the primary key's column
// order.
func (h *Handler) setVerdict(w http.ResponseWriter, r *http.Request) {
	userID, _ := r.Context().Value(ctxUserID).(string)

	mediaType := r.PathValue("mediaType")
	if !mediaTypeValues[mediaType] {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid media type"})
		return
	}
	tmdbID, err := strconv.Atoi(r.PathValue("tmdbID"))
	if err != nil || tmdbID <= 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid tmdb id"})
		return
	}

	var verdict *string
	if r.Method == http.MethodPut {
		var body setVerdictBody
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil || !verdictValues[body.Verdict] {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid verdict"})
			return
		}
		verdict = &body.Verdict
	}

	if err := h.saveVerdict(r.Context(), userID, tmdbID, mediaType, verdict); err != nil {
		if errors.Is(err, errVerdictLocked) {
			// Not routine like an expired token, but not a server failure
			// either — the normal UI path never attempts this (title-card.tsx
			// disables both controls once a judgment is locked), so reaching
			// here means a stale client or a direct API call.
			slog.WarnContext(r.Context(), "verdict write rejected: already locked",
				"tmdb_id", tmdbID, "media_type", mediaType)
			writeJSON(w, http.StatusConflict, map[string]string{"error": "verdict already set"})
			return
		}
		slog.ErrorContext(r.Context(), "verdict write failed", "error", dbError(err))
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "temporarily unavailable"})
		return
	}

	result := ""
	if verdict != nil {
		result = *verdict
	}
	writeJSON(w, http.StatusOK, map[string]string{"verdict": result})
}
