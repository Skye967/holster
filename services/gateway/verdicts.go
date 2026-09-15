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
// context — so the two can never disagree about what a user has judged.
//
// Newest-first because the agent keeps only the first few liked titles for its
// taste hint (catalog_tool.py's MAX_TASTE_TITLES) and carries no timestamp to
// re-sort by. Ordered by verdict_set_at, not created_at: saveVerdict's
// conflict clause refreshes it whenever the verdict changes, so a title
// bookmarked in January and liked today sorts as today. created_at stays an
// untouched "first saved" column (see watchlist.go, which orders by it). The
// tie-break is insurance for a batched write or backfill — saveVerdict writes
// one row per transaction, so real timestamps are distinct.
//
// Uncapped, because the agent needs every row to answer "has this user judged
// this title" — a LIMIT would quietly expire that guarantee. The sort is not
// index-covered: title_verdicts carries only its primary key
// (user_id, tmdb_id, media_type), so this reads the user's rows and sorts them
// in memory. Fine at these sizes; a (user_id, verdict_set_at desc) index is the
// fix when it stops being — not a LIMIT, which the paragraph above rules out.
func loadVerdicts(db *pgxpool.Pool) func(ctx context.Context, userID string) ([]Verdict, error) {
	return func(ctx context.Context, userID string) ([]Verdict, error) {
		var verdicts []Verdict
		err := withUser(ctx, db, userID, func(tx pgx.Tx) error {
			rows, err := tx.Query(ctx,
				`select tmdb_id, media_type, verdict from title_verdicts
				 where user_id = $1
				 order by verdict_set_at desc, media_type, tmdb_id`,
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

// errVerdictLocked is returned by saveVerdict when a PUT would change one of
// the four judgments directly to a *different* value after it's already set
// — title_verdicts' migration comment (20260903002450_verdicts.sql) states
// those "never change" once set and assigns enforcing that to "whichever task
// builds the write path." Only want_to_watch is exempt: it is an intention
// with a lifecycle, expected to become seen later, never the reverse. A locked
// judgment can still be *cleared* via
// DELETE and re-set (ARCHITECTURE.md's "A locked judgment can be cleared"): this
// error is only about overwriting one verdict with another in a single step.
var errVerdictLocked = errors.New("verdict already set and cannot change")

// errVerdictStale is returned by saveVerdict when a DELETE's ?expect doesn't
// match because the row now holds a *different* verdict (not because it's
// already gone, which stays a silent no-op — see saveVerdict's own comment).
// Left unreported, the caller's optimistic "cleared" UI would silently
// disagree with the server for the rest of the session, surfacing later only
// as a confusing 409 on an unrelated PUT.
var errVerdictStale = errors.New("verdict no longer matches what the caller expected")

// saveVerdict mirrors saveSubscription's shape (providers.go): one function
// branching on state rather than a separate upsert/delete pair. A nil verdict
// deletes the row; otherwise it upserts. Both are idempotent, run inside
// withUser so verdict_isolation's RLS policy
// (20260903002450_verdicts.sql) applies.
func saveVerdict(db *pgxpool.Pool) func(ctx context.Context, userID string, tmdbID int, mediaType string, verdict *string, expectedVerdict string) error {
	return func(ctx context.Context, userID string, tmdbID int, mediaType string, verdict *string, expectedVerdict string) error {
		return withUser(ctx, db, userID, func(tx pgx.Tx) error {
			if verdict == nil {
				// Compare-and-delete, not an unconditional clear: only
				// removes the row when it still holds the exact verdict the
				// caller believes is there — the same idiom the upsert below
				// uses. A stale client (a second tab that hasn't seen a write
				// made elsewhere) matches nothing and is a safe no-op.
				//
				// FOR UPDATE, not a bare SELECT then DELETE: it locks the row
				// (if one exists) for the rest of this transaction, so no
				// concurrent write can land between the read and the decision
				// below — the same atomicity the upsert's single
				// INSERT..ON CONFLICT statement gets for free, made explicit
				// here since a DELETE has no equivalent one-statement form.
				var current string
				err := tx.QueryRow(ctx,
					`select verdict from title_verdicts
					 where user_id = $1 and tmdb_id = $2 and media_type = $3
					 for update`,
					userID, tmdbID, mediaType).Scan(&current)
				if errors.Is(err, pgx.ErrNoRows) {
					// Already gone — a double-click or a retried request,
					// matching setSubscription's idempotent DELETE convention.
					return nil
				}
				if err != nil {
					return err
				}
				if current != expectedVerdict {
					// Still there, holding something else: the caller's local
					// state is stale. Silently returning success here would
					// let the browser's optimistic "cleared" UI drift from
					// the server forever, with no refetch to correct it.
					return errVerdictStale
				}
				_, err = tx.Exec(ctx,
					`delete from title_verdicts
					 where user_id = $1 and tmdb_id = $2 and media_type = $3`,
					userID, tmdbID, mediaType)
				return err
			}

			// The update side of the upsert only applies when the existing
			// row is still want_to_watch (the one mutable value) or already
			// holds the requested verdict (an idempotent no-op) — once a row
			// holds a different judgment, this WHERE excludes it and
			// RowsAffected is 0, which the caller maps to a 409. Setting a
			// *different* verdict over a locked one is rejected: the caller
			// must clear it first via DELETE.
			//
			// verdict_set_at (20260904154626_verdict_set_at.sql) only
			// refreshes when the verdict actually changes, not on the
			// idempotent-resubmit branch above — a duplicate PUT (a client
			// retry after a timed-out-but-succeeded request, say) must not
			// bump a title's recency and jump it ahead of a genuinely more
			// recent one in loadVerdicts' ordering.
			tag, err := tx.Exec(ctx, `
				insert into title_verdicts (user_id, tmdb_id, media_type, verdict)
				values ($1, $2, $3, $4)
				on conflict (user_id, tmdb_id, media_type) do update
					set verdict = excluded.verdict,
						verdict_set_at = case
							when title_verdicts.verdict != excluded.verdict then now()
							else title_verdicts.verdict_set_at
						end
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
// given verdict (body: {"verdict": "..."}); DELETE clears it, but only when
// it still matches the caller's required ?expect=<verdict> — both idempotent,
// mirroring setSubscription (providers.go). mediaType comes first in the path
// to match TMDB's own convention (agent/tmdb.py's /{media_type}/{id} calls),
// not the primary key's column order.
//
// ?expect is required on DELETE, not optional: every real caller
// (title-card.tsx's bookmark and its "Clear rating" item) already knows
// exactly what verdict it's showing, and validating it the same way PUT's
// body is validated closes the door on an unscoped "clear whatever's there"
// call — see saveVerdict's comment for why that scoping is what keeps a
// stale client safe.
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
	var expectedVerdict string
	if r.Method == http.MethodPut {
		var body setVerdictBody
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil || !verdictValues[body.Verdict] {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid verdict"})
			return
		}
		verdict = &body.Verdict
	} else {
		expectedVerdict = r.URL.Query().Get("expect")
		if !verdictValues[expectedVerdict] {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "missing or invalid expect"})
			return
		}
	}

	if err := h.saveVerdict(r.Context(), userID, tmdbID, mediaType, verdict, expectedVerdict); err != nil {
		if errors.Is(err, errVerdictLocked) {
			// Not routine like an expired token, but not a server failure
			// either. Clearing a locked judgment *is* a normal UI path
			// (title-card.tsx's "Clear rating") — this only fires on
			// a PUT attempting to overwrite a locked judgment with a
			// different one directly, which the UI still disables, so
			// reaching here means a stale client or a direct API call.
			slog.WarnContext(r.Context(), "verdict write rejected: already locked",
				"tmdb_id", tmdbID, "media_type", mediaType)
			// code is the machine-readable discriminant: two different
			// errors both 409, and a client that infers which one from the
			// HTTP method alone (PUT vs DELETE) breaks the moment either
			// path grows a second cause. code is stable across wording
			// changes to error, which is meant for logs/humans, not dispatch.
			writeJSON(w, http.StatusConflict, map[string]string{"error": "verdict already set", "code": "verdict_locked"})
			return
		}
		if errors.Is(err, errVerdictStale) {
			// Distinct from errVerdictLocked's 409: this is DELETE finding a
			// row that's still there but no longer matches ?expect — the
			// caller's local state is stale, not blocked by a lock. Worth a
			// log (unlike the ordinary already-gone no-op) since it means an
			// optimistic UI was about to silently disagree with the server.
			slog.WarnContext(r.Context(), "verdict clear rejected: no longer matches expected value",
				"tmdb_id", tmdbID, "media_type", mediaType)
			writeJSON(w, http.StatusConflict, map[string]string{"error": "verdict has changed", "code": "verdict_stale"})
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
