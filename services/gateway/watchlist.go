package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// watchlistTimeout bounds the gateway's call to the agent's POST /titles —
// same reasoning as providersRefreshTimeout (providers.go), but higher: the
// agent runs one concurrent title+availability fetch pair per item, so
// worst-case latency tracks the slowest single item's TMDB retry wait, not
// providersRefreshTimeout's two-calls-total case.
var watchlistTimeout = 20 * time.Second

// watchlistItem is one title_verdicts row scoped to want_to_watch — the DB
// half of "saved" (T17's migration comment describes want_to_watch as "an
// intention with a lifecycle," the read this task is built on).
type watchlistItem struct {
	TMDBID    int
	MediaType string
}

// loadWatchlistItems mirrors loadVerdicts' shape (verdicts.go): every
// want_to_watch row for the caller, newest first.
func loadWatchlistItems(db *pgxpool.Pool) func(ctx context.Context, userID string) ([]watchlistItem, error) {
	return func(ctx context.Context, userID string) ([]watchlistItem, error) {
		var items []watchlistItem
		err := withUser(ctx, db, userID, func(tx pgx.Tx) error {
			rows, err := tx.Query(ctx,
				`select tmdb_id, media_type from title_verdicts
				 where user_id = $1 and verdict = 'want_to_watch'
				 order by created_at desc`,
				userID)
			if err != nil {
				return err
			}
			defer rows.Close()
			for rows.Next() {
				var it watchlistItem
				if err := rows.Scan(&it.TMDBID, &it.MediaType); err != nil {
					return err
				}
				items = append(items, it)
			}
			return rows.Err()
		})
		return items, err
	}
}

// agentTitleRef/agentTitlesRequest mirror agent/catalog_tool.py's
// TitleRef/TitlesRequest — the POST /titles request body.
type agentTitleRef struct {
	TMDBID    int    `json:"tmdb_id"`
	MediaType string `json:"media_type"`
}
type agentTitlesRequest struct {
	WatchRegion    string          `json:"watch_region"`
	WatchProviders []int           `json:"watch_providers,omitempty"`
	Items          []agentTitleRef `json:"items"`
}

// agentTitlesCaller fetches enriched rows for a known set of ids from the
// agent's POST /titles — see newAgentTitlesCaller.
type agentTitlesCaller func(ctx context.Context, req agentTitlesRequest) ([]agentPick, error)

// newAgentTitlesCaller mirrors newAgentCaller's request-building (chat.go)
// and newAgentProviderCaller's non-streaming response handling
// (providers.go): a plain JSON POST, decoded straight into []agentPick —
// /titles' response is the same enriched-pick shape /chat's "results" event
// already carries, just with blurb "" and each row possibly degraded
// (agentPick.Unavailable).
func newAgentTitlesCaller(client *http.Client, baseURL string) agentTitlesCaller {
	return func(ctx context.Context, req agentTitlesRequest) ([]agentPick, error) {
		body, err := json.Marshal(req)
		if err != nil {
			return nil, err
		}
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/titles", bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(httpReq)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("agent: unexpected status %d", resp.StatusCode)
		}

		var picks []agentPick
		if err := json.NewDecoder(resp.Body).Decode(&picks); err != nil {
			return nil, err
		}
		return picks, nil
	}
}

// watchlist is GET /api/watchlist (TASKS.md T18.5): every want_to_watch
// title, enriched fresh from TMDB on every call — title_verdicts holds only
// tmdb_id/media_type/verdict, nothing to re-serve even if this wanted to
// (T17), which is what makes "re-check availability at read time" the only
// possible read path here, not a design choice made in this handler.
func (h *Handler) watchlist(w http.ResponseWriter, r *http.Request) {
	userID, _ := r.Context().Value(ctxUserID).(string)

	items, err := h.loadWatchlistItems(r.Context(), userID)
	if err != nil {
		slog.ErrorContext(r.Context(), "watchlist load failed", "error", dbError(err))
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "temporarily unavailable"})
		return
	}
	if len(items) == 0 {
		writeJSON(w, http.StatusOK, []agentPick{})
		return
	}

	cc, err := h.loadChatCtx(r.Context(), userID)
	if err != nil {
		slog.ErrorContext(r.Context(), "watchlist context load failed", "error", dbError(err))
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "temporarily unavailable"})
		return
	}

	refs := make([]agentTitleRef, len(items))
	for i, it := range items {
		refs[i] = agentTitleRef{TMDBID: it.TMDBID, MediaType: it.MediaType}
	}

	ctx, cancel := context.WithTimeout(r.Context(), watchlistTimeout)
	defer cancel()
	picks, err := h.callAgentTitles(ctx, agentTitlesRequest{
		WatchRegion: cc.Region, WatchProviders: cc.Providers, Items: refs,
	})
	if err != nil {
		slog.ErrorContext(r.Context(), "watchlist enrichment failed", "error", err.Error())
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "temporarily unavailable"})
		return
	}
	if picks == nil {
		picks = []agentPick{}
	}
	writeJSON(w, http.StatusOK, picks)
}
