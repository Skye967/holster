package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// providersRefreshTimeout bounds the gateway's call to the agent when
// streaming_providers is missing or stale. agentClient (main.go) deliberately
// carries no client-level Timeout — chat's streaming call needs the request to
// stay open — so this plain JSON call bounds itself instead, the same pattern
// provisionTimeout uses for the upsertUser path.
//
// 15s, not the agent's own worst case: agent/tmdb.py's retry logic can wait up
// to ~20s (two TMDB-instructed retry-after waits, one per concurrent movie/tv
// call) before giving up. 15s covers a single such wait — the common transient
// case — without making every /connections load that hits it block that long;
// the rare tail where both calls independently hit a full retry-after still
// times out here and correctly falls through to serving stale/empty, the
// designed degrade, not a bug.
var providersRefreshTimeout = 15 * time.Second

// providerCacheTTL is T15's "lazy cache, no scheduler" window (../../TASKS.md):
// serve what's cached, refresh from the agent only when it is older than this.
const providerCacheTTL = 24 * time.Hour

// Provider is one entry of streaming_providers.providers — every service
// available in a country, in TMDB's own ranking (DisplayPriority). Kept
// separate from chat.go's cachedProvider (below), which only needs the
// id/name pair to resolve names for the "interpreting" line and is already
// tested there — if one of these two decoders' field names changes, check
// the other, since both read the same jsonb column independently.
//
// LogoURL, not a raw logo path: the comment on streaming_providers in
// 20260831230634_streaming.sql predates agent/tmdb.py's established Provider
// convention of pre-building a full image URL via _image_url(). jsonb has no
// enforced key schema, so following that already-established shape here is a
// comment-level correction, not a migration change — the same way T9 itself
// superseded a stale comment on init.sql without editing the applied file.
type Provider struct {
	ProviderID   int    `json:"provider_id"`
	ProviderName string `json:"provider_name"`
	// A pointer: agent's logo_url is str | None and genuinely sends null for
	// a provider TMDB gave no logo to. Unmarshaling null into a non-pointer
	// string is a silent no-op that would erase that distinction.
	LogoURL         *string `json:"logo_url"`
	DisplayPriority int     `json:"display_priority"`
}

// providerCatalogEntry is what GET /api/providers returns: the region's
// catalog merged with which of them the caller has ticked — one response, not
// two round trips from the browser.
type providerCatalogEntry struct {
	Provider
	Subscribed bool `json:"subscribed"`
}

// agentProviderCaller fetches one region's provider list from the agent. A
// plain JSON GET, not agentCaller's streaming-NDJSON shape (chat.go), so it
// is its own type rather than forced into that one.
type agentProviderCaller func(ctx context.Context, region string) ([]Provider, error)

// newAgentProviderCaller mirrors newAgentCaller's shape (chat.go) for a
// simple, non-streaming call to the agent's GET /providers.
func newAgentProviderCaller(client *http.Client, baseURL string) agentProviderCaller {
	return func(ctx context.Context, region string) ([]Provider, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet,
			baseURL+"/providers?region="+url.QueryEscape(region), nil)
		if err != nil {
			return nil, err
		}

		resp, err := client.Do(req)
		if err != nil {
			return nil, err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("agent: unexpected status %d", resp.StatusCode)
		}

		var providers []Provider
		if err := json.NewDecoder(resp.Body).Decode(&providers); err != nil {
			return nil, err
		}
		return providers, nil
	}
}

// loadProviders reads the country's cached streaming_providers row, or the
// agent's live list if there is none or it is older than providerCacheTTL —
// T15's "lazy cache, no scheduler." A refresh failure degrades to whatever is
// already cached (or an empty list if nothing is), the same shape
// loadChatContext already uses for its own missing-row case. A benign race
// between two concurrent stale refreshes — both call the agent, both
// upsert — is acceptable here: this is a low-traffic settings page, and
// there is no advisory lock guarding it.
//
// Not wrapped in withUser: provider_access's `using (true)` policy
// (20260831233121_rls_roles.sql) carries no per-user predicate for this
// table — only the grant and RLS being enabled matter.
func loadProviders(db *pgxpool.Pool, refresh agentProviderCaller) func(ctx context.Context, country string) ([]Provider, error) {
	return func(ctx context.Context, country string) ([]Provider, error) {
		var (
			cachedJSON []byte
			fetchedAt  time.Time
		)
		err := db.QueryRow(ctx,
			`select providers, fetched_at from streaming_providers where country = $1`,
			country).Scan(&cachedJSON, &fetchedAt)

		var cached []Provider
		switch {
		case errors.Is(err, pgx.ErrNoRows):
			// No row yet for this country — fall through to refresh.
		case err != nil:
			return nil, err
		default:
			if jsonErr := json.Unmarshal(cachedJSON, &cached); jsonErr != nil {
				return nil, jsonErr
			}
			if time.Since(fetchedAt) < providerCacheTTL {
				return cached, nil
			}
		}

		refreshCtx, cancel := context.WithTimeout(ctx, providersRefreshTimeout)
		defer cancel()
		fresh, err := refresh(refreshCtx, country)
		if err != nil {
			slog.WarnContext(ctx, "provider refresh failed, serving cached",
				"error", err.Error(), "country", country)
			return cached, nil
		}

		freshJSON, err := json.Marshal(fresh)
		if err != nil {
			return nil, err
		}
		if _, err := db.Exec(ctx, `
			insert into streaming_providers (country, providers, fetched_at)
			values ($1, $2, now())
			on conflict (country) do update
				set providers = excluded.providers, fetched_at = excluded.fetched_at`,
			country, freshJSON); err != nil {
			// The fresh list is still good — only the cache write failed. Serve
			// it anyway; the next stale read tries the write again.
			slog.WarnContext(ctx, "provider cache write failed, serving fresh anyway",
				"error", dbError(err), "country", country)
		}
		return fresh, nil
	}
}

// saveSubscription mirrors upsertUser's shape (main.go): a plain function
// closed over the pool, run inside withUser so subscription_isolation's RLS
// policy (20260831233121_rls_roles.sql) applies.
func saveSubscription(db *pgxpool.Pool) func(ctx context.Context, userID string, providerID int, subscribed bool) error {
	return func(ctx context.Context, userID string, providerID int, subscribed bool) error {
		return withUser(ctx, db, userID, func(tx pgx.Tx) error {
			var err error
			if subscribed {
				_, err = tx.Exec(ctx,
					`insert into streaming_subscriptions (user_id, tmdb_provider_id) values ($1, $2)
					 on conflict do nothing`, userID, providerID)
			} else {
				_, err = tx.Exec(ctx,
					`delete from streaming_subscriptions where user_id = $1 and tmdb_provider_id = $2`,
					userID, providerID)
			}
			return err
		})
	}
}

// providers is GET /api/providers — the picker's one round trip (TASKS.md
// T15): the caller's region's catalog, each entry flagged with whether the
// caller already subscribes to it.
func (h *Handler) providers(w http.ResponseWriter, r *http.Request) {
	userID, _ := r.Context().Value(ctxUserID).(string)

	cc, err := h.loadChatCtx(r.Context(), userID)
	if err != nil {
		slog.ErrorContext(r.Context(), "provider context load failed", "error", dbError(err))
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "temporarily unavailable"})
		return
	}

	catalog, err := h.loadProviders(r.Context(), cc.Region)
	if err != nil {
		slog.ErrorContext(r.Context(), "provider catalog load failed", "error", dbError(err))
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "temporarily unavailable"})
		return
	}

	subscribed := make(map[int]bool, len(cc.Providers))
	for _, id := range cc.Providers {
		subscribed[id] = true
	}

	entries := make([]providerCatalogEntry, len(catalog))
	for i, p := range catalog {
		entries[i] = providerCatalogEntry{Provider: p, Subscribed: subscribed[p.ProviderID]}
	}
	writeJSON(w, http.StatusOK, entries)
}

// setSubscription is PUT/DELETE /api/subscriptions/{providerID} — one
// service, toggled immediately, no request body (TASKS.md T15: "toggles save
// on flip"). PUT is the idempotent "make it subscribed" (on conflict do
// nothing — ticking an already-ticked service is a no-op, not an error);
// DELETE is "make it not."
func (h *Handler) setSubscription(w http.ResponseWriter, r *http.Request) {
	userID, _ := r.Context().Value(ctxUserID).(string)

	providerID, err := strconv.Atoi(r.PathValue("providerID"))
	if err != nil || providerID <= 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid provider id"})
		return
	}

	subscribed := r.Method == http.MethodPut
	if err := h.saveSubscription(r.Context(), userID, providerID, subscribed); err != nil {
		slog.ErrorContext(r.Context(), "subscription write failed", "error", dbError(err))
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "temporarily unavailable"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"subscribed": subscribed})
}
