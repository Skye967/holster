package main

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/MicahParks/keyfunc/v3"
)

// newWatchlistTestServer wires a full Handler (real routing/auth) with fake
// loadWatchlistItems/callAgentTitles/loadChatCtx, mirroring
// newVerdictsTestServer (verdicts_test.go) for the endpoint watchlist.go adds.
func newWatchlistTestServer(t *testing.T,
	loadChatCtx func(context.Context, string) (chatContext, error),
	loadWatchlistItems func(context.Context, string) ([]watchlistItem, error),
	callAgentTitles agentTitlesCaller,
) (*httptest.Server, string) {
	t.Helper()
	key, _ := rsa.GenerateKey(rand.Reader, 2048)
	jwksSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(jwksJSON("kid_A", key))
	}))
	t.Cleanup(jwksSrv.Close)

	jwks, err := keyfunc.NewDefaultCtx(t.Context(), []string{jwksSrv.URL})
	if err != nil {
		t.Fatal(err)
	}

	h, err := newHandler(jwks.Keyfunc, testIssuer, testAudience, map[string]struct{}{testOrigin: {}},
		func(context.Context, string, string) error { return nil },
		loadChatCtx, noopAgentCaller, t.Context(),
		noopLoadProviders, noopSaveSubscription,
		noopLoadVerdicts, noopSaveVerdict,
		loadWatchlistItems, callAgentTitles,
		noopLoadConversation, noopLoadConversationTurns, noopSaveMessages,
		noopLoadConversationSummaries, noopDeleteConversation,
	)
	if err != nil {
		t.Fatal(err)
	}

	srv := httptest.NewServer(h.routes())
	t.Cleanup(srv.Close)

	token := mint(t, "kid_A", key, validClaims())
	return srv, token
}

func TestWatchlistReturnsEnrichedPicks(t *testing.T) {
	loadWatchlistItems := func(context.Context, string) ([]watchlistItem, error) {
		return []watchlistItem{{TMDBID: 550, MediaType: "movie"}}, nil
	}
	var gotReq agentTitlesRequest
	callAgentTitles := func(_ context.Context, req agentTitlesRequest) ([]agentPick, error) {
		gotReq = req
		return []agentPick{{TMDBID: 550, MediaType: "movie", Title: "Fight Club"}}, nil
	}
	loadChatCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{Region: "US", Providers: []int{8}, ProviderNames: []string{"Netflix"}}, nil
	}
	srv, token := newWatchlistTestServer(t, loadChatCtx, loadWatchlistItems, callAgentTitles)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/watchlist")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var got []agentPick
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].TMDBID != 550 || got[0].Title != "Fight Club" {
		t.Errorf("got %+v, want the one enriched pick", got)
	}

	if gotReq.WatchRegion != "US" || len(gotReq.WatchProviders) != 1 || gotReq.WatchProviders[0] != 8 {
		t.Errorf("agent request = %+v, want the caller's real region/providers", gotReq)
	}
	if len(gotReq.Items) != 1 || gotReq.Items[0].TMDBID != 550 || gotReq.Items[0].MediaType != "movie" {
		t.Errorf("agent request items = %+v, want the watchlist row", gotReq.Items)
	}
}

// TestWatchlistEmptySkipsAgentCall guards both the "no wasted round trip"
// behavior and the nil-slice-marshals-to-null pitfall covered elsewhere
// (TestGetVerdictsReturnsEmptyArray, TestGetSubscriptionsReturnsEmptyArray).
func TestWatchlistEmptySkipsAgentCall(t *testing.T) {
	loadWatchlistItems := func(context.Context, string) ([]watchlistItem, error) {
		return nil, nil
	}
	callAgentTitles := func(context.Context, agentTitlesRequest) ([]agentPick, error) {
		t.Fatal("callAgentTitles should not run for an empty watchlist")
		return nil, nil
	}
	loadChatCtx := func(context.Context, string) (chatContext, error) {
		t.Fatal("loadChatCtx should not run for an empty watchlist")
		return chatContext{}, nil
	}
	srv, token := newWatchlistTestServer(t, loadChatCtx, loadWatchlistItems, callAgentTitles)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/watchlist")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(body)); got != "[]" {
		t.Errorf("body = %q, want %q", got, "[]")
	}
}

func TestWatchlistDegradesOnLoadFailure(t *testing.T) {
	loadWatchlistItems := func(context.Context, string) ([]watchlistItem, error) {
		return nil, errUnauthorizedParty // any non-nil error
	}
	srv, token := newWatchlistTestServer(t, noopChatCtx, loadWatchlistItems, noopCallAgentTitles)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/watchlist")
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
}

func TestWatchlistDegradesOnChatCtxFailure(t *testing.T) {
	loadWatchlistItems := func(context.Context, string) ([]watchlistItem, error) {
		return []watchlistItem{{TMDBID: 550, MediaType: "movie"}}, nil
	}
	loadChatCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{}, errUnauthorizedParty // any non-nil error
	}
	srv, token := newWatchlistTestServer(t, loadChatCtx, loadWatchlistItems, noopCallAgentTitles)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/watchlist")
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
}

func TestWatchlistDegradesOnAgentFailure(t *testing.T) {
	loadWatchlistItems := func(context.Context, string) ([]watchlistItem, error) {
		return []watchlistItem{{TMDBID: 550, MediaType: "movie"}}, nil
	}
	callAgentTitles := func(context.Context, agentTitlesRequest) ([]agentPick, error) {
		return nil, errUnauthorizedParty // any non-nil error
	}
	srv, token := newWatchlistTestServer(t, noopChatCtx, loadWatchlistItems, callAgentTitles)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/watchlist")
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
}

// The real routes, not a stand-in — same rationale as
// TestProviderRoutesRequireAuth (providers_test.go).
func TestWatchlistRouteRequiresAuth(t *testing.T) {
	srv, _ := newWatchlistTestServer(t, noopChatCtx, noopLoadWatchlistItems, noopCallAgentTitles)

	if resp := doJSON(t, srv, "", http.MethodGet, "/api/watchlist"); resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", resp.StatusCode)
	}
}
