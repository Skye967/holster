package main

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/MicahParks/keyfunc/v3"
)

// newProvidersTestServer wires a full Handler (real routing/auth) with fake
// loadChatCtx/loadProviders/saveSubscription, mirroring chat_test.go's
// newChatTestServer for the endpoints providers.go adds.
func newProvidersTestServer(t *testing.T,
	loadChatCtx func(context.Context, string) (chatContext, error),
	loadProviders func(context.Context, string) ([]Provider, error),
	saveSubscription func(context.Context, string, int, bool) error,
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
		loadProviders, saveSubscription,
	)
	if err != nil {
		t.Fatal(err)
	}

	srv := httptest.NewServer(h.routes())
	t.Cleanup(srv.Close)

	token := mint(t, "kid_A", key, validClaims())
	return srv, token
}

// doJSON sends token as a Bearer header, or omits Authorization entirely when
// token is "" — used by TestProviderRoutesRequireAuth to exercise the
// no-token path against the real routes rather than a separate helper.
func doJSON(t *testing.T, srv *httptest.Server, token, method, path string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(method, srv.URL+path, nil)
	if err != nil {
		t.Fatal(err)
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

// --- GET /api/providers -----------------------------------------------------

func TestGetProvidersMergesSubscribedFlag(t *testing.T) {
	loadChatCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{Region: "US", Providers: []int{8}}, nil
	}
	loadProviders := func(_ context.Context, region string) ([]Provider, error) {
		if region != "US" {
			t.Errorf("region = %q, want US", region)
		}
		return []Provider{
			{ProviderID: 8, ProviderName: "Netflix", DisplayPriority: 1},
			{ProviderID: 15, ProviderName: "Hulu", DisplayPriority: 2},
		}, nil
	}
	srv, token := newProvidersTestServer(t, loadChatCtx, loadProviders, noopSaveSubscription)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/providers")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var got []providerCatalogEntry
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d entries, want 2", len(got))
	}
	if !got[0].Subscribed || got[0].ProviderID != 8 {
		t.Errorf("entry[0] = %+v, want provider 8 subscribed", got[0])
	}
	if got[1].Subscribed || got[1].ProviderID != 15 {
		t.Errorf("entry[1] = %+v, want provider 15 not subscribed", got[1])
	}
}

func TestGetProvidersDegradesOnContextFailure(t *testing.T) {
	loadChatCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{}, errUnauthorizedParty // any non-nil error
	}
	srv, token := newProvidersTestServer(t, loadChatCtx, noopLoadProviders, noopSaveSubscription)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/providers")
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
}

// --- PUT/DELETE /api/subscriptions/{providerID} -----------------------------

func TestSetSubscriptionInsertsAndDeletes(t *testing.T) {
	type call struct {
		userID     string
		providerID int
		subscribed bool
	}
	var calls []call
	saveSubscription := func(_ context.Context, userID string, providerID int, subscribed bool) error {
		calls = append(calls, call{userID, providerID, subscribed})
		return nil
	}
	srv, token := newProvidersTestServer(t, noopChatCtx, noopLoadProviders, saveSubscription)

	resp := doJSON(t, srv, token, http.MethodPut, "/api/subscriptions/8")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("PUT status = %d, want 200", resp.StatusCode)
	}
	var putBody map[string]bool
	if err := json.NewDecoder(resp.Body).Decode(&putBody); err != nil {
		t.Fatal(err)
	}
	if !putBody["subscribed"] {
		t.Errorf("PUT response = %v, want subscribed: true", putBody)
	}

	resp = doJSON(t, srv, token, http.MethodDelete, "/api/subscriptions/8")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("DELETE status = %d, want 200", resp.StatusCode)
	}
	var delBody map[string]bool
	if err := json.NewDecoder(resp.Body).Decode(&delBody); err != nil {
		t.Fatal(err)
	}
	if delBody["subscribed"] {
		t.Errorf("DELETE response = %v, want subscribed: false", delBody)
	}

	want := []call{
		{"user_abc", 8, true},
		{"user_abc", 8, false},
	}
	if len(calls) != len(want) {
		t.Fatalf("calls = %+v, want %+v", calls, want)
	}
	for i, c := range calls {
		if c != want[i] {
			t.Errorf("calls[%d] = %+v, want %+v", i, c, want[i])
		}
	}
}

func TestSetSubscriptionRejectsBadProviderID(t *testing.T) {
	srv, token := newProvidersTestServer(t, noopChatCtx, noopLoadProviders,
		func(context.Context, string, int, bool) error {
			t.Fatal("saveSubscription should not run for an invalid provider id")
			return nil
		},
	)

	for _, path := range []string{"/api/subscriptions/abc", "/api/subscriptions/0", "/api/subscriptions/-1"} {
		resp := doJSON(t, srv, token, http.MethodPut, path)
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("PUT %s status = %d, want 400", path, resp.StatusCode)
		}
	}
}

func TestSetSubscriptionDegradesOnWriteFailure(t *testing.T) {
	saveSubscription := func(context.Context, string, int, bool) error {
		return errUnauthorizedParty // any non-nil error
	}
	srv, token := newProvidersTestServer(t, noopChatCtx, noopLoadProviders, saveSubscription)

	resp := doJSON(t, srv, token, http.MethodPut, "/api/subscriptions/8")
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
}

// The real routes, not a stand-in: main_test.go's TestRoutesRequireAuthOnEveryMethod
// used to prove a subscriptions-shaped route inherits auth via a placeholder
// registered on its own mux. This proves it on the actual routes this task
// added, via the actual h.routes() — see TASKS.md T15.
func TestProviderRoutesRequireAuth(t *testing.T) {
	srv, _ := newProvidersTestServer(t, noopChatCtx, noopLoadProviders, noopSaveSubscription)

	for _, p := range []struct{ method, path string }{
		{http.MethodGet, "/api/providers"},
		{http.MethodPut, "/api/subscriptions/8"},
		{http.MethodDelete, "/api/subscriptions/8"},
	} {
		if resp := doJSON(t, srv, "", p.method, p.path); resp.StatusCode != http.StatusUnauthorized {
			t.Errorf("%s %s without a token = %d, want 401", p.method, p.path, resp.StatusCode)
		}
	}
}
