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

// newConversationsTestServer wires a full Handler (real routing/auth) with a
// fake loadConversationTurns, mirroring newWatchlistTestServer (watchlist_test.go)
// for the endpoint conversations.go adds.
func newConversationsTestServer(t *testing.T,
	loadConversationTurns func(context.Context, string) ([]conversationTurn, error),
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
		noopChatCtx, noopAgentCaller, t.Context(),
		noopLoadProviders, noopSaveSubscription,
		noopLoadVerdicts, noopSaveVerdict,
		noopLoadWatchlistItems, noopCallAgentTitles,
		noopLoadConversation, loadConversationTurns, noopSaveMessages,
	)
	if err != nil {
		t.Fatal(err)
	}

	srv := httptest.NewServer(h.routes())
	t.Cleanup(srv.Close)

	token := mint(t, "kid_A", key, validClaims())
	return srv, token
}

func TestChatHistoryReturnsTurns(t *testing.T) {
	loadConversationTurns := func(context.Context, string) ([]conversationTurn, error) {
		return []conversationTurn{
			{UserText: "a heist movie", AssistantText: "Suggested: Heat"},
		}, nil
	}
	srv, token := newConversationsTestServer(t, loadConversationTurns)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/chat/history")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var got []conversationTurn
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].UserText != "a heist movie" || got[0].AssistantText != "Suggested: Heat" {
		t.Errorf("got %+v, want the one stored turn", got)
	}
}

// Mirrors TestWatchlistEmptySkipsAgentCall's nil-slice-marshals-to-null
// guard, for the same reason (TestGetVerdictsReturnsEmptyArray et al.).
func TestChatHistoryReturnsEmptyArray(t *testing.T) {
	loadConversationTurns := func(context.Context, string) ([]conversationTurn, error) {
		return nil, nil
	}
	srv, token := newConversationsTestServer(t, loadConversationTurns)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/chat/history")
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

func TestChatHistoryDegradesOnLoadFailure(t *testing.T) {
	loadConversationTurns := func(context.Context, string) ([]conversationTurn, error) {
		return nil, errUnauthorizedParty // any non-nil error
	}
	srv, token := newConversationsTestServer(t, loadConversationTurns)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/chat/history")
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
}

// The real routes, not a stand-in — same rationale as
// TestProviderRoutesRequireAuth (providers_test.go).
func TestChatHistoryRouteRequiresAuth(t *testing.T) {
	srv, _ := newConversationsTestServer(t, noopLoadConversationTurns)

	if resp := doJSON(t, srv, "", http.MethodGet, "/api/chat/history"); resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", resp.StatusCode)
	}
}
