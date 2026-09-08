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

// newConversationsTestServer wires a full Handler (real routing/auth) with
// fake loadConversationTurns/loadConversationSummaries/deleteConversation,
// mirroring newWatchlistTestServer (watchlist_test.go) for the endpoints
// conversations.go adds. Callers that don't exercise one of the three pass
// the matching noop. Returns the Handler too, not just the server, so a test
// can inspect h.conversationDeletions directly (e.g. to prove a no-op delete
// never bumped it).
func newConversationsTestServer(t *testing.T,
	loadConversationTurns func(context.Context, string, string) ([]conversationTurn, error),
	loadConversationSummaries func(context.Context, string) ([]conversationSummary, error),
	deleteConversation func(context.Context, string, string) (bool, error),
) (*httptest.Server, string, *Handler) {
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
		loadConversationSummaries, deleteConversation,
		testWebhookSecretBytes, noopDeleteUser, noopUpdateUserEmail,
	)
	if err != nil {
		t.Fatal(err)
	}

	srv := httptest.NewServer(h.routes())
	t.Cleanup(srv.Close)

	token := mint(t, "kid_A", key, validClaims())
	return srv, token, h
}

// --- GET /api/chat/history/{conversationID} ---------------------------------

func TestChatHistoryReturnsTurns(t *testing.T) {
	loadConversationTurns := func(_ context.Context, _ string, conversationID string) ([]conversationTurn, error) {
		if conversationID != testConversationID {
			t.Errorf("loadConversationTurns called with %q, want %q", conversationID, testConversationID)
		}
		return []conversationTurn{
			{UserText: "a heist movie", AssistantText: "Suggested: Heat"},
		}, nil
	}
	srv, token, _ := newConversationsTestServer(t, loadConversationTurns, noopLoadConversationSummaries, noopDeleteConversation)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/chat/history/"+testConversationID)
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

// TestChatHistoryCanonicalizesTheConversationID is the regression guard: a
// urn:uuid:-prefixed id is valid per uuid.Parse but not per Postgres's uuid
// column, so chatHistory must query with the canonical (parsed.String())
// form, not the raw path value — otherwise this 200s past validation only
// to 503 at the database.
func TestChatHistoryCanonicalizesTheConversationID(t *testing.T) {
	loadConversationTurns := func(_ context.Context, _ string, conversationID string) ([]conversationTurn, error) {
		if conversationID != testConversationID {
			t.Errorf("loadConversationTurns called with %q, want the canonical %q", conversationID, testConversationID)
		}
		return nil, nil
	}
	srv, token, _ := newConversationsTestServer(t, loadConversationTurns, noopLoadConversationSummaries, noopDeleteConversation)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/chat/history/urn:uuid:"+testConversationID)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
}

// Mirrors TestWatchlistEmptySkipsAgentCall's nil-slice-marshals-to-null
// guard, for the same reason (TestGetVerdictsReturnsEmptyArray et al.).
func TestChatHistoryReturnsEmptyArray(t *testing.T) {
	loadConversationTurns := func(context.Context, string, string) ([]conversationTurn, error) {
		return nil, nil
	}
	srv, token, _ := newConversationsTestServer(t, loadConversationTurns, noopLoadConversationSummaries, noopDeleteConversation)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/chat/history/"+testConversationID)
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
	loadConversationTurns := func(context.Context, string, string) ([]conversationTurn, error) {
		return nil, errUnauthorizedParty // any non-nil error
	}
	srv, token, _ := newConversationsTestServer(t, loadConversationTurns, noopLoadConversationSummaries, noopDeleteConversation)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/chat/history/"+testConversationID)
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
}

func TestChatHistoryRejectsInvalidConversationID(t *testing.T) {
	srv, token, _ := newConversationsTestServer(t, noopLoadConversationTurns, noopLoadConversationSummaries, noopDeleteConversation)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/chat/history/not-a-uuid")
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

// The real routes, not a stand-in — same rationale as
// TestProviderRoutesRequireAuth (providers_test.go).
func TestChatHistoryRouteRequiresAuth(t *testing.T) {
	srv, _, _ := newConversationsTestServer(t, noopLoadConversationTurns, noopLoadConversationSummaries, noopDeleteConversation)

	if resp := doJSON(t, srv, "", http.MethodGet, "/api/chat/history/"+testConversationID); resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", resp.StatusCode)
	}
}

// --- GET /api/conversations --------------------------------------------------

func TestConversationsReturnsList(t *testing.T) {
	loadConversationSummaries := func(context.Context, string) ([]conversationSummary, error) {
		return []conversationSummary{
			{ID: testConversationID, Title: "a heist movie"},
		}, nil
	}
	srv, token, _ := newConversationsTestServer(t, noopLoadConversationTurns, loadConversationSummaries, noopDeleteConversation)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/conversations")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var got []conversationSummary
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].ID != testConversationID || got[0].Title != "a heist movie" {
		t.Errorf("got %+v, want the one stored conversation", got)
	}
}

func TestConversationsReturnsEmptyArray(t *testing.T) {
	loadConversationSummaries := func(context.Context, string) ([]conversationSummary, error) {
		return nil, nil
	}
	srv, token, _ := newConversationsTestServer(t, noopLoadConversationTurns, loadConversationSummaries, noopDeleteConversation)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/conversations")
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

func TestConversationsDegradesOnLoadFailure(t *testing.T) {
	loadConversationSummaries := func(context.Context, string) ([]conversationSummary, error) {
		return nil, errUnauthorizedParty // any non-nil error
	}
	srv, token, _ := newConversationsTestServer(t, noopLoadConversationTurns, loadConversationSummaries, noopDeleteConversation)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/conversations")
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
}

func TestConversationsRouteRequiresAuth(t *testing.T) {
	srv, _, _ := newConversationsTestServer(t, noopLoadConversationTurns, noopLoadConversationSummaries, noopDeleteConversation)

	if resp := doJSON(t, srv, "", http.MethodGet, "/api/conversations"); resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", resp.StatusCode)
	}
}

// --- DELETE /api/conversations/{id} -----------------------------------------

func TestDeleteConversationSucceeds(t *testing.T) {
	var gotID string
	deleteConversation := func(_ context.Context, _ string, conversationID string) (bool, error) {
		gotID = conversationID
		return true, nil
	}
	srv, token, _ := newConversationsTestServer(t, noopLoadConversationTurns, noopLoadConversationSummaries, deleteConversation)

	resp := doJSON(t, srv, token, http.MethodDelete, "/api/conversations/"+testConversationID)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if gotID != testConversationID {
		t.Errorf("deleteConversation called with %q, want %q", gotID, testConversationID)
	}
}

// TestDeleteConversationReportsTrueWhenARowWasRemoved pins the happy path
// for the response body's `deleted` field.
func TestDeleteConversationReportsTrueWhenARowWasRemoved(t *testing.T) {
	deleteConversation := func(context.Context, string, string) (bool, error) {
		return true, nil
	}
	srv, token, _ := newConversationsTestServer(t, noopLoadConversationTurns, noopLoadConversationSummaries, deleteConversation)

	resp := doJSON(t, srv, token, http.MethodDelete, "/api/conversations/"+testConversationID)
	var body map[string]bool
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if !body["deleted"] {
		t.Errorf("body = %v, want deleted=true", body)
	}
}

// TestDeleteConversationReportsFalseForANoOpDelete is the regression guard:
// the body was previously hardcoded to {"deleted": true} regardless of what
// deleteConversation actually reported.
func TestDeleteConversationReportsFalseForANoOpDelete(t *testing.T) {
	deleteConversation := func(context.Context, string, string) (bool, error) {
		return false, nil
	}
	srv, token, _ := newConversationsTestServer(t, noopLoadConversationTurns, noopLoadConversationSummaries, deleteConversation)

	resp := doJSON(t, srv, token, http.MethodDelete, "/api/conversations/"+testConversationID)
	var body map[string]bool
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body["deleted"] {
		t.Errorf("body = %v, want deleted=false", body)
	}
}

func TestDeleteConversationRejectsInvalidID(t *testing.T) {
	deleteConversation := func(context.Context, string, string) (bool, error) {
		t.Error("deleteConversation must not be called for an invalid id")
		return false, nil
	}
	srv, token, _ := newConversationsTestServer(t, noopLoadConversationTurns, noopLoadConversationSummaries, deleteConversation)

	resp := doJSON(t, srv, token, http.MethodDelete, "/api/conversations/not-a-uuid")
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestDeleteConversationDegradesOnFailure(t *testing.T) {
	deleteConversation := func(context.Context, string, string) (bool, error) {
		return false, errUnauthorizedParty // any non-nil error
	}
	srv, token, _ := newConversationsTestServer(t, noopLoadConversationTurns, noopLoadConversationSummaries, deleteConversation)

	resp := doJSON(t, srv, token, http.MethodDelete, "/api/conversations/"+testConversationID)
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
}

func TestDeleteConversationRouteRequiresAuth(t *testing.T) {
	srv, _, _ := newConversationsTestServer(t, noopLoadConversationTurns, noopLoadConversationSummaries, noopDeleteConversation)

	if resp := doJSON(t, srv, "", http.MethodDelete, "/api/conversations/"+testConversationID); resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", resp.StatusCode)
	}
}

// TestDeleteConversationDoesNotTombstoneANoOpDelete is the regression guard
// for the cross-user poisoning hazard: a delete that matched no row (already
// gone, or never this caller's — RLS makes a foreign id look identical) must
// never advance the conversation's deletion generation, or any caller could
// suppress another user's still-legitimate straggling turn just by
// attempting to delete their conversation id.
func TestDeleteConversationDoesNotTombstoneANoOpDelete(t *testing.T) {
	deleteConversation := func(context.Context, string, string) (bool, error) {
		return false, nil
	}
	srv, token, h := newConversationsTestServer(t, noopLoadConversationTurns, noopLoadConversationSummaries, deleteConversation)

	resp := doJSON(t, srv, token, http.MethodDelete, "/api/conversations/"+testConversationID)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if got := h.conversationDeletions.generation(testConversationID); got != 0 {
		t.Errorf("generation = %d, want 0 (a no-op delete must not tombstone the id)", got)
	}
}

// TestDeleteConversationTombstonesARealDelete is the happy-path counterpart
// to the test above, pinning the exact condition that should advance the
// generation.
func TestDeleteConversationTombstonesARealDelete(t *testing.T) {
	deleteConversation := func(context.Context, string, string) (bool, error) {
		return true, nil
	}
	srv, token, h := newConversationsTestServer(t, noopLoadConversationTurns, noopLoadConversationSummaries, deleteConversation)

	resp := doJSON(t, srv, token, http.MethodDelete, "/api/conversations/"+testConversationID)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if got := h.conversationDeletions.generation(testConversationID); got == 0 {
		t.Error("generation = 0, want a real delete to have tombstoned the id")
	}
}

// TestTitleFromMessageUsesTheDefaultForEmptyOrWhitespaceOnlyText proves the
// fallback path shares defaultConversationTitle with loadConversationSummaries'
// own coalesce fallback, rather than each hardcoding "New chat" independently.
func TestTitleFromMessageUsesTheDefaultForEmptyOrWhitespaceOnlyText(t *testing.T) {
	for _, text := range []string{"", "   ", "\n\t "} {
		if got := titleFromMessage(text); got != defaultConversationTitle {
			t.Errorf("titleFromMessage(%q) = %q, want %q", text, got, defaultConversationTitle)
		}
	}
}

func TestTitleFromMessageTruncatesLongPlainTextAtMaxTitleLength(t *testing.T) {
	text := strings.Repeat("a", maxTitleLength+10)
	got := titleFromMessage(text)
	want := strings.Repeat("a", maxTitleLength) + "…"
	if got != want {
		t.Errorf("titleFromMessage(...) = %q, want %q", got, want)
	}
}

// TestTitleFromMessageDoesNotBisectAFlagEmoji is the regression guard: a
// cutoff landing between a flag's two regional-indicator codepoints must
// drop the stranded half rather than truncate to a single boxed letter.
func TestTitleFromMessageDoesNotBisectAFlagEmoji(t *testing.T) {
	prefix := strings.Repeat("a", maxTitleLength-1)
	got := titleFromMessage(prefix + "🇺🇸") // cut lands between the flag's two codepoints
	runes := []rune(strings.TrimSuffix(got, "…"))
	if len(runes) != maxTitleLength-1 {
		t.Errorf("titleFromMessage(...) = %q, want the stranded regional-indicator half dropped", got)
	}
	if last := runes[len(runes)-1]; last >= 0x1F1E6 && last <= 0x1F1FF {
		t.Errorf("titleFromMessage(...) = %q, still ends in an unpaired regional indicator", got)
	}
}

// TestTitleFromMessageDropsADanglingJoiner is the regression guard for a
// cutoff landing right on a zero-width joiner with nothing left to join.
func TestTitleFromMessageDropsADanglingJoiner(t *testing.T) {
	prefix := strings.Repeat("a", maxTitleLength-1)
	got := titleFromMessage(prefix + "👩‍👩") // ZWJ-joined pair; cut lands on the joiner
	runes := []rune(strings.TrimSuffix(got, "…"))
	if last := runes[len(runes)-1]; last == 0x200D {
		t.Errorf("titleFromMessage(...) = %q, still ends in a dangling zero-width joiner", got)
	}
}

// TestTitleFromMessageKeepsACompleteCharacterAtTheBoundary guards the
// mistake a broader "strip every trailing mark" approach would make: a base
// plus its combining mark that both land inside the cutoff are already
// complete and must not be stripped — suffix truncation can only drop what
// comes after a kept rune, so a kept mark's base is always present too.
func TestTitleFromMessageKeepsACompleteCharacterAtTheBoundary(t *testing.T) {
	base := strings.Repeat("a", maxTitleLength-2) + "é" // "e" + combining acute, both within the cutoff
	if len([]rune(base)) != maxTitleLength {
		t.Fatalf("test setup: base has %d runes, want %d", len([]rune(base)), maxTitleLength)
	}
	got := titleFromMessage(base + " and more")
	want := base + "…"
	if got != want {
		t.Errorf("titleFromMessage(...) = %q, want %q (a complete trailing character must not be stripped)", got, want)
	}
}
