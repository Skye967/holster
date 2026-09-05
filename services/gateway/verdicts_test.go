package main

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"encoding/json"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/MicahParks/keyfunc/v3"
)

// newVerdictsTestServer wires a full Handler (real routing/auth) with fake
// loadVerdicts/saveVerdict, mirroring providers_test.go's
// newProvidersTestServer for the endpoints verdicts.go adds.
func newVerdictsTestServer(t *testing.T,
	loadVerdicts func(context.Context, string) ([]Verdict, error),
	saveVerdict func(context.Context, string, int, string, *string, string) error,
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
		loadVerdicts, saveVerdict,
		noopLoadWatchlistItems, noopCallAgentTitles,
		noopLoadConversation, noopLoadConversationTurns, noopSaveMessages,
	)
	if err != nil {
		t.Fatal(err)
	}

	srv := httptest.NewServer(h.routes())
	t.Cleanup(srv.Close)

	token := mint(t, "kid_A", key, validClaims())
	return srv, token
}

// doJSONBody sends token and body as a Bearer-authenticated JSON request —
// doJSON (providers_test.go) has no body parameter, and PUT /api/verdicts
// needs one.
func doJSONBody(t *testing.T, srv *httptest.Server, token, method, path, body string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(method, srv.URL+path, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

// --- GET /api/verdicts -------------------------------------------------------

func TestGetVerdictsReturnsRows(t *testing.T) {
	loadVerdicts := func(context.Context, string) ([]Verdict, error) {
		return []Verdict{
			{TMDBID: 550, MediaType: "movie", Verdict: "liked"},
			{TMDBID: 1399, MediaType: "tv", Verdict: "want_to_watch"},
		}, nil
	}
	srv, token := newVerdictsTestServer(t, loadVerdicts, noopSaveVerdict)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/verdicts")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	var got []Verdict
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].TMDBID != 550 || got[1].Verdict != "want_to_watch" {
		t.Errorf("got %+v, want the two seeded rows", got)
	}
}

// TestGetVerdictsReturnsEmptyArray guards against loadVerdicts' nil
// zero-value marshaling to JSON `null` — the same guard
// TestGetSubscriptionsReturnsEmptyArray checks for that field.
func TestGetVerdictsReturnsEmptyArray(t *testing.T) {
	srv, token := newVerdictsTestServer(t, noopLoadVerdicts, noopSaveVerdict)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/verdicts")
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

func TestGetVerdictsDegradesOnLoadFailure(t *testing.T) {
	loadVerdicts := func(context.Context, string) ([]Verdict, error) {
		return nil, errUnauthorizedParty // any non-nil error
	}
	srv, token := newVerdictsTestServer(t, loadVerdicts, noopSaveVerdict)

	resp := doJSON(t, srv, token, http.MethodGet, "/api/verdicts")
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
}

// --- PUT/DELETE /api/verdicts/{mediaType}/{tmdbID} --------------------------

func TestSetVerdictUpsertsAndClears(t *testing.T) {
	type call struct {
		userID          string
		tmdbID          int
		mediaType       string
		verdict         *string
		expectedVerdict string
	}
	var calls []call
	saveVerdict := func(_ context.Context, userID string, tmdbID int, mediaType string, verdict *string, expectedVerdict string) error {
		calls = append(calls, call{userID, tmdbID, mediaType, verdict, expectedVerdict})
		return nil
	}
	srv, token := newVerdictsTestServer(t, noopLoadVerdicts, saveVerdict)

	resp := doJSONBody(t, srv, token, http.MethodPut, "/api/verdicts/movie/550", `{"verdict":"liked"}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("PUT status = %d, want 200", resp.StatusCode)
	}
	var putBody map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&putBody); err != nil {
		t.Fatal(err)
	}
	if putBody["verdict"] != "liked" {
		t.Errorf("PUT response = %v, want verdict: liked", putBody)
	}

	resp = doJSON(t, srv, token, http.MethodDelete, "/api/verdicts/movie/550?expect=liked")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("DELETE status = %d, want 200", resp.StatusCode)
	}
	var delBody map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&delBody); err != nil {
		t.Fatal(err)
	}
	if delBody["verdict"] != "" {
		t.Errorf("DELETE response = %v, want empty verdict", delBody)
	}

	if len(calls) != 2 {
		t.Fatalf("calls = %+v, want 2 entries", calls)
	}
	if calls[0].verdict == nil || *calls[0].verdict != "liked" || calls[0].tmdbID != 550 || calls[0].mediaType != "movie" {
		t.Errorf("calls[0] = %+v, want a liked upsert for movie 550", calls[0])
	}
	if calls[1].verdict != nil || calls[1].expectedVerdict != "liked" {
		t.Errorf("calls[1] = %+v, want a nil verdict and expectedVerdict %q", calls[1], "liked")
	}
}

// DELETE's ?expect must reach saveVerdict unchanged — it's the only signal
// that tells the SQL which specific verdict it's allowed to remove (see
// saveVerdict's own comment for why comparing against it is the fix).
func TestSetVerdictDeletePassesExpectThrough(t *testing.T) {
	var lastExpected string
	saveVerdict := func(_ context.Context, _ string, _ int, _ string, _ *string, expectedVerdict string) error {
		lastExpected = expectedVerdict
		return nil
	}
	srv, token := newVerdictsTestServer(t, noopLoadVerdicts, saveVerdict)

	resp := doJSON(t, srv, token, http.MethodDelete, "/api/verdicts/movie/550?expect=want_to_watch")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if lastExpected != "want_to_watch" {
		t.Errorf("expectedVerdict = %q, want %q", lastExpected, "want_to_watch")
	}

	resp = doJSON(t, srv, token, http.MethodDelete, "/api/verdicts/movie/550?expect=liked")
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if lastExpected != "liked" {
		t.Errorf("expectedVerdict = %q, want %q", lastExpected, "liked")
	}
}

// A missing or invalid ?expect must 400 before ever reaching saveVerdict —
// the whole point of requiring it is that no caller can fall back to an
// unscoped "clear whatever's there".
func TestSetVerdictRejectsMissingOrInvalidExpect(t *testing.T) {
	saveVerdict := func(context.Context, string, int, string, *string, string) error {
		t.Fatal("saveVerdict should not run for a missing or invalid ?expect")
		return nil
	}
	srv, token := newVerdictsTestServer(t, noopLoadVerdicts, saveVerdict)

	for _, path := range []string{
		"/api/verdicts/movie/550",
		"/api/verdicts/movie/550?expect=",
		"/api/verdicts/movie/550?expect=maybe",
	} {
		resp := doJSON(t, srv, token, http.MethodDelete, path)
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("DELETE %s status = %d, want 400", path, resp.StatusCode)
		}
	}
}

func TestSetVerdictRejectsBadMediaTypeOrID(t *testing.T) {
	saveVerdict := func(context.Context, string, int, string, *string, string) error {
		t.Fatal("saveVerdict should not run for an invalid path")
		return nil
	}
	srv, token := newVerdictsTestServer(t, noopLoadVerdicts, saveVerdict)

	for _, path := range []string{
		"/api/verdicts/book/550", "/api/verdicts/movie/abc",
		"/api/verdicts/movie/0", "/api/verdicts/movie/-1",
	} {
		resp := doJSONBody(t, srv, token, http.MethodPut, path, `{"verdict":"liked"}`)
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("PUT %s status = %d, want 400", path, resp.StatusCode)
		}
	}
}

func TestSetVerdictRejectsBadVerdictValue(t *testing.T) {
	saveVerdict := func(context.Context, string, int, string, *string, string) error {
		t.Fatal("saveVerdict should not run for an invalid verdict")
		return nil
	}
	srv, token := newVerdictsTestServer(t, noopLoadVerdicts, saveVerdict)

	resp := doJSONBody(t, srv, token, http.MethodPut, "/api/verdicts/movie/550", `{"verdict":"amazing"}`)
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestSetVerdictReturns409WhenLocked(t *testing.T) {
	saveVerdict := func(context.Context, string, int, string, *string, string) error {
		return errVerdictLocked
	}
	srv, token := newVerdictsTestServer(t, noopLoadVerdicts, saveVerdict)

	resp := doJSONBody(t, srv, token, http.MethodPut, "/api/verdicts/movie/550", `{"verdict":"disliked"}`)
	if resp.StatusCode != http.StatusConflict {
		t.Errorf("status = %d, want 409", resp.StatusCode)
	}
	var body map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body["code"] != "verdict_locked" {
		t.Errorf("code = %q, want verdict_locked", body["code"])
	}
}

// The client (chat-panel.tsx) distinguishes this from errVerdictLocked's 409
// by this stable code, not by wording or by which HTTP method it sent — this
// test pins the exact value that contract depends on.
func TestSetVerdictReturns409WhenStale(t *testing.T) {
	saveVerdict := func(context.Context, string, int, string, *string, string) error {
		return errVerdictStale
	}
	srv, token := newVerdictsTestServer(t, noopLoadVerdicts, saveVerdict)

	resp := doJSON(t, srv, token, http.MethodDelete, "/api/verdicts/movie/550?expect=liked")
	if resp.StatusCode != http.StatusConflict {
		t.Errorf("status = %d, want 409", resp.StatusCode)
	}
	var body map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body["code"] != "verdict_stale" {
		t.Errorf("code = %q, want verdict_stale", body["code"])
	}
}

func TestSetVerdictDegradesOnWriteFailure(t *testing.T) {
	saveVerdict := func(context.Context, string, int, string, *string, string) error {
		return errUnauthorizedParty // any non-nil error
	}
	srv, token := newVerdictsTestServer(t, noopLoadVerdicts, saveVerdict)

	resp := doJSONBody(t, srv, token, http.MethodPut, "/api/verdicts/movie/550", `{"verdict":"liked"}`)
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", resp.StatusCode)
	}
}

// The real routes, not a stand-in — same rationale as
// TestProviderRoutesRequireAuth (providers_test.go).
func TestVerdictRoutesRequireAuth(t *testing.T) {
	srv, _ := newVerdictsTestServer(t, noopLoadVerdicts, noopSaveVerdict)

	for _, p := range []struct{ method, path string }{
		{http.MethodGet, "/api/verdicts"},
		{http.MethodPut, "/api/verdicts/movie/550"},
		{http.MethodDelete, "/api/verdicts/movie/550"},
	} {
		if resp := doJSON(t, srv, "", p.method, p.path); resp.StatusCode != http.StatusUnauthorized {
			t.Errorf("%s %s without a token = %d, want 401", p.method, p.path, resp.StatusCode)
		}
	}
}

// TestVerdictJSONTagsAreTheCrossServiceContract pins the wire names, which
// three things decode independently: web/src/lib/verdicts.ts's VerdictEntry,
// the agent's TitleVerdict (agent/catalog_tool.py), and this package's own
// round trip. Every other test here decodes back into Verdict, so it would
// stay green through a rename — and so would the agent's Python-side test,
// which hardcodes these same strings without ever seeing Go. Renaming a tag
// to suit one consumer would leave every title card unmarked in the browser
// AND 422 every chat turn at the agent, with both suites passing. This is
// the assertion that fails instead.
func TestVerdictJSONTagsAreTheCrossServiceContract(t *testing.T) {
	got, err := json.Marshal(Verdict{TMDBID: 101, MediaType: "movie", Verdict: "seen"})
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"tmdb_id":101,"media_type":"movie","verdict":"seen"}`
	if string(got) != want {
		t.Errorf("Verdict JSON = %s, want %s", got, want)
	}

	// The values are as much of the contract as the field names. The agent
	// decides what to exclude with a bare string compare against
	// "want_to_watch" (agent/catalog_tool.py's search()), so renaming that
	// value here for the browser's sake would silently invert the meaning of
	// every bookmarked title — they would start being excluded from
	// recommendations instead of staying eligible — with the Go, Python and
	// TypeScript suites all still green. Pin the spelling here so it isn't.
	wantVerdicts := map[string]bool{
		"liked": true, "disliked": true, "seen": true,
		"not_interested": true, "want_to_watch": true,
	}
	if !maps.Equal(verdictValues, wantVerdicts) {
		t.Errorf("verdictValues = %v, want %v", verdictValues, wantVerdicts)
	}

	// The outer key matters as much as the inner three, and fails more
	// quietly: Pydantic's verdicts field has a default, so a renamed key is
	// accepted as an empty list rather than rejected. Every turn would
	// silently revert to "no verdicts" — judged titles back on screen — with
	// nothing failing. The round-trip tests can't catch it; they decode back
	// through the same Go struct.
	body, err := json.Marshal(agentChatRequest{
		Message:     "hi",
		WatchRegion: "US",
		Verdicts:    []Verdict{{TMDBID: 101, MediaType: "movie", Verdict: "seen"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var keyed map[string]json.RawMessage
	if err := json.Unmarshal(body, &keyed); err != nil {
		t.Fatal(err)
	}
	if _, ok := keyed["verdicts"]; !ok {
		t.Errorf("agentChatRequest JSON = %s, want a \"verdicts\" key", body)
	}
}
