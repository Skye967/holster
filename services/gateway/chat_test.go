package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/MicahParks/keyfunc/v3"
	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgconn"
	"golang.org/x/time/rate"
)

// testConversationID is the conversation id most tests below send on every
// "message" frame — its value is arbitrary (TASKS.md T20.5: conversation ids
// are client-generated, opaque UUIDs), only used where a test isn't itself
// about multi-conversation behavior.
const testConversationID = "11111111-1111-1111-1111-111111111111"
const otherConversationID = "22222222-2222-2222-2222-222222222222"

// --- ticketStore -------------------------------------------------------

func TestTicketConsumeIsSingleUse(t *testing.T) {
	s := newChatTicketStore()
	id, err := s.mint("user_1")
	if err != nil {
		t.Fatal(err)
	}

	got, ok := s.consume(id)
	if !ok || got.userID != "user_1" {
		t.Fatalf("consume(1st) = %+v, %v; want user_1, true", got, ok)
	}

	if _, ok := s.consume(id); ok {
		t.Error("a ticket was consumed twice")
	}
}

func TestTicketConsumeRejectsUnknownOrEmpty(t *testing.T) {
	s := newChatTicketStore()
	if _, ok := s.consume("does-not-exist"); ok {
		t.Error("consumed a ticket that was never minted")
	}
	if _, ok := s.consume(""); ok {
		t.Error("consumed an empty ticket")
	}
}

func TestTicketExpires(t *testing.T) {
	old := chatTicketTTL
	chatTicketTTL = 10 * time.Millisecond
	t.Cleanup(func() { chatTicketTTL = old })

	s := newChatTicketStore()
	id, err := s.mint("user_1")
	if err != nil {
		t.Fatal(err)
	}
	time.Sleep(50 * time.Millisecond)

	if _, ok := s.consume(id); ok {
		t.Error("consumed a ticket past its TTL")
	}
}

// --- history windowing ---------------------------------------------------

// sameTurns compares the role/text window, which is all the agent is sent —
// historyTurn also carries TitleRefs (a slice, so the struct isn't
// comparable), and that half is asserted by the shown-set tests instead.
func sameTurns(a, b []historyTurn) bool {
	return slices.EqualFunc(a, b, func(x, y historyTurn) bool {
		return x.Role == y.Role && x.Text == y.Text
	})
}

func TestWindowHistoryKeepsOnlyTheLastFewExchanges(t *testing.T) {
	var history []historyTurn
	for i := range maxHistoryExchanges + 3 {
		history = append(history,
			historyTurn{Role: "user", Text: strings.Repeat("u", i)},
			historyTurn{Role: "assistant", Text: strings.Repeat("a", i)},
		)
	}
	got := windowHistory(history)
	if len(got) != maxHistoryExchanges*2 {
		t.Fatalf("len(windowHistory(...)) = %d, want %d", len(got), maxHistoryExchanges*2)
	}
	if !sameTurns(got[:1], history[len(history)-maxHistoryExchanges*2:][:1]) {
		t.Error("windowHistory did not keep the most recent exchanges")
	}
}

func TestWindowHistoryPassesShortHistoryThrough(t *testing.T) {
	history := []historyTurn{{Role: "user", Text: "hi"}}
	got := windowHistory(history)
	if len(got) != 1 {
		t.Errorf("len = %d, want 1", len(got))
	}
}

// --- interpreting-line template ------------------------------------------

func TestInterpretingLineComposesGenreYearRuntimeAndProviders(t *testing.T) {
	gte, lte, runtime := 1990, 1999, 90
	intent := agentIntent{
		MediaType:         "movie",
		Genres:            []string{"comedy"},
		ReleaseYearGte:    &gte,
		ReleaseYearLte:    &lte,
		MaxRuntimeMinutes: &runtime,
	}
	line := interpretingLine(intent, []string{"Netflix", "Hulu"})

	for _, want := range []string{"comedy", "1990-1999", "under 90 minutes", "Netflix and Hulu"} {
		if !strings.Contains(line, want) {
			t.Errorf("interpretingLine() = %q, missing %q", line, want)
		}
	}
}

// tmdb.py OR-joins with_keywords, and the interpret prompt now encourages
// near-synonyms for one mood, so a comma list here promised a conjunction the
// results provably do not satisfy.
func TestInterpretingLineReadsMultipleMoodsAsEitherNotBoth(t *testing.T) {
	line := interpretingLine(agentIntent{
		MediaType: "movie",
		Keywords:  []string{"slow burn", "tense"},
	}, nil)

	if !strings.Contains(line, "about slow burn or tense") {
		t.Errorf("interpretingLine() = %q, want the moods joined with \"or\"", line)
	}
}

func TestSummarizePicksDistinguishesALookupFromASuggestion(t *testing.T) {
	// This text is what a reload renders and what feeds the next turn's
	// interpret(), so "Suggested" on a lookup would both misreport the turn
	// and be read back as a recommendation that never happened.
	picks := []agentPick{{TMDBID: 1, Title: "Heat"}}

	if got := summarizePicks(picks, false); got != "Suggested: Heat" {
		t.Errorf("summarizePicks(discover) = %q, want %q", got, "Suggested: Heat")
	}
	if got := summarizePicks(picks, true); got != "Looked up: Heat" {
		t.Errorf("summarizePicks(lookup) = %q, want %q", got, "Looked up: Heat")
	}
	if got := summarizePicks(nil, true); got != "" {
		t.Errorf("summarizePicks(no picks) = %q, want empty", got)
	}
}

func TestSummarizePicksDisambiguatesRepeatedTitles(t *testing.T) {
	// Exact name matches carry the same title by definition, so a lookup on an
	// ambiguous name repeats it once per card. This string is what a reload
	// renders and what feeds the next turn's interpret(); "Dune, Dune" tells
	// neither of them which two films were on screen.
	y1, y2 := 2021, 1984
	picks := []agentPick{
		{TMDBID: 1, Title: "Dune", Year: &y1},
		{TMDBID: 2, Title: "Dune", Year: &y2},
	}

	want := "Looked up: Dune (2021), Dune (1984)"
	if got := summarizePicks(picks, true); got != want {
		t.Errorf("summarizePicks() = %q, want %q", got, want)
	}
}

func TestSummarizePicksLeavesDistinctTitlesAlone(t *testing.T) {
	// Only a repeat needs the year; a franchise row is already unambiguous.
	y := 1977
	picks := []agentPick{
		{TMDBID: 1, Title: "Star Wars", Year: &y},
		{TMDBID: 2, Title: "Star Wars: The Last Jedi", Year: &y},
	}

	want := "Looked up: Star Wars, Star Wars: The Last Jedi"
	if got := summarizePicks(picks, true); got != want {
		t.Errorf("summarizePicks() = %q, want %q", got, want)
	}
}

func TestAgentEventDecodesTheResultsKind(t *testing.T) {
	// The gateway no longer infers a lookup from intent.Title, so this tag is
	// the only thing standing between a lookup and being stored as "Suggested".
	var ev agentEvent
	if err := json.Unmarshal([]byte(`{"type":"results","kind":"lookup"}`), &ev); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if ev.Kind != "lookup" {
		t.Errorf("agentEvent.Kind = %q, want %q", ev.Kind, "lookup")
	}
}

func TestInterpretingLineNamesTheTitleOnALookup(t *testing.T) {
	// A named-title turn skips discover() on the agent side, so every clause
	// the discover path composes would describe a search that never ran —
	// "Looking for movies on Netflix" over a card for one specific film.
	intent := agentIntent{
		MediaType: "movie",
		Title:     "Heat",
		Genres:    []string{"comedy"},
	}
	line := interpretingLine(intent, []string{"Netflix", "Hulu"})

	if !strings.Contains(line, "\u201cHeat\u201d") {
		t.Errorf("interpretingLine() = %q, want it to name the title", line)
	}
	for _, unwanted := range []string{"comedy", "movies", "Netflix"} {
		if strings.Contains(line, unwanted) {
			t.Errorf("interpretingLine() = %q, should not mention %q", line, unwanted)
		}
	}
}

func TestInterpretingLineDoesNotEscapeQuotesInATitle(t *testing.T) {
	// fmt's %q is strconv.Quote: it backslash-escapes quotes inside the
	// value, and this line goes straight to the user as a comprehension
	// check. A title carrying its own quotes must read as itself.
	intent := agentIntent{MediaType: "movie", Title: `"Weird Al" Yankovic`}
	line := interpretingLine(intent, nil)

	if strings.Contains(line, `\"`) {
		t.Errorf("interpretingLine() = %q, want no backslash-escaped quotes", line)
	}
	if !strings.Contains(line, `"Weird Al" Yankovic`) {
		t.Errorf("interpretingLine() = %q, want the title verbatim", line)
	}
}

func TestAgentIntentDecodesTheTitleField(t *testing.T) {
	// agentIntent is hand-kept against the agent's DiscoverIntent dump and an
	// unknown key vanishes silently on decode, so the json tag is pinned here
	// rather than left to interpretingLine's own test to notice.
	var intent agentIntent
	if err := json.Unmarshal([]byte(`{"media_type":"movie","title":"Fargo"}`), &intent); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if intent.Title != "Fargo" {
		t.Errorf("agentIntent.Title = %q, want %q", intent.Title, "Fargo")
	}
}

func TestInterpretingLineDegradesGracefullyWithNoProviderNames(t *testing.T) {
	line := interpretingLine(agentIntent{MediaType: "tv"}, nil)
	if strings.Contains(line, " on ") {
		t.Errorf("interpretingLine() = %q, should not name a provider clause", line)
	}
	if !strings.Contains(line, "TV shows") {
		t.Errorf("interpretingLine() = %q, want it to mention TV shows", line)
	}
}

func TestHumanJoin(t *testing.T) {
	cases := map[string]string{
		"":                      humanJoin(nil),
		"Netflix":               humanJoin([]string{"Netflix"}),
		"Netflix and Hulu":      humanJoin([]string{"Netflix", "Hulu"}),
		"Netflix, Hulu and Max": humanJoin([]string{"Netflix", "Hulu", "Max"}),
	}
	for want, got := range cases {
		if got != want {
			t.Errorf("humanJoin(...) = %q, want %q", got, want)
		}
	}
}

// --- provider name lookup -------------------------------------------------

func TestParseProviderNamesResolvesOnlyWantedIDsInOrder(t *testing.T) {
	raw := []byte(`[
		{"provider_id": 9, "provider_name": "Amazon"},
		{"provider_id": 8, "provider_name": "Netflix"},
		{"provider_id": 15, "provider_name": "Hulu"}
	]`)
	got := parseProviderNames(raw, []int{8, 9, 999})
	want := []string{"Netflix", "Amazon"} // 999 unresolved, dropped; order follows `wanted`
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Errorf("parseProviderNames(...) = %v, want %v", got, want)
	}
}

// --- agent HTTP streaming client ------------------------------------------

// assertPickEnrichment checks the TASKS.md T16 fields (GenreNames,
// RuntimeMinutes, AvailableOn) shared by TestNewAgentCallerStreamsEventsInOrder
// (real JSON decode) and TestChatTurnStreamsInterpretingResultsAndDone
// (gateway passthrough) — one helper so a future field change can't
// silently diverge between what the two tests check. Cast isn't included:
// only the first of those two tests asserts on it.
func assertPickEnrichment(t *testing.T, pick agentPick) {
	t.Helper()
	if len(pick.GenreNames) != 1 || pick.GenreNames[0] != "Crime" {
		t.Errorf("GenreNames = %v, want [Crime]", pick.GenreNames)
	}
	if pick.RuntimeMinutes == nil || *pick.RuntimeMinutes != 102 {
		t.Errorf("RuntimeMinutes = %v, want 102", pick.RuntimeMinutes)
	}
	if len(pick.AvailableOn) != 1 || pick.AvailableOn[0].ProviderName != "Netflix" {
		t.Errorf("AvailableOn = %+v, want one entry named Netflix", pick.AvailableOn)
	}
}

func TestNewAgentCallerStreamsEventsInOrder(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req agentChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("agent received unparseable body: %v", err)
		}
		if req.Message != "hello" || req.WatchRegion != "US" {
			t.Errorf("agent received %+v, want message=hello region=US", req)
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		flusher := w.(http.Flusher)
		for _, line := range []string{
			`{"type":"intent","intent":{"media_type":"movie"}}`,
			`{"type":"results","picks":[{"tmdb_id":1,"title":"Fake Heist","genre_names":["Crime"],"runtime_minutes":102,"cast":["Star"],"available_on":[{"provider_id":8,"provider_name":"Netflix"}],"blurb":"fits"}]}`,
			`{"type":"done"}`,
		} {
			w.Write([]byte(line + "\n"))
			flusher.Flush()
		}
	}))
	defer srv.Close()

	caller := newAgentCaller(srv.Client(), srv.URL)
	events, err := caller(t.Context(), agentChatRequest{Message: "hello", WatchRegion: "US"})
	if err != nil {
		t.Fatal(err)
	}

	var all []agentEvent
	for ev := range events {
		all = append(all, ev)
	}
	want := []string{"intent", "results", "done"}
	if len(all) != len(want) {
		t.Fatalf("got %v events, want %v", all, want)
	}
	for i := range want {
		if all[i].Type != want[i] {
			t.Errorf("event[%d].Type = %q, want %q", i, all[i].Type, want[i])
		}
	}

	// Real json.Unmarshal, not a struct literal (see TestChatTurnStreamsInterpretingResultsAndDone,
	// which injects Go values directly and so cannot catch a tag/key mismatch
	// against agent/chat.py's actual dict keys) — proves the enrichment
	// fields (TASKS.md T16) really decode from the agent's wire shape.
	pick := all[1].Picks[0]
	assertPickEnrichment(t, pick)
	if len(pick.Cast) != 1 || pick.Cast[0] != "Star" {
		t.Errorf("Cast = %v, want [Star]", pick.Cast)
	}
}

func TestNewAgentCallerRejectsNon200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	caller := newAgentCaller(srv.Client(), srv.URL)
	if _, err := caller(t.Context(), agentChatRequest{Message: "hi", WatchRegion: "US"}); err == nil {
		t.Error("a 500 from the agent was not reported as an error")
	}
}

func TestNewAgentCallerStopsOnContextCancel(t *testing.T) {
	block := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher := w.(http.Flusher)
		w.Write([]byte(`{"type":"intent","intent":{"media_type":"movie"}}` + "\n"))
		flusher.Flush()
		<-block // hang until the test lets go, simulating a stuck agent
	}))
	defer srv.Close()
	defer close(block)

	ctx, cancel := context.WithCancel(t.Context())
	caller := newAgentCaller(srv.Client(), srv.URL)
	events, err := caller(ctx, agentChatRequest{Message: "hi", WatchRegion: "US"})
	if err != nil {
		t.Fatal(err)
	}

	<-events // the "intent" event
	cancel()

	select {
	case _, ok := <-events:
		if ok {
			t.Error("received another event after the caller's context was cancelled")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("events channel never closed after cancellation")
	}
}

// TestAgentPickAvailableOnDistinguishesNullFromEmpty pins the property
// AvailableOn's doc comment relies on: a real json.Unmarshal, not just Go's
// documented behavior in the abstract, since a later refactor of this
// struct (a custom UnmarshalJSON, an intermediate map, an added omitempty)
// could silently collapse the one distinction this whole feature exists to
// preserve (TASKS.md T16: "couldn't check" must never read as "confirmed
// nowhere the caller subscribes").
func TestAgentPickAvailableOnDistinguishesNullFromEmpty(t *testing.T) {
	var failed, confirmedEmpty agentPick
	if err := json.Unmarshal([]byte(`{"available_on":null}`), &failed); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(`{"available_on":[]}`), &confirmedEmpty); err != nil {
		t.Fatal(err)
	}

	if failed.AvailableOn != nil {
		t.Errorf("AvailableOn = %#v, want nil (the check itself failed)", failed.AvailableOn)
	}
	if confirmedEmpty.AvailableOn == nil {
		t.Error("AvailableOn = nil, want a non-nil empty slice (TMDB confirmed nowhere)")
	}
	if len(confirmedEmpty.AvailableOn) != 0 {
		t.Errorf("AvailableOn = %#v, want empty", confirmedEmpty.AvailableOn)
	}
}

// --- full connection: ticket -> WS -> turn --------------------------------

func fakeAgentEvents(events ...agentEvent) agentCaller {
	return func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		ch := make(chan agentEvent, len(events))
		for _, e := range events {
			ch <- e
		}
		close(ch)
		return ch, nil
	}
}

// newChatTestServer wires a full Handler (real ticket store, real WS routing)
// with fake loadChatCtx/callAgent, and returns an httptest.Server plus a
// ready-to-use Bearer token for its one test user.
func newChatTestServer(t *testing.T, loadCtx func(context.Context, string) (chatContext, error), callAgent agentCaller, loadVerdicts func(context.Context, string) ([]Verdict, error), opts ...func(*Handler)) (*httptest.Server, string) {
	t.Helper()
	return newChatTestServerWithConversations(t, loadCtx, callAgent, loadVerdicts, noopLoadConversation, noopSaveMessages, opts...)
}

// newChatTestServerWithConversations is newChatTestServer plus the two T20/
// T20.5 dependencies, for the tests below that need to fake conversation
// hydration or observe what gets persisted — every other test goes through
// the plain wrapper above and gets the default behaviour (no history, saves
// discarded).
func newChatTestServerWithConversations(
	t *testing.T,
	loadCtx func(context.Context, string) (chatContext, error),
	callAgent agentCaller,
	loadVerdicts func(context.Context, string) ([]Verdict, error),
	loadConversation func(context.Context, string, string) ([]historyTurn, error),
	saveMessages func(context.Context, string, string, string, string, []agentTitleRef) (bool, error),
	opts ...func(*Handler),
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
		loadCtx, noopLoadGuestChatCtx, callAgent, t.Context(),
		noopLoadProviders, noopSaveSubscription,
		loadVerdicts, noopSaveVerdict,
		noopLoadWatchlistItems, noopCallAgentTitles,
		loadConversation, noopLoadConversationTurns, saveMessages,
		noopLoadConversationSummaries, noopDeleteConversation,
		testWebhookSecretBytes, noopDeleteUser, noopUpdateUserEmail,
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, opt := range opts {
		opt(h)
	}

	srv := httptest.NewServer(h.routes())
	t.Cleanup(srv.Close)

	token := mint(t, "kid_A", key, validClaims())
	return srv, token
}

func mintTicket(t *testing.T, srv *httptest.Server, token string) string {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, srv.URL+"/api/chat/ticket", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("ticket mint status = %d, want 200", resp.StatusCode)
	}
	var body map[string]string
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if body["ticket"] == "" {
		t.Fatal("ticket mint returned an empty ticket")
	}
	return body["ticket"]
}

func dialChat(t *testing.T, srv *httptest.Server, ticket string) *websocket.Conn {
	t.Helper()
	url := strings.Replace(srv.URL, "http://", "ws://", 1) + "/ws/chat?ticket=" + ticket
	conn, _, err := websocket.Dial(t.Context(), url, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { conn.CloseNow() })
	return conn
}

func TestChatTurnStreamsInterpretingResultsAndDone(t *testing.T) {
	loadCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{Region: "US", Providers: []int{8}, ProviderNames: []string{"Netflix"}}, nil
	}
	runtimeMinutes := 102
	callAgent := fakeAgentEvents(
		agentEvent{Type: "intent", Intent: json.RawMessage(`{"media_type":"movie"}`)},
		agentEvent{Type: "results", Picks: []agentPick{{
			TMDBID:         1,
			Title:          "Fake Heist",
			GenreNames:     []string{"Crime"},
			RuntimeMinutes: &runtimeMinutes,
			Cast:           []string{"Star"},
			AvailableOn:    []agentProvider{{ProviderID: 8, ProviderName: "Netflix"}},
			Blurb:          "fits",
		}}},
		agentEvent{Type: "done"},
	)
	srv, token := newChatTestServer(t, loadCtx, callAgent, noopLoadVerdicts)
	ticket := mintTicket(t, srv, token)
	conn := dialChat(t, srv, ticket)

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "a heist movie", Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}

	var got []outboundEvent
	for range 3 {
		var ev outboundEvent
		if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
			t.Fatal(err)
		}
		got = append(got, ev)
	}

	wantTypes := []string{"interpreting", "results", "done"}
	for i, ev := range got {
		if ev.Type != wantTypes[i] {
			t.Errorf("event[%d].Type = %q, want %q", i, ev.Type, wantTypes[i])
		}
		if ev.Turn != "t1" {
			t.Errorf("event[%d].Turn = %q, want t1", i, ev.Turn)
		}
	}
	if !strings.Contains(got[0].Text, "Netflix") {
		t.Errorf("interpreting text = %q, want it to name the ticked provider", got[0].Text)
	}
	if len(got[1].Picks) != 1 || got[1].Picks[0].Title != "Fake Heist" {
		t.Errorf("results picks = %+v", got[1].Picks)
	}
	// Proves only the gateway -> browser leg: fakeAgentEvents injects these
	// agentPick values directly, with no JSON decode of the agent's own wire
	// shape involved. TestNewAgentCallerStreamsEventsInOrder is what proves
	// the enrichment fields (TASKS.md T16) actually decode from real agent
	// JSON with the right keys — this just confirms they still reach the
	// browser once decoded.
	assertPickEnrichment(t, got[1].Picks[0])
}

// TestChatTurnSendsProviderNamesToTheAgent proves runTurn forwards
// chatContext.ProviderNames on to the agent (TASKS.md T16.5) — the same
// values TestChatTurnStreamsInterpretingResultsAndDone above already proves
// reach interpretingLine, reused rather than re-resolved so a
// capability-question answer can name the caller's services.
func TestChatTurnSendsProviderNamesToTheAgent(t *testing.T) {
	loadCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{Region: "US", Providers: []int{8}, ProviderNames: []string{"Netflix", "Hulu"}}, nil
	}
	var got agentChatRequest
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		got = req
		events := make(chan agentEvent, 1)
		events <- agentEvent{Type: "done"}
		close(events)
		return events, nil
	}
	srv, token := newChatTestServer(t, loadCtx, callAgent, noopLoadVerdicts)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "what can you do?", Conversation: testConversationID})
	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}

	if want := []string{"Netflix", "Hulu"}; !slices.Equal(got.WatchProviderNames, want) {
		t.Errorf("WatchProviderNames = %v, want %v", got.WatchProviderNames, want)
	}
}

func TestChatErrorReasonBecomesFriendlyText(t *testing.T) {
	loadCtx := func(context.Context, string) (chatContext, error) {
		cc := newChatContext()
		cc.Region = "US"
		return cc, nil
	}
	callAgent := fakeAgentEvents(agentEvent{Type: "error", Reason: "tmdb_unavailable"})
	srv, token := newChatTestServer(t, loadCtx, callAgent, noopLoadVerdicts)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "anything", Conversation: testConversationID})

	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.Type != "error" {
		t.Fatalf("event type = %q, want error", ev.Type)
	}
	if strings.Contains(ev.Text, "tmdb_unavailable") {
		t.Errorf("error text leaked the raw reason code: %q", ev.Text)
	}
	if ev.Text != "Can't reach the film database right now." {
		t.Errorf("error text = %q", ev.Text)
	}
}

func TestChatCancelStopsTheAgentCall(t *testing.T) {
	var sawCancel atomic.Bool
	blocked := make(chan struct{})
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		close(blocked)
		<-ctx.Done()
		sawCancel.Store(true)
		return nil, ctx.Err()
	}
	srv, token := newChatTestServer(t, noopChatCtx, callAgent, noopLoadVerdicts)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "anything", Conversation: testConversationID})
	<-blocked // the fake agent call is now hanging on ctx.Done()
	wsjson.Write(t.Context(), conn, inboundMessage{Type: "cancel", Turn: "t1"})

	deadline := time.After(2 * time.Second)
	for !sawCancel.Load() {
		select {
		case <-deadline:
			t.Fatal("cancel message never reached the agent call's context")
		case <-time.After(10 * time.Millisecond):
		}
	}
}

// TestChatNewMessageSupersedesTheInFlightTurn also proves T32's fix: a
// superseded turn is not just cancelled, it sends its own terminal event
// (never left silently unresolved on the client) — so both t1's error and
// t2's done must reach the browser, in whichever order the two goroutines
// happen to write them.
func TestChatNewMessageSupersedesTheInFlightTurn(t *testing.T) {
	var firstCancelled atomic.Bool
	firstBlocked := make(chan struct{})
	var mu sync.Mutex
	var calls int

	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		mu.Lock()
		calls++
		n := calls
		mu.Unlock()

		if n == 1 {
			close(firstBlocked)
			<-ctx.Done()
			firstCancelled.Store(true)
			return nil, ctx.Err()
		}
		ch := make(chan agentEvent, 1)
		ch <- agentEvent{Type: "done"}
		close(ch)
		return ch, nil
	}

	srv, token := newChatTestServer(t, noopChatCtx, callAgent, noopLoadVerdicts)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "first", Conversation: testConversationID})
	<-firstBlocked
	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t2", Text: "second", Conversation: testConversationID})

	byTurn := make(map[string]outboundEvent, 2)
	for len(byTurn) < 2 {
		var ev outboundEvent
		if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
			t.Fatalf("read: %v (got so far: %+v)", err, byTurn)
		}
		byTurn[ev.Turn] = ev
	}
	if got := byTurn["t1"]; got.Type != "error" {
		t.Errorf("t1's terminal event = %+v, want type error", got)
	}
	if got := byTurn["t2"]; got.Type != "done" {
		t.Errorf("t2's terminal event = %+v, want type done", got)
	}

	deadline := time.After(2 * time.Second)
	for !firstCancelled.Load() {
		select {
		case <-deadline:
			t.Fatal("first turn was never cancelled by the second message")
		case <-time.After(10 * time.Millisecond):
		}
	}
}

// --- regression coverage for the code-review fixes ------------------------

// A direct unit test of runTurn's deferred send, independent of the WS
// harness: `done` is unbuffered and nobody ever reads it here — before the
// fix this would block the goroutine forever. ctx is pre-cancelled to
// reproduce exactly the state runChatConnection leaves a turn's context in
// once the connection is gone (see chat.go's runTurn doc comment).
func TestRunTurnDoesNotBlockSendingToAnUnreadDoneChannel(t *testing.T) {
	turnCtx, cancel := context.WithCancel(t.Context())
	cancel()
	connCtx, connCancel := context.WithCancel(t.Context())
	connCancel()                  // the connection is gone, not just this turn
	done := make(chan turnRecord) // unbuffered, no reader

	h := &Handler{
		loadChatCtx:  noopChatCtx,
		loadVerdicts: noopLoadVerdicts,
		callAgent: func(context.Context, agentChatRequest) (<-chan agentEvent, error) {
			ch := make(chan agentEvent)
			close(ch)
			return ch, nil
		},
	}

	finished := make(chan struct{})
	go func() {
		h.runTurn(turnCtx, connCtx, nil, "user_1", "t1", testConversationID, "hi", nil, nil, nil, 0, done)
		close(finished)
	}()

	select {
	case <-finished:
	case <-time.After(2 * time.Second):
		t.Fatal("runTurn blocked forever sending to done with no receiver and a dead connection")
	}
}

// A turn's context must be released promptly on normal completion, not left
// to idle until turnDeadline fires on its own (30s) — see the `done` case in
// runChatConnection.
func TestChatCancelsTurnContextPromptlyOnNormalCompletion(t *testing.T) {
	ctxCh := make(chan context.Context, 1)
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		ctxCh <- ctx
		ch := make(chan agentEvent, 1)
		ch <- agentEvent{Type: "done"}
		close(ch)
		return ch, nil
	}
	srv, token := newChatTestServer(t, noopChatCtx, callAgent, noopLoadVerdicts)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "anything", Conversation: testConversationID})

	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.Type != "done" {
		t.Fatalf("event type = %q, want done", ev.Type)
	}

	turnCtx := <-ctxCh
	deadline := time.After(2 * time.Second)
	for turnCtx.Err() == nil {
		select {
		case <-deadline:
			t.Fatal("turn context was never cancelled after normal completion")
		case <-time.After(10 * time.Millisecond):
		}
	}
}

// An agent stream that ends without ever sending "done" or "error" (a
// crashed agent process, a dropped connection) must still leave the browser
// with a terminal event — never hang the turn indefinitely. See runTurn's
// post-loop fallback and TASKS.md's "a dropped stream ... never just stops
// mid-sentence looking finished."
func TestChatDroppedAgentStreamStillSendsATerminalEvent(t *testing.T) {
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		ch := make(chan agentEvent, 1)
		ch <- agentEvent{Type: "intent", Intent: json.RawMessage(`{"media_type":"movie"}`)}
		close(ch) // no "done", no "error" — the stream just stops
		return ch, nil
	}
	loadCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{Region: "US", Providers: []int{8}}, nil
	}
	srv, token := newChatTestServer(t, loadCtx, callAgent, noopLoadVerdicts)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "anything", Conversation: testConversationID})

	var got []outboundEvent
	for range 2 {
		var ev outboundEvent
		if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
			t.Fatal(err)
		}
		got = append(got, ev)
	}

	if got[0].Type != "interpreting" {
		t.Errorf("event[0].Type = %q, want interpreting", got[0].Type)
	}
	if got[1].Type != "error" {
		t.Errorf("event[1].Type = %q, want error (the dropped-stream fallback)", got[1].Type)
	}
}

// An unrecognized event type from the agent (schema drift, a typo) must be
// logged and skipped, not silently dropped with the turn otherwise completing
// as if nothing happened.
func TestChatUnknownAgentEventTypeIsLoggedAndSkipped(t *testing.T) {
	var buf bytes.Buffer
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, nil)))
	t.Cleanup(func() { slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil))) })

	callAgent := fakeAgentEvents(
		agentEvent{Type: "some_future_type"},
		agentEvent{Type: "done"},
	)
	srv, token := newChatTestServer(t, noopChatCtx, callAgent, noopLoadVerdicts)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "anything", Conversation: testConversationID})

	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.Type != "done" {
		t.Fatalf("event type = %q, want done — the unknown event must not abort the turn", ev.Type)
	}
	if !strings.Contains(buf.String(), "unknown agent event type") {
		t.Errorf("log = %q, want a warning about the unknown event type", buf.String())
	}
}

// A malformed "intent" payload must be logged, not silently swallowed, and
// must not stop the rest of the turn from proceeding.
func TestChatMalformedIntentIsLogged(t *testing.T) {
	var buf bytes.Buffer
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, nil)))
	t.Cleanup(func() { slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil))) })

	callAgent := fakeAgentEvents(
		agentEvent{Type: "intent", Intent: json.RawMessage(`not-json`)},
		agentEvent{Type: "done"},
	)
	srv, token := newChatTestServer(t, noopChatCtx, callAgent, noopLoadVerdicts)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "anything", Conversation: testConversationID})

	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.Type != "done" {
		t.Fatalf("event type = %q, want done — no interpreting event should have been sent", ev.Type)
	}
	if !strings.Contains(buf.String(), "chat intent decode failed") {
		t.Errorf("log = %q, want a warning about the decode failure", buf.String())
	}
}

// interpretingLine must surface mood/exclusion/cast/crew, not just
// genre/year/runtime — these are the fields the model is explicitly directed
// to extract (catalog_tool.py's _INTERPRET_SYSTEM_PROMPT), and dropping them
// silently defeats the line's purpose as a comprehension check.
func TestInterpretingLineSurfacesKeywordsExclusionsAndCastCrew(t *testing.T) {
	intent := agentIntent{
		MediaType:       "movie",
		Keywords:        []string{"heist"},
		WithoutKeywords: []string{"bleak"},
		WithoutGenres:   []string{"horror"},
		Cast:            []string{"Tom Hardy"},
		Crew:            []string{"Christopher Nolan"},
	}
	line := interpretingLine(intent, nil)

	for _, want := range []string{"heist", "bleak", "horror", "Tom Hardy", "Christopher Nolan"} {
		if !strings.Contains(line, want) {
			t.Errorf("interpretingLine() = %q, missing %q", line, want)
		}
	}
}

// TestChatTurnSendsVerdictsToTheAgent proves runTurn forwards what
// loadVerdicts returns on to the agent (TASKS.md T19) — a load of its own,
// not a field on chatContext, which is why this is the one call site that
// passes a real loadVerdicts rather than the noop.
func TestChatTurnSendsVerdictsToTheAgent(t *testing.T) {
	verdicts := []Verdict{
		{TMDBID: 101, MediaType: "movie", Verdict: "seen"},
		{TMDBID: 202, MediaType: "tv", Verdict: "liked"},
	}
	loadCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{Region: "US", Providers: []int{8}}, nil
	}
	loadVerdicts := func(context.Context, string) ([]Verdict, error) {
		return verdicts, nil
	}
	var got agentChatRequest
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		got = req
		events := make(chan agentEvent, 1)
		events <- agentEvent{Type: "done"}
		close(events)
		return events, nil
	}
	srv, token := newChatTestServer(t, loadCtx, callAgent, loadVerdicts)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "something good", Conversation: testConversationID})
	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(got.Verdicts, verdicts) {
		t.Errorf("Verdicts = %+v, want %+v", got.Verdicts, verdicts)
	}
}

// TestChatTurnFailsWhenVerdictsCannotLoad pins the deliberate choice at
// runTurn's verdict load: fail the turn, never degrade to no verdicts.
// Degrading would silently put titles the user marked seen back on screen —
// the one guarantee T19 exists to make — and TASKS.md says outright that the
// chat cannot degrade. Without this test the fix is a comment: every other
// call site here passes a loadVerdicts that cannot fail, so a later change
// to the degrade-shape used elsewhere in this codebase would go unnoticed.
func TestChatTurnFailsWhenVerdictsCannotLoad(t *testing.T) {
	loadCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{Region: "US", Providers: []int{8}}, nil
	}
	loadVerdicts := func(context.Context, string) ([]Verdict, error) {
		return nil, errors.New("verdict read failed")
	}
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		t.Error("agent must not be called when verdicts fail to load")
		return nil, errors.New("unreachable")
	}
	srv, token := newChatTestServer(t, loadCtx, callAgent, loadVerdicts)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "something good", Conversation: testConversationID})

	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.Type != "error" || ev.Text != genericErrorText {
		t.Errorf("event = %+v, want an error event with the generic text", ev)
	}
}

// TestChatTurnWithNoSubscriptionsSkipsVerdicts pins the exception to the
// fail-closed rule above. A caller with nothing ticked gets the agent's "pick
// your services" answer, which never reads verdicts - so a title_verdicts
// problem must not be what stops a brand-new user, who has no verdicts
// anyway, from being onboarded.
func TestChatTurnWithNoSubscriptionsSkipsVerdicts(t *testing.T) {
	loadCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{Region: "US", Providers: []int{}}, nil
	}
	loadVerdicts := func(context.Context, string) ([]Verdict, error) {
		t.Error("verdicts must not be loaded when the caller has no subscriptions")
		return nil, errors.New("title_verdicts unavailable")
	}
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		events := make(chan agentEvent, 2)
		events <- agentEvent{Type: "message", Text: "Pick your services."}
		events <- agentEvent{Type: "done"}
		close(events)
		return events, nil
	}
	srv, token := newChatTestServer(t, loadCtx, callAgent, loadVerdicts)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "hi", Conversation: testConversationID})

	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.Type != "token" || ev.Text != "Pick your services." {
		t.Errorf("event = %+v, want the agent's message forwarded", ev)
	}
}

// --- T20/T20.5: conversation hydration, switching, and persistence ---------

// TestChatConnectionHydratesHistoryFromStoredConversation proves the "message"
// case in runChatConnection seeds its in-memory history from loadConversation
// the first time it sees a given conversation id (TASKS.md T20/T20.5) — a
// reload/reconnect that resends its conversation id on the first message
// carries prior turns into the very first agent call, not just into what the
// browser re-renders.
func TestChatConnectionHydratesHistoryFromStoredConversation(t *testing.T) {
	loadConversation := func(_ context.Context, _ string, conversationID string) ([]historyTurn, error) {
		if conversationID != testConversationID {
			t.Errorf("loadConversation called with %q, want %q", conversationID, testConversationID)
		}
		return []historyTurn{
			{Role: "user", Text: "a heist movie"},
			{Role: "assistant", Text: "Suggested: Heat"},
		}, nil
	}
	loadCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{Region: "US"}, nil
	}
	var got agentChatRequest
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		got = req
		events := make(chan agentEvent, 1)
		events <- agentEvent{Type: "done"}
		close(events)
		return events, nil
	}
	srv, token := newChatTestServerWithConversations(t, loadCtx, callAgent, noopLoadVerdicts,
		loadConversation, noopSaveMessages)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "something shorter", Conversation: testConversationID})
	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}

	want := []historyTurn{{Role: "user", Text: "a heist movie"}, {Role: "assistant", Text: "Suggested: Heat"}}
	if !sameTurns(got.History, want) {
		t.Errorf("History sent to the agent = %+v, want the stored conversation %+v", got.History, want)
	}
}

// TestChatCompletedTurnPersistsViaSaveMessages proves a successfully
// completed turn calls saveMessages with the conversation id the client sent,
// the exact text shown, and the title ids from its picks — DECISIONS.md's
// "clean text and the title IDs shown."
func TestChatCompletedTurnPersistsViaSaveMessages(t *testing.T) {
	loadCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{Region: "US", Providers: []int{8}}, nil
	}
	callAgent := fakeAgentEvents(
		agentEvent{Type: "results", Picks: []agentPick{
			{TMDBID: 550, MediaType: "movie", Title: "Fight Club"},
		}},
		agentEvent{Type: "done"},
	)

	type saveCall struct {
		conversationID, userText, assistantText string
		titleRefs                               []agentTitleRef
	}
	saved := make(chan saveCall, 1)
	saveMessages := func(_ context.Context, _ string, conversationID, userText, assistantText string, titleRefs []agentTitleRef) (bool, error) {
		saved <- saveCall{conversationID, userText, assistantText, titleRefs}
		return false, nil
	}
	srv, token := newChatTestServerWithConversations(t, loadCtx, callAgent, noopLoadVerdicts,
		noopLoadConversation, saveMessages)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "a heist movie", Conversation: testConversationID})
	for range 2 { // "results", then "done"
		var ev outboundEvent
		if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
			t.Fatal(err)
		}
	}

	select {
	case call := <-saved:
		if call.conversationID != testConversationID {
			t.Errorf("conversationID = %q, want %q", call.conversationID, testConversationID)
		}
		if call.userText != "a heist movie" {
			t.Errorf("userText = %q, want %q", call.userText, "a heist movie")
		}
		if call.assistantText != "Suggested: Fight Club" {
			t.Errorf("assistantText = %q, want %q", call.assistantText, "Suggested: Fight Club")
		}
		if len(call.titleRefs) != 1 || call.titleRefs[0].TMDBID != 550 || call.titleRefs[0].MediaType != "movie" {
			t.Errorf("titleRefs = %+v, want [{550 movie}]", call.titleRefs)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("saveMessages was never called")
	}
}

// TestChatBackToBackTurnsPersistInOrder proves two turns on the same
// conversation, completed one after another, persist in the order they
// completed, each carrying the conversation id the client sent.
func TestChatBackToBackTurnsPersistInOrder(t *testing.T) {
	loadCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{Region: "US"}, nil
	}
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		events := make(chan agentEvent, 2)
		events <- agentEvent{Type: "message", Text: "ok " + req.Message}
		events <- agentEvent{Type: "done"}
		close(events)
		return events, nil
	}
	type saveCall struct{ userText, conversationID string }
	saved := make(chan saveCall, 2)
	saveMessages := func(_ context.Context, _ string, conversationID, userText, assistantText string, titleRefs []agentTitleRef) (bool, error) {
		saved <- saveCall{userText: userText, conversationID: conversationID}
		return false, nil
	}
	srv, token := newChatTestServerWithConversations(t, loadCtx, callAgent, noopLoadVerdicts,
		noopLoadConversation, saveMessages)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	for _, turn := range []string{"t1", "t2"} {
		if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: turn, Text: turn, Conversation: testConversationID}); err != nil {
			t.Fatal(err)
		}
		for range 2 { // "token", then "done"
			var ev outboundEvent
			if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
				t.Fatal(err)
			}
		}
	}

	want := []saveCall{
		{userText: "t1", conversationID: testConversationID},
		{userText: "t2", conversationID: testConversationID},
	}
	for _, want := range want {
		select {
		case got := <-saved:
			if got != want {
				t.Errorf("save = %+v, want %+v", got, want)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("save for %q never completed", want.userText)
		}
	}
}

// TestChatDisconnectAfterResultsButBeforeDonePersists guards a wider version
// of the same disconnect race TestChatBackToBackTurnsPersistInOrder's
// neighboring fix addresses: a turn can have rec.ok already true from an
// earlier "results"/"message" event well before it reaches "done" — the
// fake agent here deliberately never sends "done", simulating exactly that
// window — so a disconnect must still persist it via the cancel-then-wait
// path in runChatConnection's `!ok` branch, not just the narrower
// already-on-the-channel case a non-blocking peek alone would catch.
func TestChatDisconnectAfterResultsButBeforeDonePersists(t *testing.T) {
	loadCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{Region: "US"}, nil
	}
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		events := make(chan agentEvent, 1)
		events <- agentEvent{Type: "message", Text: "partial answer"}
		// No "done" — mirrors newAgentCaller's real shape, where an
		// aborted HTTP request (turnCtx cancelled) is what eventually
		// closes the channel, not a well-formed terminal event.
		go func() {
			<-ctx.Done()
			close(events)
		}()
		return events, nil
	}
	saved := make(chan string, 1)
	saveMessages := func(_ context.Context, _ string, _ string, _ string, assistantText string, _ []agentTitleRef) (bool, error) {
		saved <- assistantText
		return false, nil
	}
	srv, token := newChatTestServerWithConversations(t, loadCtx, callAgent, noopLoadVerdicts,
		noopLoadConversation, saveMessages)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "hi", Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}
	// Wait for the "message" event to actually reach the browser before
	// disconnecting, so rec.ok is already true — the exact state this test
	// exists to cover — when the socket closes.
	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.Type != "token" {
		t.Fatalf("got %+v, want a token event before disconnecting", ev)
	}
	conn.CloseNow()

	select {
	case got := <-saved:
		if got != "partial answer" {
			t.Errorf("persisted text = %q, want %q", got, "partial answer")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("turn was not persisted after disconnecting mid-stream")
	}
}

// TestChatDisconnectPersistsBothSupersededAndCurrentTurns proves both a
// superseded turn and the turn that superseded it end up persisted after a
// disconnect — a superseded turn's goroutine keeps running after
// cancelCurrent (it still has to unwind and report), so more than one can be
// outstanding at once. Server-side disconnect-detection timing isn't
// controllable from this black-box test, so this doesn't reliably fail
// against the narrower single-receive-plus-one-peek drain it replaced the
// way TestChatDisconnectAfterResultsButBeforeDonePersists does — it's a
// correctness check on the end state, not a proven regression guard for
// this specific race.
func TestChatDisconnectPersistsBothSupersededAndCurrentTurns(t *testing.T) {
	loadCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{Region: "US"}, nil
	}
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		events := make(chan agentEvent, 2)
		if req.Message == "A" {
			events <- agentEvent{Type: "message", Text: "answer A"}
			// Never sends "done" on its own — only unwinds once superseded
			// (turnCtx cancelled), same shape as
			// TestChatDisconnectAfterResultsButBeforeDonePersists.
			go func() {
				<-ctx.Done()
				close(events)
			}()
		} else {
			events <- agentEvent{Type: "message", Text: "answer B"}
			// Delayed well past any plausible disconnect-detection latency:
			// forces the disconnect below to land while B is still
			// mid-stream, not yet at its own deferred send — exactly the
			// interleaving a single non-blocking peek would miss.
			go func() {
				time.Sleep(500 * time.Millisecond)
				events <- agentEvent{Type: "done"}
				close(events)
			}()
		}
		return events, nil
	}
	saved := make(chan string, 2)
	saveMessages := func(_ context.Context, _ string, _ string, _ string, assistantText string, _ []agentTitleRef) (bool, error) {
		saved <- assistantText
		return false, nil
	}
	srv, token := newChatTestServerWithConversations(t, loadCtx, callAgent, noopLoadVerdicts,
		noopLoadConversation, saveMessages)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "tA", Text: "A", Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}
	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil { // A's token
		t.Fatal(err)
	}
	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "tB", Text: "B", Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil { // B's token
		t.Fatal(err)
	}
	conn.CloseNow() // B is still 500ms from its own "done"; A is still unwinding.

	got := map[string]bool{}
	for range 2 {
		select {
		case text := <-saved:
			got[text] = true
		case <-time.After(2 * time.Second):
			t.Fatalf("only persisted %v, want both answer A and answer B", got)
		}
	}
	if !got["answer A"] || !got["answer B"] {
		t.Errorf("persisted = %v, want both answer A and answer B", got)
	}
}

// TestChatSwitchingConversationsDoesNotBleedHistoryOrPersistence is the
// regression guard for T20.5's core hazard: a turn still in flight on one
// conversation when the user switches to another must (a) still persist
// against the conversation it actually ran on, never the one that's current
// by the time it finishes, and (b) never have its answer merged into the
// in-memory history window the new conversation's next turn sees.
func TestChatSwitchingConversationsDoesNotBleedHistoryOrPersistence(t *testing.T) {
	const convX = "22222222-2222-2222-2222-222222222222"
	const convY = "33333333-3333-3333-3333-333333333333"
	ySeed := []historyTurn{{Role: "user", Text: "seed"}, {Role: "assistant", Text: "seed-reply"}}

	loadConversation := func(_ context.Context, _ string, conversationID string) ([]historyTurn, error) {
		if conversationID == convY {
			return append([]historyTurn(nil), ySeed...), nil
		}
		return nil, nil
	}
	loadCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{Region: "US"}, nil
	}
	gotYReq := make(chan agentChatRequest, 1)
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		if req.Message == "A" {
			events := make(chan agentEvent, 1)
			events <- agentEvent{Type: "message", Text: "answer A"}
			// Only unwinds once superseded by B's switch — never sends
			// "done" on its own, same shape as the disconnect tests above.
			go func() {
				<-ctx.Done()
				close(events)
			}()
			return events, nil
		}
		gotYReq <- req
		events := make(chan agentEvent, 2)
		events <- agentEvent{Type: "message", Text: "answer B"}
		events <- agentEvent{Type: "done"}
		close(events)
		return events, nil
	}
	type saveCall struct{ conversationID, userText, assistantText string }
	saved := make(chan saveCall, 2)
	saveMessages := func(_ context.Context, _ string, conversationID, userText, assistantText string, _ []agentTitleRef) (bool, error) {
		saved <- saveCall{conversationID, userText, assistantText}
		return false, nil
	}
	srv, token := newChatTestServerWithConversations(t, loadCtx, callAgent, noopLoadVerdicts,
		loadConversation, saveMessages)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "tA", Text: "A", Conversation: convX}); err != nil {
		t.Fatal(err)
	}
	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil { // A's token — proves A is dispatched before B switches away
		t.Fatal(err)
	}

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "tB", Text: "B", Conversation: convY}); err != nil {
		t.Fatal(err)
	}
	for range 2 { // B's "token", then "done"
		if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
			t.Fatal(err)
		}
	}

	select {
	case req := <-gotYReq:
		if !sameTurns(req.History, ySeed) {
			t.Errorf("B's History = %+v, want conv Y's own seed %+v (not A's exchange)", req.History, ySeed)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the agent was never called for conversation Y")
	}

	want := map[string]saveCall{
		convX: {conversationID: convX, userText: "A", assistantText: "answer A"},
		convY: {conversationID: convY, userText: "B", assistantText: "answer B"},
	}
	for range 2 {
		select {
		case got := <-saved:
			if w, ok := want[got.conversationID]; !ok || got != w {
				t.Errorf("save = %+v, unexpected or wrong for its conversation", got)
			} else {
				delete(want, got.conversationID)
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("only got saves for %v, missing the rest", want)
		}
	}
	if len(want) != 0 {
		t.Errorf("missing saves for: %v", want)
	}
}

// TestChatMessageRejectsAMalformedConversationID proves the WS "message" case
// validates the conversation id the same way the REST handlers do (uuid.Parse)
// rather than only checking for an empty string — a malformed id must never
// reach loadConversation/callAgent, since conversations.id is a uuid column
// and letting a bad value through would surface as a database type-cast error
// instead of a clean signal (see runChatConnection's "message" case).
func TestChatMessageRejectsAMalformedConversationID(t *testing.T) {
	for _, conv := range []string{"", "not-a-uuid", "11111111-1111-1111-1111-11111111111"} {
		t.Run(conv, func(t *testing.T) {
			loadConversation := func(context.Context, string, string) ([]historyTurn, error) {
				t.Error("loadConversation must not be called for a malformed conversation id")
				return nil, nil
			}
			callAgent := func(context.Context, agentChatRequest) (<-chan agentEvent, error) {
				t.Error("the agent must not be called for a malformed conversation id")
				return nil, errors.New("unreachable")
			}
			srv, token := newChatTestServerWithConversations(t, noopChatCtx, callAgent, noopLoadVerdicts,
				loadConversation, noopSaveMessages)
			conn := dialChat(t, srv, mintTicket(t, srv, token))

			if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "hi", Conversation: conv}); err != nil {
				t.Fatal(err)
			}
			var ev outboundEvent
			if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
				t.Fatal(err)
			}
			if ev.Type != "error" || ev.Text != genericErrorText {
				t.Errorf("event = %+v, want a generic error event", ev)
			}
		})
	}
}

// TestChatMessageRejectsAnEmptyTurnID mirrors
// TestChatMessageRejectsAMalformedConversationID for msg.Turn — the
// regression guard for T32's nil-cancelCurrent panic, whose root cause was
// letting "" (the coordinator's own "no turn active" sentinel) double as a
// real turn id. Only emptiness is rejected — unlike conversationID, turn is
// never parsed as a uuid, and the test suite's own convention of plain
// placeholder ids ("t1", "t2", ...) is deliberately still valid.
func TestChatMessageRejectsAnEmptyTurnID(t *testing.T) {
	callAgent := func(context.Context, agentChatRequest) (<-chan agentEvent, error) {
		t.Error("the agent must not be called for an empty turn id")
		return nil, errors.New("unreachable")
	}
	srv, token := newChatTestServer(t, noopChatCtx, callAgent, noopLoadVerdicts)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "", Text: "hi", Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}
	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.Type != "error" || ev.Text != genericErrorText {
		t.Errorf("event = %+v, want a generic error event", ev)
	}
}

// TestChatDuplicateTurnIDsDoNotPanic is the direct regression guard for T32's
// nil-cancelCurrent panic: two "message" frames sharing the same (now
// UUID-valid) turn id must not crash the connection, even though only the
// first should ever be "current" by the time the second's done-record
// arrives.
func TestChatDuplicateTurnIDsDoNotPanic(t *testing.T) {
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		ch := make(chan agentEvent, 1)
		ch <- agentEvent{Type: "done"}
		close(ch)
		return ch, nil
	}
	srv, token := newChatTestServer(t, noopChatCtx, callAgent, noopLoadVerdicts)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	turn := uuid.NewString()
	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: turn, Text: "first", Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}
	var first outboundEvent
	if err := wsjson.Read(t.Context(), conn, &first); err != nil {
		t.Fatal(err)
	}
	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: turn, Text: "second", Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}
	var second outboundEvent
	if err := wsjson.Read(t.Context(), conn, &second); err != nil {
		t.Fatalf("connection did not survive a duplicate turn id: %v", err)
	}
	if second.Type != "done" {
		t.Errorf("second event = %+v, want done", second)
	}
}

// TestChatOverLongMessageIsRejected proves maxMessageLength is enforced
// server-side, independent of whatever cap the client applies — a client bug
// or a non-browser caller must not be able to send a frame large enough to
// trip coder/websocket's own read limit and silently kill the connection.
func TestChatOverLongMessageIsRejected(t *testing.T) {
	callAgent := func(context.Context, agentChatRequest) (<-chan agentEvent, error) {
		t.Error("the agent must not be called for an over-long message")
		return nil, errors.New("unreachable")
	}
	srv, token := newChatTestServer(t, noopChatCtx, callAgent, noopLoadVerdicts)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	over := strings.Repeat("a", maxMessageLength+1)
	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: over, Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}
	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.Type != "error" {
		t.Errorf("event = %+v, want type error", ev)
	}
}

// TestChatTurnDeadlineSendsATerminalErrorEvent is the direct regression test
// for T32's original bug: a turn that never comes back before turnDeadline
// must still end with a terminal event, not silence. Shortens the package
// var so the test doesn't wait a real 30s.
func TestChatTurnDeadlineSendsATerminalErrorEvent(t *testing.T) {
	original := turnDeadline
	turnDeadline = 50 * time.Millisecond
	t.Cleanup(func() { turnDeadline = original })

	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		<-ctx.Done() // never replies before turnDeadline fires
		return nil, ctx.Err()
	}
	srv, token := newChatTestServer(t, noopChatCtx, callAgent, noopLoadVerdicts)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "anything", Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}
	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.Type != "error" || ev.Turn != "t1" {
		t.Errorf("event = %+v, want a t1 error event", ev)
	}
}

// TestFinishTurnSurfacesAnRLSViolationAsAVisibleError proves the other half
// of T32's "foreign conversation id" fix: when saveMessages fails because
// this session's RLS policy rejected the write (SQLSTATE 42501 — a
// conversation foreign to this caller, or deleted between dispatch and
// persist), the browser gets a clear, visible error rather than a silent
// log — the turn's answer was already shown but will otherwise never be
// saved with nothing to tell the user so.
func TestFinishTurnSurfacesAnRLSViolationAsAVisibleError(t *testing.T) {
	loadCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{Region: "US", Providers: []int{8}}, nil
	}
	callAgent := fakeAgentEvents(
		agentEvent{Type: "results", Picks: []agentPick{{TMDBID: 550, MediaType: "movie", Title: "Fight Club"}}},
		agentEvent{Type: "done"},
	)
	saveMessages := func(context.Context, string, string, string, string, []agentTitleRef) (bool, error) {
		return false, &pgconn.PgError{Code: "42501", Message: "new row violates row-level security policy"}
	}
	srv, token := newChatTestServerWithConversations(t, loadCtx, callAgent, noopLoadVerdicts,
		noopLoadConversation, saveMessages)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "a heist movie", Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}

	var got []outboundEvent
	for range 3 { // "results", "done", then the persist-failure error
		var ev outboundEvent
		if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
			t.Fatalf("read event[%d]: %v (got so far: %+v)", len(got), err, got)
		}
		got = append(got, ev)
	}
	last := got[len(got)-1]
	if last.Type != "error" || last.Text == "" {
		t.Fatalf("last event = %+v, want a non-empty visible error", last)
	}
	if last.Text == genericErrorText {
		t.Errorf("error text = %q, want the specific RLS-rejection copy, not the generic fallback", last.Text)
	}
}

// TestFinishTurnSurfacesANotWritableConversationAsAVisibleError proves the
// T34 path alongside TestFinishTurnSurfacesAnRLSViolationAsAVisibleError:
// saveMessages' own explicit ownership check (errConversationNotWritable) is
// what actually fires for a foreign or deleted conversation id today, and it
// must reach the browser exactly like the RLS backstop does — logged under a
// stable message (not the generic, error-text-carrying "message persist
// failed" line), so an operator's alert on this case survives regardless of
// which of the two independent layers caught it.
func TestFinishTurnSurfacesANotWritableConversationAsAVisibleError(t *testing.T) {
	var buf bytes.Buffer
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, nil)))
	t.Cleanup(func() { slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, nil))) })

	loadCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{Region: "US", Providers: []int{8}}, nil
	}
	callAgent := fakeAgentEvents(
		agentEvent{Type: "results", Picks: []agentPick{{TMDBID: 550, MediaType: "movie", Title: "Fight Club"}}},
		agentEvent{Type: "done"},
	)
	saveMessages := func(context.Context, string, string, string, string, []agentTitleRef) (bool, error) {
		return false, errConversationNotWritable
	}
	srv, token := newChatTestServerWithConversations(t, loadCtx, callAgent, noopLoadVerdicts,
		noopLoadConversation, saveMessages)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "a heist movie", Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}

	var got []outboundEvent
	for range 3 { // "results", "done", then the persist-failure error
		var ev outboundEvent
		if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
			t.Fatalf("read event[%d]: %v (got so far: %+v)", len(got), err, got)
		}
		got = append(got, ev)
	}
	last := got[len(got)-1]
	if last.Type != "error" || last.Text == "" {
		t.Fatalf("last event = %+v, want a non-empty visible error", last)
	}
	if last.Text == genericErrorText {
		t.Errorf("error text = %q, want the specific rejection copy, not the generic fallback", last.Text)
	}
	if !strings.Contains(buf.String(), "message persist rejected: conversation not writable") {
		t.Errorf("log = %q, want the stable rejection message", buf.String())
	}
}

// --- conversationDeletions -------------------------------------------------

// TestConversationDeletionsNeverForgetsARealDelete is the regression guard
// for a previous, expiring implementation: a real delete's generation must
// never revert to "never deleted," no matter how long afterward it's
// checked — an earlier version aged it out after a fixed TTL, and a
// long-lived connection's own in-memory state could then compare clean
// again against a conversation it had already been told was deleted,
// silently serving pre-deletion history as agent context.
func TestConversationDeletionsNeverForgetsARealDelete(t *testing.T) {
	c := newConversationDeletions()
	c.bump("conv1")
	if got := c.generation("conv1"); got != 1 {
		t.Fatalf("generation = %d, want 1", got)
	}
	if got := c.generation("conv1"); got != 1 {
		t.Errorf("generation = %d, want still 1 (a real delete must never be forgotten)", got)
	}
}

func TestConversationDeletionsBumpIncrementsOnEachRealDelete(t *testing.T) {
	c := newConversationDeletions()
	if got := c.bump("conv1"); got != 1 {
		t.Errorf("bump = %d, want 1", got)
	}
	if got := c.bump("conv1"); got != 2 {
		t.Errorf("bump = %d, want 2", got)
	}
}

// --- finishTurn --------------------------------------------------------------

// TestFinishTurnSkipsPersistingAStragglerFromBeforeItsConversationsDeletion
// covers the original delete race: a turn dispatched before its conversation
// was deleted (captured generation 0) must be suppressed once the generation
// has since advanced.
func TestFinishTurnSkipsPersistingAStragglerFromBeforeItsConversationsDeletion(t *testing.T) {
	var saveCalled bool
	h := &Handler{
		saveMessages: func(context.Context, string, string, string, string, []agentTitleRef) (bool, error) {
			saveCalled = true
			return false, nil
		},
		conversationDeletions: newConversationDeletions(),
	}
	h.conversationDeletions.bump(testConversationID) // generation -> 1, after dispatch

	rec := turnRecord{turn: "t1", conversation: testConversationID, deletionGen: 0, ok: true, userText: "hi", assistantText: "hello"}
	h.finishTurn(t.Context(), t.Context(), "user_1", rec, nil)

	if saveCalled {
		t.Error("finishTurn persisted a turn dispatched before its conversation's deletion")
	}
}

// TestFinishTurnPersistsANewMessageDispatchedAfterItsConversationsDeletion is
// the regression guard for the "browser back button to a since-deleted
// conversation" hazard: a turn dispatched *after* a delete captures the
// post-delete generation itself, so it must never be suppressed.
func TestFinishTurnPersistsANewMessageDispatchedAfterItsConversationsDeletion(t *testing.T) {
	var saveCalled bool
	h := &Handler{
		saveMessages: func(context.Context, string, string, string, string, []agentTitleRef) (bool, error) {
			saveCalled = true
			return false, nil
		},
		conversationDeletions: newConversationDeletions(),
	}
	gen := h.conversationDeletions.bump(testConversationID) // deleted, then...

	rec := turnRecord{turn: "t1", conversation: testConversationID, deletionGen: gen, ok: true, userText: "hi", assistantText: "hello"} // ...a fresh dispatch captures the post-delete generation
	h.finishTurn(t.Context(), t.Context(), "user_1", rec, nil)

	if !saveCalled {
		t.Error("finishTurn suppressed a legitimate turn dispatched after its conversation's deletion")
	}
}

func TestFinishTurnPersistsWhenNeverDeleted(t *testing.T) {
	var saveCalled bool
	h := &Handler{
		saveMessages: func(context.Context, string, string, string, string, []agentTitleRef) (bool, error) {
			saveCalled = true
			return false, nil
		},
		conversationDeletions: newConversationDeletions(),
	}
	rec := turnRecord{turn: "t1", conversation: testConversationID, ok: true, userText: "hi", assistantText: "hello"}
	h.finishTurn(t.Context(), t.Context(), "user_1", rec, nil)

	if !saveCalled {
		t.Error("finishTurn did not persist an ordinary successful turn")
	}
}

// TestDeleteConversationPreventsAStragglingTurnFromResurrectingIt is the
// regression guard for the delete race: a turn still in flight on a
// conversation when it's deleted must not have its own completion resurrect
// that conversation via saveMessages' on-conflict-do-nothing insert. The
// DELETE is forced to land strictly before the turn's own "done" via the
// proceed gate below, so this proves the tombstone is actually consulted, not
// just that events happened to land in a hoped-for order.
func TestDeleteConversationPreventsAStragglingTurnFromResurrectingIt(t *testing.T) {
	proceed := make(chan struct{})
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		events := make(chan agentEvent, 2)
		events <- agentEvent{Type: "message", Text: "partial answer"}
		go func() {
			<-proceed
			events <- agentEvent{Type: "done"}
			close(events)
		}()
		return events, nil
	}
	saveMessages := func(context.Context, string, string, string, string, []agentTitleRef) (bool, error) {
		t.Error("a straggling turn resurrected a just-deleted conversation via saveMessages")
		return false, nil
	}
	srv, token := newChatTestServerWithConversations(t, noopChatCtx, callAgent, noopLoadVerdicts,
		noopLoadConversation, saveMessages)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "hi", Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}
	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.Type != "token" {
		t.Fatalf("got %+v, want a token event before deleting", ev)
	}

	req, err := http.NewRequest(http.MethodDelete, srv.URL+"/api/conversations/"+testConversationID, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("delete status = %d, want 200", resp.StatusCode)
	}

	close(proceed) // let the turn finish, strictly after the delete landed
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.Type != "done" {
		t.Fatalf("event type = %q, want done", ev.Type)
	}

	time.Sleep(100 * time.Millisecond) // margin for finishTurn to run server-side
}

// TestDeleteConversationTombstoneAppliesRegardlessOfIDCase is the regression
// guard for case-normalization: conversationDeletions is keyed by the
// parsed-and-reserialized uuid, not the caller's raw string, so a delete
// sent under one casing of a conversation id must still tombstone a
// straggling turn dispatched under a differently-cased (but equal) uuid.
func TestDeleteConversationTombstoneAppliesRegardlessOfIDCase(t *testing.T) {
	const lower = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
	const upper = "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"

	proceed := make(chan struct{})
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		events := make(chan agentEvent, 2)
		events <- agentEvent{Type: "message", Text: "partial answer"}
		go func() {
			<-proceed
			events <- agentEvent{Type: "done"}
			close(events)
		}()
		return events, nil
	}
	saveMessages := func(context.Context, string, string, string, string, []agentTitleRef) (bool, error) {
		t.Error("a straggling turn resurrected a just-deleted conversation via saveMessages")
		return false, nil
	}
	srv, token := newChatTestServerWithConversations(t, noopChatCtx, callAgent, noopLoadVerdicts,
		noopLoadConversation, saveMessages)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "hi", Conversation: lower}); err != nil {
		t.Fatal(err)
	}
	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.Type != "token" {
		t.Fatalf("got %+v, want a token event before deleting", ev)
	}

	req, err := http.NewRequest(http.MethodDelete, srv.URL+"/api/conversations/"+upper, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("delete status = %d, want 200", resp.StatusCode)
	}

	close(proceed)
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.Type != "done" {
		t.Fatalf("event type = %q, want done", ev.Type)
	}
	time.Sleep(100 * time.Millisecond) // margin for finishTurn to run server-side
}

// TestChatTurnDegradesToEmptyHistoryWhenConversationLoadFails proves a
// conversation-history load failure degrades to an empty history window
// rather than failing the turn outright — history is a nice-to-have for the
// agent (loadConversation's own contract already treats "nothing saved yet"
// as a normal nil result), not a hard requirement like T19's verdicts, so a
// transient read error here must not stop the user from getting an answer.
func TestChatTurnDegradesToEmptyHistoryWhenConversationLoadFails(t *testing.T) {
	loadConversation := func(context.Context, string, string) ([]historyTurn, error) {
		return nil, errors.New("transient db error")
	}
	var got agentChatRequest
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		got = req
		ch := make(chan agentEvent, 1)
		ch <- agentEvent{Type: "done"}
		close(ch)
		return ch, nil
	}
	srv, token := newChatTestServerWithConversations(t, noopChatCtx, callAgent, noopLoadVerdicts,
		loadConversation, noopSaveMessages)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "hi", Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}
	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.Type != "done" {
		t.Fatalf("event type = %q, want done — a history load failure must not fail the turn", ev.Type)
	}
	if len(got.History) != 0 {
		t.Errorf("History sent to the agent = %+v, want empty after a failed load", got.History)
	}
}

// TestChatNewMessageToASinceDeletedConversationIsNotDropped is the
// end-to-end regression guard for the "browser back button to a
// since-deleted conversation" hazard: a brand-new, legitimate message sent
// to a conversation id after it was deleted must persist normally, not be
// silently dropped by the tombstone built to catch a pre-delete straggler.
func TestChatNewMessageToASinceDeletedConversationIsNotDropped(t *testing.T) {
	saved := make(chan string, 1)
	saveMessages := func(_ context.Context, _ string, _ string, _ string, assistantText string, _ []agentTitleRef) (bool, error) {
		saved <- assistantText
		return false, nil
	}
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		ch := make(chan agentEvent, 2)
		ch <- agentEvent{Type: "message", Text: "welcome back"}
		ch <- agentEvent{Type: "done"}
		close(ch)
		return ch, nil
	}
	srv, token := newChatTestServerWithConversations(t, noopChatCtx, callAgent, noopLoadVerdicts,
		noopLoadConversation, saveMessages)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	req, err := http.NewRequest(http.MethodDelete, srv.URL+"/api/conversations/"+testConversationID, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("delete status = %d, want 200", resp.StatusCode)
	}

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "hi again", Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}
	for range 2 { // token, then done
		var ev outboundEvent
		if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
			t.Fatal(err)
		}
	}

	select {
	case got := <-saved:
		if got != "welcome back" {
			t.Errorf("persisted text = %q, want %q", got, "welcome back")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("a legitimate new message to a since-deleted conversation id was dropped")
	}
}

// TestChatDeleteDuringInFlightLoadStillSuppressesTheStragglingTurn is the
// regression guard for capturing deletionGen too late: loadConversation here
// blocks until the test's own DELETE has already landed and bumped the
// generation. A capture of deletionGen taken *after* the load returns (as
// the code used to do) would read the already-bumped generation and wrongly
// treat this turn as not predating the delete, letting it persist and
// resurrect the conversation.
func TestChatDeleteDuringInFlightLoadStillSuppressesTheStragglingTurn(t *testing.T) {
	loadStarted := make(chan struct{})
	deleteDone := make(chan struct{})
	loadConversation := func(context.Context, string, string) ([]historyTurn, error) {
		close(loadStarted)
		<-deleteDone
		return nil, nil
	}
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		ch := make(chan agentEvent, 2)
		ch <- agentEvent{Type: "message", Text: "hello"}
		ch <- agentEvent{Type: "done"}
		close(ch)
		return ch, nil
	}
	saveMessages := func(context.Context, string, string, string, string, []agentTitleRef) (bool, error) {
		t.Error("a turn whose load raced a delete of its own conversation must not persist")
		return false, nil
	}
	srv, token := newChatTestServerWithConversations(t, noopChatCtx, callAgent, noopLoadVerdicts,
		loadConversation, saveMessages)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "hi", Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}
	<-loadStarted // deletionGen must already be captured by now

	req, err := http.NewRequest(http.MethodDelete, srv.URL+"/api/conversations/"+testConversationID, nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("delete status = %d, want 200", resp.StatusCode)
	}
	close(deleteDone) // let the load, and the turn behind it, proceed

	var ev outboundEvent
	for range 2 { // "token", then "done"
		if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
			t.Fatal(err)
		}
	}
	time.Sleep(100 * time.Millisecond) // margin for finishTurn to run server-side
}

// --- conversation_created event ----------------------------------------------
//
// Findings traced to the sidebar-refresh signal being inferred client-side,
// from turn-id bookkeeping and the "done" event's timing — neither of which
// survives the dispatching ChatPanel unmounting, avoids racing finishTurn's
// own persist, or fires when the stream drops before "done". These tests
// cover the replacement: finishTurn sends "conversation_created" itself,
// once, strictly after saveMessages reports it actually created the row.

// TestChatConversationCreatedFiresOnceOnFirstSuccessfulPersist proves the
// event fires exactly once, on the turn whose persist actually created the
// conversation, and never again for a later turn on the same conversation.
func TestChatConversationCreatedFiresOnceOnFirstSuccessfulPersist(t *testing.T) {
	callAgent := fakeAgentEvents(
		agentEvent{Type: "message", Text: "hello"},
		agentEvent{Type: "done"},
	)
	var calls int
	saveMessages := func(context.Context, string, string, string, string, []agentTitleRef) (bool, error) {
		calls++
		return calls == 1, nil // created only on the first call
	}
	srv, token := newChatTestServerWithConversations(t, noopChatCtx, callAgent, noopLoadVerdicts,
		noopLoadConversation, saveMessages)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "t1", Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}
	var got []string
	for range 3 { // "token", "done", then the new event
		var ev outboundEvent
		if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
			t.Fatal(err)
		}
		got = append(got, ev.Type)
	}
	if want := []string{"token", "done", "conversation_created"}; !slices.Equal(got, want) {
		t.Fatalf("first turn's events = %v, want %v", got, want)
	}

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t2", Text: "t2", Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}
	got = nil
	for range 2 { // "token", "done" — no repeat conversation_created
		var ev outboundEvent
		if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
			t.Fatal(err)
		}
		got = append(got, ev.Type)
	}
	if want := []string{"token", "done"}; !slices.Equal(got, want) {
		t.Errorf("second turn's events = %v, want %v (no repeat conversation_created)", got, want)
	}
}

// TestChatConversationCreatedFiresEvenWhenTheStreamDropsMidFlight is the
// regression guard for the dropped-stream case: a turn that already produced
// real content (rec.ok true from an earlier "message" event) but whose agent
// stream then drops before "done" or "error" ever arrives is still persisted
// by finishTurn (runTurn's dropped-stream fallback sends only "error") — and
// must still tell the browser to refresh the sidebar, even though "done"
// itself never fired.
func TestChatConversationCreatedFiresEvenWhenTheStreamDropsMidFlight(t *testing.T) {
	callAgent := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		ch := make(chan agentEvent, 1)
		ch <- agentEvent{Type: "message", Text: "partial answer"}
		close(ch) // no "done", no "error" — the stream just stops
		return ch, nil
	}
	saveMessages := func(context.Context, string, string, string, string, []agentTitleRef) (bool, error) {
		return true, nil
	}
	srv, token := newChatTestServerWithConversations(t, noopChatCtx, callAgent, noopLoadVerdicts,
		noopLoadConversation, saveMessages)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "hi", Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}
	var got []string
	for range 3 { // "token", the dropped-stream fallback "error", then the new event
		var ev outboundEvent
		if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
			t.Fatal(err)
		}
		got = append(got, ev.Type)
	}
	if want := []string{"token", "error", "conversation_created"}; !slices.Equal(got, want) {
		t.Errorf("event types = %v, want %v", got, want)
	}
}

// TestChatConversationCreatedIsNotSentWhenPersistFails proves finishTurn
// never announces a conversation the database didn't actually end up with —
// saveMessages' own error takes priority over its created return value.
func TestChatConversationCreatedIsNotSentWhenPersistFails(t *testing.T) {
	callAgent := fakeAgentEvents(
		agentEvent{Type: "message", Text: "ok"},
		agentEvent{Type: "done"},
	)
	saveMessages := func(context.Context, string, string, string, string, []agentTitleRef) (bool, error) {
		return true, errors.New("db unavailable")
	}
	srv, token := newChatTestServerWithConversations(t, noopChatCtx, callAgent, noopLoadVerdicts,
		noopLoadConversation, saveMessages)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "hi", Conversation: testConversationID}); err != nil {
		t.Fatal(err)
	}
	var got []string
	for range 2 { // "token", "done" — never a conversation_created for a failed persist
		var ev outboundEvent
		if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
			t.Fatal(err)
		}
		got = append(got, ev.Type)
	}
	if want := []string{"token", "done"}; !slices.Equal(got, want) {
		t.Errorf("event types = %v, want %v", got, want)
	}

	readCtx, cancel := context.WithTimeout(t.Context(), 300*time.Millisecond)
	defer cancel()
	var ev outboundEvent
	if err := wsjson.Read(readCtx, conn, &ev); err == nil {
		t.Errorf("unexpected event after a failed persist: %+v", ev)
	}
}

// TestFinishTurnSkipsTheWebSocketSendOnceItsEventContextIsDone is the
// regression guard for the shutdown-drain path silently defeating
// sendTerminalEvent's own cancellation guard: finishTurn must still persist
// via dbCtx even when eventCtx is already done, but must not attempt the WS
// send that eventCtx being done exists to suppress.
func TestFinishTurnSkipsTheWebSocketSendOnceItsEventContextIsDone(t *testing.T) {
	serverConnCh := make(chan *websocket.Conn, 1)
	handlerDone := make(chan struct{})
	t.Cleanup(func() { close(handlerDone) })
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		serverConnCh <- conn
		<-handlerDone // keep the handler alive; the test closes both ends
	}))
	defer srv.Close()

	clientConn, _, err := websocket.Dial(t.Context(), strings.Replace(srv.URL, "http://", "ws://", 1), nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer clientConn.CloseNow()
	serverConn := <-serverConnCh
	defer serverConn.CloseNow()

	var saveCalled bool
	h := &Handler{
		saveMessages: func(context.Context, string, string, string, string, []agentTitleRef) (bool, error) {
			saveCalled = true
			return true, nil // "created" — finishTurn would send conversation_created if allowed to
		},
		conversationDeletions: newConversationDeletions(),
	}
	rec := turnRecord{turn: "t1", conversation: testConversationID, ok: true, userText: "hi", assistantText: "hello"}

	eventCtx, cancel := context.WithCancel(t.Context())
	cancel() // already done, as runChatConnection's shutdown drain leaves it

	h.finishTurn(t.Context(), eventCtx, "user_1", rec, serverConn)

	if !saveCalled {
		t.Error("finishTurn must still persist via dbCtx even when eventCtx is already done")
	}

	readCtx, readCancel := context.WithTimeout(t.Context(), 300*time.Millisecond)
	defer readCancel()
	var ev outboundEvent
	if err := wsjson.Read(readCtx, clientConn, &ev); err == nil {
		t.Errorf("received an event despite an already-done event context: %+v", ev)
	}
}

// --- guests (TASKS.md T27) ---------------------------------------------------

// newGuestTestServer wires a full Handler where every user-scoped dependency
// fails the test if it is ever called — that, not any single assertion, is
// what proves a guest turn reaches its own loader and the agent and nothing
// else.
// opts tweak the handler before it starts serving — the guest turn cap is the
// only one so far, set per handler rather than on a package var so these tests
// stay independent of each other.
func newGuestTestServer(t *testing.T, callAgent agentCaller, loadGuestChatCtx func(context.Context, []int) (chatContext, error), opts ...func(*Handler)) *httptest.Server {
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

	forbid := func(name string) {
		t.Errorf("%s was called on a guest socket — guests must never touch a user-scoped table", name)
	}
	h, err := newHandler(jwks.Keyfunc, testIssuer, testAudience, map[string]struct{}{testOrigin: {}},
		func(context.Context, string, string) error { forbid("ensureUser"); return nil },
		func(context.Context, string) (chatContext, error) {
			forbid("loadChatCtx")
			return newChatContext(), nil
		},
		loadGuestChatCtx, callAgent, t.Context(),
		func(context.Context, string) ([]Provider, error) {
			forbid("loadProviders")
			return nil, nil
		},
		noopSaveSubscription,
		func(context.Context, string) ([]Verdict, error) { forbid("loadVerdicts"); return nil, nil },
		noopSaveVerdict,
		noopLoadWatchlistItems, noopCallAgentTitles,
		func(context.Context, string, string) ([]historyTurn, error) {
			forbid("loadConversation")
			return nil, nil
		},
		noopLoadConversationTurns,
		func(context.Context, string, string, string, string, []agentTitleRef) (bool, error) {
			forbid("saveMessages")
			return false, nil
		},
		noopLoadConversationSummaries, noopDeleteConversation,
		testWebhookSecretBytes, noopDeleteUser, noopUpdateUserEmail,
	)
	if err != nil {
		t.Fatal(err)
	}
	// Unlimited by default: only tests about the rate limit itself need to
	// override this, via opts below. Every other guest test would otherwise
	// have to know to opt out of a production default it has nothing to do
	// with, on pain of an unrelated, confusing failure the day it grows one
	// provider-carrying message past the real burst.
	h.guestLimiters = newGuestRateLimiters(t.Context(), rate.Inf, 1000, time.Hour, time.Hour)
	for _, opt := range opts {
		opt(h)
	}
	srv := httptest.NewServer(h.routes())
	t.Cleanup(srv.Close)
	return srv
}

func dialGuest(t *testing.T, srv *httptest.Server) *websocket.Conn {
	t.Helper()
	url := strings.Replace(srv.URL, "http://", "ws://", 1) + "/ws/chat?guest=1"
	conn, _, err := websocket.Dial(t.Context(), url, nil)
	if err != nil {
		t.Fatalf("guest dial: %v", err)
	}
	t.Cleanup(func() { conn.CloseNow() })
	return conn
}

// captureAgentRequests records what the agent was asked, in order, and
// answers every turn with the given events followed by "done".
func captureAgentRequests(reply ...agentEvent) (agentCaller, func() []agentChatRequest) {
	var mu sync.Mutex
	var got []agentChatRequest
	caller := func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		mu.Lock()
		got = append(got, req)
		mu.Unlock()
		ch := make(chan agentEvent, len(reply)+1)
		for _, ev := range reply {
			ch <- ev
		}
		ch <- agentEvent{Type: "done"}
		close(ch)
		return ch, nil
	}
	return caller, func() []agentChatRequest {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(got)
	}
}

func readEvent(t *testing.T, conn *websocket.Conn) outboundEvent {
	t.Helper()
	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}
	return ev
}

// sendAndDrain writes one message frame and reads events until the turn's
// "done", returning them all.
func sendAndDrain(t *testing.T, conn *websocket.Conn, msg inboundMessage) []outboundEvent {
	t.Helper()
	if err := wsjson.Write(t.Context(), conn, msg); err != nil {
		t.Fatal(err)
	}
	var events []outboundEvent
	for {
		ev := readEvent(t, conn)
		events = append(events, ev)
		if ev.Type == "done" || ev.Type == "error" {
			return events
		}
	}
}

func guestMessage(turn, text string, providers []int) inboundMessage {
	return inboundMessage{Type: "message", Turn: turn, Text: text, Conversation: testConversationID, Providers: providers}
}

// oneShotGuestRateLimiter allows exactly one guest message ever (rate 0
// never refills), deterministically — no real wait needed.
func oneShotGuestRateLimiter(t *testing.T) func(*Handler) {
	return func(h *Handler) {
		h.guestLimiters = newGuestRateLimiters(t.Context(), 0, 1, time.Hour, time.Hour)
	}
}

func TestGuestSocketTouchesNoUserTables(t *testing.T) {
	callAgent, requests := captureAgentRequests()
	loadGuest := func(_ context.Context, providers []int) (chatContext, error) {
		return chatContext{Region: guestRegion, Providers: providers, ProviderNames: []string{"Netflix"}}, nil
	}
	srv := newGuestTestServer(t, callAgent, loadGuest)
	conn := dialGuest(t, srv)

	events := sendAndDrain(t, conn, guestMessage("t1", "a heist", []int{8}))
	if last := events[len(events)-1]; last.Type != "done" {
		t.Fatalf("events = %+v, want to end in done", events)
	}

	got := requests()
	if len(got) != 1 {
		t.Fatalf("agent called %d times, want 1", len(got))
	}
	if got[0].WatchRegion != guestRegion {
		t.Errorf("WatchRegion = %q, want %q", got[0].WatchRegion, guestRegion)
	}
	if want := []int{8}; !slices.Equal(got[0].WatchProviders, want) {
		t.Errorf("WatchProviders = %v, want %v", got[0].WatchProviders, want)
	}
	if want := []string{"Netflix"}; !slices.Equal(got[0].WatchProviderNames, want) {
		t.Errorf("WatchProviderNames = %v, want %v", got[0].WatchProviderNames, want)
	}
	if len(got[0].Verdicts) != 0 {
		t.Errorf("Verdicts = %v, want none for a guest", got[0].Verdicts)
	}
}

// A guest's loader fails the turn the same way an account's does — no
// special-case degrade path to keep in sync with runTurn's error handling.
func TestGuestContextLoadFailureFailsTheTurn(t *testing.T) {
	callAgent, requests := captureAgentRequests()
	failing := func(context.Context, []int) (chatContext, error) { return chatContext{}, errUnauthorizedParty }
	srv := newGuestTestServer(t, callAgent, failing)
	conn := dialGuest(t, srv)

	events := sendAndDrain(t, conn, guestMessage("t1", "a heist", []int{8}))
	if len(events) != 1 || events[0].Type != "error" || events[0].Text != genericErrorText {
		t.Errorf("events = %+v, want one generic error", events)
	}
	if n := len(requests()); n != 0 {
		t.Errorf("agent called %d times after a failed context load, want 0", n)
	}
}

// Ticket → account, ?guest=1 → guest, neither → 401. A bad ticket is a
// rejection, never a downgrade; a missing one is never a guest by default.
func TestChatSocketRefusesAnythingButATicketOrTheGuestFlag(t *testing.T) {
	srv := newGuestTestServer(t, noopAgentCaller, noopLoadGuestChatCtx)
	base := strings.Replace(srv.URL, "http://", "ws://", 1) + "/ws/chat"
	for name, query := range map[string]string{"no ticket, no flag": "", "bogus ticket": "?ticket=bogus", "wrong flag value": "?guest=yes"} {
		_, resp, err := websocket.Dial(t.Context(), base+query, nil)
		if err == nil {
			t.Fatalf("%s: connected", name)
		}
		if resp == nil {
			t.Errorf("%s: dial failed with no HTTP response: %v", name, err)
			continue
		}
		if resp.StatusCode != http.StatusUnauthorized {
			t.Errorf("%s: status = %d, want 401", name, resp.StatusCode)
		}
		if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
			t.Errorf("%s: content-type = %q, want application/json", name, ct)
		}
	}
}

func TestMintRefusesAnEmptyUserID(t *testing.T) {
	s := newChatTicketStore()
	if _, err := s.mint(""); !errors.Is(err, errNoUserID) {
		t.Errorf("mint(\"\") error = %v, want errNoUserID", err)
	}
	if len(s.tickets) != 0 {
		t.Error("a ticket was stored for an empty user id")
	}
}

// The guest loader's own logic needs no database: with nothing picked it
// never queries, so a nil pool proves that path stays offline. The shared
// cached-row read is db_test.go's TestCachedProviderNames.
func TestLoadGuestChatContextWithNothingPickedNeverQueries(t *testing.T) {
	cc, err := loadGuestChatContext(nil)(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if cc.Region != guestRegion || len(cc.Providers) != 0 || cc.ProviderNames == nil || len(cc.ProviderNames) != 0 {
		t.Errorf("cc = %+v, want Region=%s, no providers, non-nil empty names", cc, guestRegion)
	}
}

func TestAccountSocketRejectsAProvidersFrame(t *testing.T) {
	var agentCalls atomic.Int32
	callAgent := func(context.Context, agentChatRequest) (<-chan agentEvent, error) {
		agentCalls.Add(1)
		ch := make(chan agentEvent, 1)
		ch <- agentEvent{Type: "done"}
		close(ch)
		return ch, nil
	}
	srv, token := newChatTestServer(t, noopChatCtx, callAgent, noopLoadVerdicts)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	events := sendAndDrain(t, conn, guestMessage("t1", "hi", []int{8}))
	if len(events) != 1 || events[0].Type != "error" || events[0].Turn != "t1" {
		t.Errorf("events = %+v, want one error for a providers frame on an account socket", events)
	}
	if n := agentCalls.Load(); n != 0 {
		t.Errorf("agent called %d times, want 0", n)
	}
}

func TestGuestHistoryIsPerConnection(t *testing.T) {
	// A "message"-type agent event is what makes a turn worth remembering
	// (rec.ok) — "done" alone records nothing, exactly as for an account.
	callAgent, requests := captureAgentRequests(agentEvent{Type: "message", Text: "a reply"})
	srv := newGuestTestServer(t, callAgent, noopLoadGuestChatCtx)

	first := dialGuest(t, srv)
	sendAndDrain(t, first, guestMessage("one", "one", []int{8}))
	sendAndDrain(t, first, guestMessage("two", "two", []int{8}))

	second := dialGuest(t, srv)
	sendAndDrain(t, second, guestMessage("three", "three", []int{8}))

	got := requests()
	if len(got) != 3 {
		t.Fatalf("agent called %d times, want 3", len(got))
	}
	if len(got[0].History) != 0 {
		t.Errorf("first turn history = %+v, want empty", got[0].History)
	}
	want := []historyTurn{{Role: "user", Text: "one"}, {Role: "assistant", Text: "a reply"}}
	if !sameTurns(got[1].History, want) {
		t.Errorf("second turn history = %+v, want %+v", got[1].History, want)
	}
	if len(got[2].History) != 0 {
		t.Errorf("a second socket naming the same conversation saw history %+v — guest history leaked across connections", got[2].History)
	}
}

// One guest socket holds one conversation's history, not a map keyed by the
// client's conversation id — that id is chosen by the browser, so a map has
// no bound on its key count. Switching threads drops the old one, which is
// also what the panel does: it is keyed by conversation id and remounts
// empty, so the model never answers from turns the guest can no longer see.
func TestGuestHistoryFollowsOnlyTheCurrentConversation(t *testing.T) {
	callAgent, requests := captureAgentRequests(agentEvent{Type: "message", Text: "a reply"})
	srv := newGuestTestServer(t, callAgent, noopLoadGuestChatCtx)
	conn := dialGuest(t, srv)

	other := inboundMessage{
		Type: "message", Turn: "two", Text: "two",
		Conversation: "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0", Providers: []int{8},
	}
	sendAndDrain(t, conn, guestMessage("one", "one", []int{8}))
	sendAndDrain(t, conn, other)
	sendAndDrain(t, conn, guestMessage("three", "three", []int{8}))

	got := requests()
	if len(got) != 3 {
		t.Fatalf("agent called %d times, want 3", len(got))
	}
	if len(got[1].History) != 0 {
		t.Errorf("a different conversation saw history %+v, want empty", got[1].History)
	}
	if len(got[2].History) != 0 {
		t.Errorf("returning to the first conversation saw history %+v — a guest's dropped thread came back", got[2].History)
	}
}

func TestGuestTurnCapCountsOnlyTurnsThatReachTheModel(t *testing.T) {
	callAgent, requests := captureAgentRequests()
	srv := newGuestTestServer(t, callAgent, noopLoadGuestChatCtx,
		func(h *Handler) { h.guestTurnCap = 2 })
	conn := dialGuest(t, srv)

	// Nothing picked: the agent's canned nudge, never a turn against the cap
	// — three of these must not consume a single slot.
	for _, turn := range []string{"n1", "n2", "n3"} {
		if events := sendAndDrain(t, conn, guestMessage(turn, "hi", nil)); events[len(events)-1].Type != "done" {
			t.Fatalf("turn %s with no providers: events = %+v, want done", turn, events)
		}
	}
	for _, turn := range []string{"t1", "t2"} {
		if events := sendAndDrain(t, conn, guestMessage(turn, "hi", []int{8})); events[len(events)-1].Type != "done" {
			t.Fatalf("turn %s: events = %+v, want done", turn, events)
		}
	}
	events := sendAndDrain(t, conn, guestMessage("t3", "hi", []int{8}))
	if len(events) != 1 || events[0].Type != "error" || events[0].Turn != "t3" || events[0].Text != guestTurnCapText {
		t.Errorf("capped turn events = %+v, want one error with guestTurnCapText", events)
	}
	if n := len(requests()); n != 5 {
		t.Errorf("agent called %d times, want 5 — the capped turn must never reach it", n)
	}
}

// A rate of 0 never refills, so burst 1 allows exactly one message ever —
// deterministic, no real wait needed.
func TestGuestRateLimitBlocksAfterTheBucketIsSpent(t *testing.T) {
	callAgent, requests := captureAgentRequests()
	srv := newGuestTestServer(t, callAgent, noopLoadGuestChatCtx, oneShotGuestRateLimiter(t))
	conn := dialGuest(t, srv)

	if events := sendAndDrain(t, conn, guestMessage("t1", "hi", []int{8})); events[len(events)-1].Type != "done" {
		t.Fatalf("first message: events = %+v, want done", events)
	}
	events := sendAndDrain(t, conn, guestMessage("t2", "hi", []int{8}))
	if len(events) != 1 || events[0].Type != "error" || events[0].Turn != "t2" || events[0].Text != guestRateLimitText {
		t.Errorf("second message: events = %+v, want one error with guestRateLimitText", events)
	}
	if n := len(requests()); n != 1 {
		t.Errorf("agent called %d times, want 1 — the rate-limited message must never reach it", n)
	}
}

// The bucket is per-IP, not per-socket (TASKS.md T33): a client opening a
// fresh guest socket must not get a fresh budget.
func TestGuestRateLimitIsSharedAcrossSocketsFromTheSameIP(t *testing.T) {
	callAgent, requests := captureAgentRequests()
	srv := newGuestTestServer(t, callAgent, noopLoadGuestChatCtx, oneShotGuestRateLimiter(t))

	first := dialGuest(t, srv)
	if events := sendAndDrain(t, first, guestMessage("t1", "hi", []int{8})); events[len(events)-1].Type != "done" {
		t.Fatalf("first socket: events = %+v, want done", events)
	}

	second := dialGuest(t, srv)
	events := sendAndDrain(t, second, guestMessage("t2", "hi", []int{8}))
	if len(events) != 1 || events[0].Type != "error" || events[0].Text != guestRateLimitText {
		t.Errorf("second socket: events = %+v, want one error with guestRateLimitText — the bucket must be shared per IP, not per socket", events)
	}
	if n := len(requests()); n != 1 {
		t.Errorf("agent called %d times, want 1", n)
	}
}

// Mirrors the turn cap's own carve-out: only a turn that can reach the model
// should ever spend a token.
func TestGuestRateLimitIgnoresMessagesWithNoProvidersPicked(t *testing.T) {
	callAgent, requests := captureAgentRequests()
	srv := newGuestTestServer(t, callAgent, noopLoadGuestChatCtx, oneShotGuestRateLimiter(t))
	conn := dialGuest(t, srv)

	for _, turn := range []string{"n1", "n2", "n3"} {
		if events := sendAndDrain(t, conn, guestMessage(turn, "hi", nil)); events[len(events)-1].Type != "done" {
			t.Fatalf("turn %s with no providers: events = %+v, want done", turn, events)
		}
	}
	if events := sendAndDrain(t, conn, guestMessage("t1", "hi", []int{8})); events[len(events)-1].Type != "done" {
		t.Errorf("first provider-carrying message: events = %+v, want done — the bucket's one token must still be available", events)
	}
	// The no-provider turns above still reach the agent (its own canned
	// nudge) — what they must not do is spend the bucket's one token.
	if n := len(requests()); n != 4 {
		t.Errorf("agent called %d times, want 4", n)
	}
}

// The rate limit and the turn cap must stay distinguishable on the wire —
// neither may masquerade as the other. Regression guard for the check
// order in runChatConnection: the cap is checked first, so it — not the
// rate limit — is what a client sees when both are exhausted at once.
func TestGuestRateLimitAndTurnCapReportDistinctErrors(t *testing.T) {
	callAgent, _ := captureAgentRequests()
	srv := newGuestTestServer(t, callAgent, noopLoadGuestChatCtx,
		oneShotGuestRateLimiter(t),
		func(h *Handler) { h.guestTurnCap = 1 })
	conn := dialGuest(t, srv)

	if events := sendAndDrain(t, conn, guestMessage("t1", "hi", []int{8})); events[len(events)-1].Type != "done" {
		t.Fatalf("first message: events = %+v, want done", events)
	}
	// Both mechanisms are now spent — the cap is checked first, so this
	// must come back as the sign-in nudge, not the rate limit's text.
	events := sendAndDrain(t, conn, guestMessage("t2", "hi", []int{8}))
	if len(events) != 1 || events[0].Text != guestTurnCapText {
		t.Errorf("events = %+v, want guestTurnCapText, not guestRateLimitText", events)
	}
}

// The guest bucket must gate only the `if guest` branch — an authenticated
// socket must never consult it.
func TestSignedInSocketIsNeverRateLimitedByTheGuestBucket(t *testing.T) {
	var agentCalls atomic.Int32
	callAgent := func(context.Context, agentChatRequest) (<-chan agentEvent, error) {
		agentCalls.Add(1)
		ch := make(chan agentEvent, 1)
		ch <- agentEvent{Type: "done"}
		close(ch)
		return ch, nil
	}
	srv, token := newChatTestServer(t, noopChatCtx, callAgent, noopLoadVerdicts, oneShotGuestRateLimiter(t))
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	for _, turn := range []string{"t1", "t2", "t3"} {
		if events := sendAndDrain(t, conn, guestMessage(turn, "hi", nil)); events[len(events)-1].Type != "done" {
			t.Fatalf("turn %s: events = %+v, want done — a spent guest bucket must never affect the signed-in path", turn, events)
		}
	}
	if n := agentCalls.Load(); n != 3 {
		t.Errorf("agent called %d times, want 3", n)
	}
}

// Direct unit test of the janitor's sweep, no WS harness needed — proves the
// map can't grow without bound over a long-running process.
func TestGuestRateLimitersSweepEvictsIdleIPs(t *testing.T) {
	g := newGuestRateLimiters(t.Context(), rate.Every(time.Second), 1, time.Minute, time.Hour)
	g.allow("1.2.3.4")
	if _, ok := g.byIP["1.2.3.4"]; !ok {
		t.Fatal("allow did not create an entry")
	}

	g.sweep(time.Now().Add(30 * time.Second))
	if _, ok := g.byIP["1.2.3.4"]; !ok {
		t.Error("entry evicted before idleTTL elapsed")
	}

	g.sweep(time.Now().Add(2 * time.Minute))
	if _, ok := g.byIP["1.2.3.4"]; ok {
		t.Error("entry not evicted once idle past idleTTL")
	}
}

func TestGuestRejectsAMalformedProvidersList(t *testing.T) {
	callAgent, requests := captureAgentRequests()
	srv := newGuestTestServer(t, callAgent, noopLoadGuestChatCtx)
	conn := dialGuest(t, srv)

	tooMany := make([]int, maxGuestProviders+1)
	for i := range tooMany {
		tooMany[i] = i + 1
	}
	for name, providers := range map[string][]int{
		"non-positive": {8, 0},
		"duplicate":    {8, 15, 8},
		"too many":     tooMany,
		// The ceiling web/src/lib/guest.ts mirrors: a cookie parser that
		// accepted 1e21 as an integer could otherwise put one on the wire.
		"above the id ceiling": {8, maxProviderID + 1},
	} {
		events := sendAndDrain(t, conn, guestMessage(name, "hi", providers))
		if len(events) != 1 || events[0].Type != "error" || events[0].Turn != name {
			t.Errorf("%s: events = %+v, want one error", name, events)
		}
	}
	if n := len(requests()); n != 0 {
		t.Errorf("agent called %d times for malformed frames, want 0", n)
	}
}

// --- what the socket has already shown (T30) --------------------------------

func picksEvent(ids ...int) agentEvent {
	picks := make([]agentPick, len(ids))
	for i, id := range ids {
		picks[i] = agentPick{TMDBID: id, MediaType: "movie", Title: fmt.Sprintf("Film %d", id)}
	}
	return agentEvent{Type: "results", Picks: picks}
}

func newShownTestServer(t *testing.T, reply ...agentEvent) (*httptest.Server, func() []agentChatRequest) {
	t.Helper()
	caller, requests := captureAgentRequests(reply...)
	loadGuest := func(_ context.Context, providers []int) (chatContext, error) {
		return chatContext{Region: guestRegion, Providers: providers, ProviderNames: []string{"Netflix"}}, nil
	}
	return newGuestTestServer(t, caller, loadGuest), requests
}

// The T30 done-when: a second "show me more" must not re-offer what the first
// turn already put on screen.
func TestShownTitlesAreSentBackToTheAgentOnTheNextTurn(t *testing.T) {
	srv, requests := newShownTestServer(t, picksEvent(101, 102))
	conn := dialGuest(t, srv)

	sendAndDrain(t, conn, guestMessage("t1", "a heist", []int{8}))
	sendAndDrain(t, conn, guestMessage("t2", "show me more", []int{8}))

	got := requests()
	if len(got) != 2 {
		t.Fatalf("agent called %d times, want 2", len(got))
	}
	if len(got[0].Shown) != 0 {
		t.Errorf("first turn Shown = %v, want none", got[0].Shown)
	}
	want := []agentTitleRef{{TMDBID: 101, MediaType: "movie"}, {TMDBID: 102, MediaType: "movie"}}
	if !slices.Equal(got[1].Shown, want) {
		t.Errorf("second turn Shown = %+v, want %+v", got[1].Shown, want)
	}
}

// A lookup answers a factual question about a title the user named. The agent
// already refuses to filter that path on the shown set; feeding it is the same
// asymmetry the other way round, and it would make the obvious answer to a
// later recommendation request the one title guaranteed to be missing from it.
//
// Guests, which is where countsAsShown is read. An account's set comes from
// messages.title_refs, which does not record that distinction — see
// shownFromHistory for why that is accepted rather than worked around.
func TestALookupTurnDoesNotFeedTheShownSet(t *testing.T) {
	lookup := picksEvent(603)
	lookup.Kind = "lookup"
	srv, requests := newShownTestServer(t, lookup)
	conn := dialGuest(t, srv)

	sendAndDrain(t, conn, guestMessage("t1", "is Heat on Netflix", []int{8}))
	sendAndDrain(t, conn, guestMessage("t2", "recommend a heist movie", []int{8}))

	got := requests()
	if len(got) != 2 {
		t.Fatalf("agent called %d times, want 2", len(got))
	}
	if len(got[1].Shown) != 0 {
		t.Errorf("Shown = %+v after a lookup turn, want none", got[1].Shown)
	}
}

// captureAgentRequestsPerCall is captureAgentRequests with a different reply
// per call, for the tests that need one turn to differ from the next.
func captureAgentRequestsPerCall(replies ...[]agentEvent) (agentCaller, func() []agentChatRequest) {
	var mu sync.Mutex
	var got []agentChatRequest
	caller := func(_ context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		mu.Lock()
		n := len(got)
		got = append(got, req)
		mu.Unlock()
		var reply []agentEvent
		if n < len(replies) {
			reply = replies[n]
		}
		ch := make(chan agentEvent, len(reply)+1)
		for _, ev := range reply {
			ch <- ev
		}
		ch <- agentEvent{Type: "done"}
		close(ch)
		return ch, nil
	}
	return caller, func() []agentChatRequest {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(got)
	}
}

// A guest's shown set and history are two halves of one memory, and the panel
// drops both when it remounts on a conversation switch. They therefore have to
// reset on the same event: gating the shown reset on the turn having picks
// left a pickless turn in another thread moving one and not the other, so
// going back showed an empty thread that still suppressed its own titles.
func TestAGuestsShownSetAndHistoryForgetTheSameThread(t *testing.T) {
	const convA = "aaaaaaaa-1111-1111-1111-111111111111"
	const convB = "bbbbbbbb-2222-2222-2222-222222222222"

	caller, requests := captureAgentRequestsPerCall(
		[]agentEvent{picksEvent(101)},                              // conv A shows a title
		[]agentEvent{{Type: "message", Text: "I can find films."}}, // conv B, no picks
		nil, // back to conv A
	)
	loadGuest := func(_ context.Context, providers []int) (chatContext, error) {
		return chatContext{Region: guestRegion, Providers: providers, ProviderNames: []string{"Netflix"}}, nil
	}
	conn := dialGuest(t, newGuestTestServer(t, caller, loadGuest))

	send := func(turn, text, conversation string) {
		sendAndDrain(t, conn, inboundMessage{
			Type: "message", Turn: turn, Text: text,
			Conversation: conversation, Providers: []int{8},
		})
	}
	send("t1", "a heist movie", convA)
	send("t2", "what can you do", convB)
	send("t3", "a heist movie", convA)

	got := requests()
	if len(got) != 3 {
		t.Fatalf("agent called %d times, want 3", len(got))
	}
	// The panel shows conv A as an empty thread here, so nothing in it may be
	// suppressed: whatever history forgot, the shown set has to have forgotten.
	if len(got[2].History) == 0 && len(got[2].Shown) > 0 {
		t.Errorf("conv A has no history yet still suppresses %+v", got[2].Shown)
	}
}

// maxShownRefs is sized in titles, so the window has to hold distinct ones.
func TestTheShownWindowCountsTitlesNotSightings(t *testing.T) {
	var history []historyTurn
	// The same title looked up over and over, then genuinely new ones.
	for range 20 {
		history = append(history, historyTurn{Role: "assistant", TitleRefs: []agentTitleRef{
			{TMDBID: 1, MediaType: "movie"},
		}})
	}
	for i := range 10 {
		history = append(history, historyTurn{Role: "assistant", TitleRefs: []agentTitleRef{
			{TMDBID: 100 + i, MediaType: "movie"},
		}})
	}

	refs := windowShown(shownFromHistory(history))
	distinct := make(map[agentTitleRef]struct{}, len(refs))
	for _, r := range refs {
		distinct[r] = struct{}{}
	}
	if len(distinct) != len(refs) {
		t.Errorf("window holds %d entries but only %d titles", len(refs), len(distinct))
	}
	// And the copy that survives is the most recent one, so a title shown a
	// moment ago is not the one the window forgets first.
	if want := (agentTitleRef{TMDBID: 109, MediaType: "movie"}); refs[len(refs)-1] != want {
		t.Errorf("newest ref = %+v, want %+v", refs[len(refs)-1], want)
	}
}

// The window has to stay shorter than how far the agent can page into a query,
// or a "show me more" run past it reports all_shown — "that's everything I
// could find" — about a query that simply had more pages. Guarding the two
// constants against each other because they live in different services and
// nothing else ties them together.
func TestShownWindowStaysShorterThanTheAgentsPagingReach(t *testing.T) {
	// agent/catalog_tool.py's MAX_DISCOVER_PAGES times tmdb.py's PAGE_SIZE.
	const agentPagingReach = 4 * 20
	// agent/catalog_tool.py's RESULT_CEILING: what paging works toward.
	const agentResultCeiling = 10
	if spare := agentPagingReach - maxShownRefs; spare < agentResultCeiling {
		t.Fatalf("maxShownRefs = %d leaves only %d of %d rows unshown; want at least %d",
			maxShownRefs, spare, agentPagingReach, agentResultCeiling)
	}
}

// The agent applies its own backstop to whatever history we send
// (agent/chat.py's MAX_HISTORY_TURNS), and silently: a gateway window wider
// than that backstop is re-truncated with no error anywhere, which is exactly
// how this window was once cut from five exchanges back to three. Only this
// direction needs guarding — the agent's cap sits deliberately at twice the
// operating point, so lowering it is not a realistic edit.
func TestTheAgentDoesNotSilentlyRetruncateOurHistoryWindow(t *testing.T) {
	// agent/chat.py's MAX_HISTORY_TURNS, counted in messages like this one.
	const agentMaxHistoryTurns = 20
	if maxHistoryExchanges*2 > agentMaxHistoryTurns {
		t.Errorf("windowHistory sends %d messages; the agent keeps only %d",
			maxHistoryExchanges*2, agentMaxHistoryTurns)
	}
}

// The load has two consumers with different appetites, and starving either is
// silent: too shallow for windowHistory and the agent loses conversation, too
// shallow for shownFromHistory and "show me more" starts repeating.
func TestSeededMessagesCoversBothItsConsumers(t *testing.T) {
	if maxSeededMessages < maxHistoryExchanges*2 {
		t.Errorf("maxSeededMessages = %d, too shallow for windowHistory's %d",
			maxSeededMessages, maxHistoryExchanges*2)
	}
	if maxSeededMessages > maxStoredHistoryMessages {
		t.Errorf("maxSeededMessages = %d exceeds what is stored (%d)",
			maxSeededMessages, maxStoredHistoryMessages)
	}
	// The other consumer. A budget, not a guarantee — a lookup row carries one
	// ref and a message row none, so a chatty thread refills less than this.
	// It still has to hold at the recommendation-only end, or "show me more"
	// repeats a title the guest path would still be suppressing.
	const agentResultFloor = 5
	if refs := (maxSeededMessages / 2) * agentResultFloor; refs < maxShownRefs {
		t.Errorf("maxSeededMessages = %d budgets only %d refs at the agent's floor of %d, short of maxShownRefs %d",
			maxSeededMessages, refs, agentResultFloor, maxShownRefs)
	}
}

func TestShownFromHistoryFlattensStoredRefsOldestFirst(t *testing.T) {
	history := []historyTurn{
		{Role: "user", Text: "a heist"},
		{Role: "assistant", Text: "Suggested: Heat", TitleRefs: []agentTitleRef{
			{TMDBID: 101, MediaType: "movie"},
		}},
		{Role: "user", Text: "show me more"},
		{Role: "assistant", Text: "Suggested: Ronin", TitleRefs: []agentTitleRef{
			{TMDBID: 102, MediaType: "movie"},
			{TMDBID: 103, MediaType: "movie"},
		}},
	}
	want := []agentTitleRef{
		{TMDBID: 101, MediaType: "movie"},
		{TMDBID: 102, MediaType: "movie"},
		{TMDBID: 103, MediaType: "movie"},
	}
	if got := shownFromHistory(history); !slices.Equal(got, want) {
		t.Errorf("shownFromHistory(...) = %+v, want %+v", got, want)
	}
}

// The T30 gap this closes: an account's shown set used to live only in the
// connection, so a dropped socket — which chat-socket.ts reconnects on next
// use, with the cards still on screen — re-offered exactly what the user was
// looking at. Reading it back from messages.title_refs makes a reconnect cost
// an account nothing here, the same promise loadConversation already makes for
// the history beside it.
func TestAnAccountsShownSetSurvivesAReconnect(t *testing.T) {
	loadConversation := func(_ context.Context, _ string, _ string) ([]historyTurn, error) {
		// What a previous connection persisted, as a fresh socket reads it.
		return []historyTurn{
			{Role: "user", Text: "a heist movie"},
			{Role: "assistant", Text: "Suggested: Heat, Ronin", TitleRefs: []agentTitleRef{
				{TMDBID: 101, MediaType: "movie"},
				{TMDBID: 102, MediaType: "movie"},
			}},
		}, nil
	}
	loadCtx := func(context.Context, string) (chatContext, error) {
		return chatContext{Region: "US"}, nil
	}
	// Through the shared helper, not a hand-rolled closure: the capture
	// happens on the runTurn goroutine and the read on this one, so it needs
	// the mutex captureAgentRequests already has.
	callAgent, requests := captureAgentRequests()
	srv, token := newChatTestServerWithConversations(t, loadCtx, callAgent, noopLoadVerdicts,
		loadConversation, noopSaveMessages)
	conn := dialChat(t, srv, mintTicket(t, srv, token))

	// The very first message on this socket, as a reconnect sends it.
	wsjson.Write(t.Context(), conn, inboundMessage{
		Type: "message", Turn: "t1", Text: "show me more", Conversation: testConversationID,
	})
	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}

	got := requests()
	if len(got) != 1 {
		t.Fatalf("agent called %d times, want 1", len(got))
	}
	want := []agentTitleRef{{TMDBID: 101, MediaType: "movie"}, {TMDBID: 102, MediaType: "movie"}}
	if !slices.Equal(got[0].Shown, want) {
		t.Errorf("Shown on a fresh connection = %+v, want the stored refs %+v", got[0].Shown, want)
	}
}

// title_refs rides along on the history load, and the agent's history window
// is text only — see ChatRequest.history in agent/chat.py.
func TestStoredTitleRefsNeverReachTheAgentsHistory(t *testing.T) {
	body, err := json.Marshal(agentChatRequest{History: []historyTurn{
		{Role: "assistant", Text: "Suggested: Heat", TitleRefs: []agentTitleRef{
			{TMDBID: 101, MediaType: "movie"},
		}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), "101") {
		t.Errorf("history carried title_refs to the agent: %s", body)
	}
}

// Switching threads must not suppress a title in the new one because the old
// one showed it — the same rule guestHistory follows.
func TestShownTitlesDoNotCrossConversations(t *testing.T) {
	srv, requests := newShownTestServer(t, picksEvent(101))
	conn := dialGuest(t, srv)

	sendAndDrain(t, conn, guestMessage("t1", "a heist", []int{8}))
	other := inboundMessage{Type: "message", Turn: "t2", Text: "something else", Conversation: otherConversationID, Providers: []int{8}}
	sendAndDrain(t, conn, other)

	got := requests()
	if len(got) != 2 {
		t.Fatalf("agent called %d times, want 2", len(got))
	}
	if len(got[1].Shown) != 0 {
		t.Errorf("Shown = %+v after a conversation switch, want none", got[1].Shown)
	}
}

// A turn that showed nothing — an error, or a plain message reply — leaves the
// set alone rather than clearing it: the earlier picks are still on screen.
func TestATurnWithNoPicksLeavesTheShownSetIntact(t *testing.T) {
	agent, requests := captureAgentRequestsPerCall(
		[]agentEvent{picksEvent(101)},
		[]agentEvent{{Type: "message", Text: "Nothing matched that."}},
		[]agentEvent{{Type: "message", Text: "Nothing matched that."}},
	)
	loadGuest := func(_ context.Context, providers []int) (chatContext, error) {
		return chatContext{Region: guestRegion, Providers: providers}, nil
	}
	srv := newGuestTestServer(t, agent, loadGuest)
	conn := dialGuest(t, srv)

	sendAndDrain(t, conn, guestMessage("t1", "a heist", []int{8}))
	sendAndDrain(t, conn, guestMessage("t2", "something else", []int{8}))
	sendAndDrain(t, conn, guestMessage("t3", "show me more", []int{8}))

	got := requests()
	if len(got) != 3 {
		t.Fatalf("agent called %d times, want 3", len(got))
	}
	want := []agentTitleRef{{TMDBID: 101, MediaType: "movie"}}
	if !slices.Equal(got[2].Shown, want) {
		t.Errorf("Shown = %+v after a pickless turn, want %+v", got[2].Shown, want)
	}
}

func TestShownSetIsBoundedAcrossManyTurns(t *testing.T) {
	refs := make([]agentTitleRef, maxShownRefs+10)
	for i := range refs {
		refs[i] = agentTitleRef{TMDBID: i, MediaType: "movie"}
	}
	got := windowShown(refs)
	if len(got) != maxShownRefs {
		t.Fatalf("len(windowShown(...)) = %d, want %d", len(got), maxShownRefs)
	}
	if got[len(got)-1].TMDBID != refs[len(refs)-1].TMDBID {
		t.Error("windowShown did not keep the most recently shown titles")
	}
}
