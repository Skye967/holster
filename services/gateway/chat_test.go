package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"encoding/json"
	"errors"
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
)

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

func TestWindowHistoryKeepsOnlyTheLastFewExchanges(t *testing.T) {
	var history []historyTurn
	for i := range 5 {
		history = append(history,
			historyTurn{Role: "user", Text: strings.Repeat("u", i)},
			historyTurn{Role: "assistant", Text: strings.Repeat("a", i)},
		)
	}
	got := windowHistory(history)
	if len(got) != maxHistoryExchanges*2 {
		t.Fatalf("len(windowHistory(...)) = %d, want %d", len(got), maxHistoryExchanges*2)
	}
	if got[0] != history[len(history)-maxHistoryExchanges*2] {
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

func TestAppendHistorySkipsFailedTurns(t *testing.T) {
	history := appendHistory(nil, turnRecord{turn: "t1", ok: false, userText: "x"})
	if len(history) != 0 {
		t.Errorf("a failed/cancelled turn was recorded in history: %+v", history)
	}
}

func TestAppendHistoryRecordsSuccessfulTurns(t *testing.T) {
	history := appendHistory(nil, turnRecord{
		turn: "t1", ok: true, userText: "hi", assistantText: "hello",
	})
	if len(history) != 2 || history[0].Role != "user" || history[1].Role != "assistant" {
		t.Errorf("history = %+v, want [user, assistant]", history)
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
func newChatTestServer(t *testing.T, loadCtx func(context.Context, string) (chatContext, error), callAgent agentCaller, loadVerdicts func(context.Context, string) ([]Verdict, error)) (*httptest.Server, string) {
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
		loadCtx, callAgent, t.Context(),
		noopLoadProviders, noopSaveSubscription,
		loadVerdicts, noopSaveVerdict,
		noopLoadWatchlistItems, noopCallAgentTitles,
	)
	if err != nil {
		t.Fatal(err)
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

func TestChatTicketIsRequiredToOpenTheSocket(t *testing.T) {
	srv, _ := newChatTestServer(t, noopChatCtx, noopAgentCaller, noopLoadVerdicts)
	url := strings.Replace(srv.URL, "http://", "ws://", 1) + "/ws/chat?ticket=bogus"
	_, resp, err := websocket.Dial(t.Context(), url, nil)
	if err == nil {
		t.Fatal("connected with an invalid ticket")
	}
	if resp == nil || resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %+v, want 401", resp)
	}
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

	if err := wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "a heist movie"}); err != nil {
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

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "what can you do?"})
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

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "anything"})

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

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "anything"})
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

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "first"})
	<-firstBlocked
	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t2", Text: "second"})

	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.Type != "done" || ev.Turn != "t2" {
		t.Errorf("event = %+v, want done for t2", ev)
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
		h.runTurn(turnCtx, connCtx, nil, "user_1", "t1", "hi", nil, done)
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

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "anything"})

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

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "anything"})

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

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "anything"})

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

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "anything"})

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

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "something good"})
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

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "something good"})

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

	wsjson.Write(t.Context(), conn, inboundMessage{Type: "message", Turn: "t1", Text: "hi"})

	var ev outboundEvent
	if err := wsjson.Read(t.Context(), conn, &ev); err != nil {
		t.Fatal(err)
	}
	if ev.Type != "token" || ev.Text != "Pick your services." {
		t.Errorf("event = %+v, want the agent's message forwarded", ev)
	}
}
