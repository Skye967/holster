package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// --- Ticket-based WS auth ---------------------------------------------------
//
// A browser WebSocket cannot set an Authorization header, so the Bearer-token
// path authMiddleware verifies cannot apply to the upgrade request directly.
// Rather than accept the upgrade unauthenticated and wait on a first "auth"
// frame (a bespoke timeout/state-machine, and one more place a raw token could
// end up in a log if that's ever gotten wrong), the browser first calls
// POST /api/chat/ticket — an ordinary authMiddleware-protected request — and
// gets back a single-use, short-lived ticket to open the socket with. The
// ticket is worthless the instant it's consumed, so it never carries the
// exposure a bearer token in a URL or log line would.
//
// In-memory and single-instance, matching the pool-sizing note in main()'s
// pgxpool setup: if this ever runs more than one gateway replica, minting and
// consuming a ticket must land on the same instance (sticky routing, or move
// the store to something shared).

// A var, not a const, so a test can shorten it without a real 30s wait —
// same reasoning as main.go's provisionTimeout.
var chatTicketTTL = 30 * time.Second

type chatTicket struct {
	userID string
}

type chatTicketStore struct {
	mu      sync.Mutex
	tickets map[string]chatTicket
}

func newChatTicketStore() *chatTicketStore {
	return &chatTicketStore{tickets: make(map[string]chatTicket)}
}

func (s *chatTicketStore) mint(userID string) (string, error) {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	id := base64.RawURLEncoding.EncodeToString(buf)

	s.mu.Lock()
	s.tickets[id] = chatTicket{userID: userID}
	s.mu.Unlock()

	// Self-expiring rather than a swept ticker: tickets are low-volume (one per
	// WS connect) and short-lived, so a timer per ticket is cheap and needs no
	// background loop to shut down on server exit.
	time.AfterFunc(chatTicketTTL, func() {
		s.mu.Lock()
		delete(s.tickets, id)
		s.mu.Unlock()
	})
	return id, nil
}

// consume is single-use: a ticket is deleted the moment it's looked up,
// matched or not, so it cannot be replayed.
func (s *chatTicketStore) consume(id string) (chatTicket, bool) {
	if id == "" {
		return chatTicket{}, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	t, ok := s.tickets[id]
	if ok {
		delete(s.tickets, id)
	}
	return t, ok
}

func (h *Handler) chatTicket(w http.ResponseWriter, r *http.Request) {
	userID, _ := r.Context().Value(ctxUserID).(string)
	id, err := h.tickets.mint(userID)
	if err != nil {
		slog.ErrorContext(r.Context(), "chat ticket mint failed", "error", err.Error())
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "temporarily unavailable"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"ticket": id})
}

// --- WebSocket upgrade and per-connection turn loop -------------------------
//
// One socket per session (../../DECISIONS.md): messages up, curated events
// down, an explicit cancel rather than hanging up and hoping the server
// notices. The gateway owns conversation state for the life of the
// connection; there is no persistence yet (TASKS.md T20), so a reconnect
// starts with empty history — the same as a page reload losing the thread
// until that task lands.

// inboundMessage is what the browser sends up the socket.
type inboundMessage struct {
	Type string `json:"type"`
	Turn string `json:"turn"`
	Text string `json:"text,omitempty"`
}

// outboundEvent is the gateway's curated, public vocabulary — never the
// agent's internal event shape. See chat.go's translation in runTurn and
// agent/chat.py's module docstring on why the two protocols differ.
type outboundEvent struct {
	Type    string      `json:"type"`
	Turn    string      `json:"turn"`
	Text    string      `json:"text,omitempty"`
	Picks   []agentPick `json:"picks,omitempty"`
	Relaxed []string    `json:"relaxed,omitempty"`
}

// historyTurn is both the connection's in-memory record of a completed turn
// and (via the same field names) the wire shape agent/chat.py's
// ChatRequest.history expects.
type historyTurn struct {
	Role string `json:"role"`
	Text string `json:"text"`
}

// turnRecord is how a finished runTurn reports back to the connection's
// coordinator goroutine, which is the sole owner of history and the
// currently-active turn — see runChatConnection.
type turnRecord struct {
	turn          string
	ok            bool // false on error/cancel: nothing worth remembering happened
	userText      string
	assistantText string
}

// Only the last few exchanges reach the agent — see agent/chat.py's
// MAX_HISTORY_TURNS and TASKS.md T14's "decide how much history goes to the
// agent." "show me more"/pagination replaying a stored DiscoverIntent (T13's
// own suggested mechanism) would serve that follow-up case better than raw
// history text, but building it is out of scope here — see agent/README.md.
const maxHistoryExchanges = 2

// The relaxation ladder in agent's search() has no combined deadline of its
// own (TASKS.md T13.5's carried-forward note: "budget for it when wiring this
// into a request path") — this is that budget. Generous relative to the
// documented 4-8s normal case so it only ever fires on a genuinely stuck call.
const turnDeadline = 30 * time.Second

func windowHistory(history []historyTurn) []historyTurn {
	n := maxHistoryExchanges * 2
	if len(history) <= n {
		return history
	}
	return history[len(history)-n:]
}

func appendHistory(history []historyTurn, rec turnRecord) []historyTurn {
	if !rec.ok {
		return history
	}
	return append(history,
		historyTurn{Role: "user", Text: rec.userText},
		historyTurn{Role: "assistant", Text: rec.assistantText},
	)
}

func (h *Handler) chatWS(w http.ResponseWriter, r *http.Request) {
	t, ok := h.tickets.consume(r.URL.Query().Get("ticket"))
	if !ok {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "invalid or expired ticket"})
		return
	}

	conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		OriginPatterns: h.originPatterns,
	})
	if err != nil {
		return // Accept has already written the response.
	}
	defer conn.CloseNow()

	// Not r.Context(): coder/websocket's docs warn that using the request
	// context after Accept returns is unreliable (it's tied to the hijacked
	// connection's lifecycle in ways that vary by server). h.rootCtx is the
	// process's own SIGINT/SIGTERM-cancelled context, so a shutdown cancels
	// every open chat turn the same deliberate way server.Shutdown drains
	// ordinary requests.
	connCtx, cancel := context.WithCancel(h.rootCtx)
	defer cancel()

	h.runChatConnection(connCtx, conn, t.userID)
}

// runChatConnection is the single coordinator for one connection: it alone
// mutates history and tracks the active turn, so neither needs a mutex. A
// dedicated goroutine does the blocking wsjson.Read loop and hands messages
// over a channel; turn goroutines (runTurn) run concurrently and report back
// over `done` — writes to conn are safe from multiple goroutines (coder/
// websocket handles that internally), but only one goroutine may ever call
// Read, which is why the read loop is separate and singular.
func (h *Handler) runChatConnection(ctx context.Context, conn *websocket.Conn, userID string) {
	inbound := make(chan inboundMessage)
	go func() {
		defer close(inbound)
		for {
			var msg inboundMessage
			if err := wsjson.Read(ctx, conn, &msg); err != nil {
				return
			}
			select {
			case inbound <- msg:
			case <-ctx.Done():
				return
			}
		}
	}()

	var history []historyTurn
	var cancelCurrent context.CancelFunc
	var currentTurn string
	done := make(chan turnRecord)

	defer func() {
		if cancelCurrent != nil {
			cancelCurrent()
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return

		case msg, ok := <-inbound:
			if !ok {
				return // socket closed or read failed
			}
			switch msg.Type {
			case "message":
				// A new message supersedes whatever is in flight — belt and
				// suspenders alongside the client disabling send while
				// streaming (TASKS.md T14): the gateway never trusts the
				// client alone to keep two turns from overlapping.
				if cancelCurrent != nil {
					cancelCurrent()
				}
				turnCtx, cancel := context.WithTimeout(ctx, turnDeadline)
				cancelCurrent = cancel
				currentTurn = msg.Turn
				// Copy: history is about to keep changing under the
				// coordinator's feet; the turn goroutine must see a stable
				// snapshot of what existed when it started.
				snapshot := append([]historyTurn(nil), history...)
				go h.runTurn(turnCtx, ctx, conn, userID, msg.Turn, msg.Text, snapshot, done)
			case "cancel":
				if cancelCurrent != nil && currentTurn == msg.Turn {
					cancelCurrent()
				}
			}

		case rec := <-done:
			if rec.turn == currentTurn {
				// Release turnDeadline's timer now rather than letting it
				// idle until it fires on its own — same reason the
				// "message" and "cancel" cases above call this eagerly.
				cancelCurrent()
				cancelCurrent = nil
				currentTurn = ""
			}
			history = appendHistory(history, rec)
		}
	}
}

// sendEvent writes one curated event down the socket, tagged with turn. It
// checks turnCtx first so a turn that has been cancelled, superseded, or has
// timed out never writes a stale event after the fact — see
// runChatConnection for the full set of reasons turnCtx becomes Done. This
// check is both necessary and sufficient; no shared "is this still current"
// state is needed beyond it.
func (h *Handler) sendEvent(turnCtx context.Context, conn *websocket.Conn, turn string, ev outboundEvent) {
	if turnCtx.Err() != nil {
		return
	}
	ev.Turn = turn
	writeCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := wsjson.Write(writeCtx, conn, ev); err != nil {
		slog.WarnContext(turnCtx, "chat event write failed", "turn", turn, "error", err.Error())
	}
}

const (
	genericErrorText     = "Something went wrong — try again."
	agentUnavailableText = "I'm having trouble thinking — try again in a moment."
)

// friendlyError maps agent/chat.py's _error_reason() output to user-facing
// copy — mirrors main.go's authFailure() in spirit: the reason code is
// classified here, never shown to the user raw. Keep this switch's cases in
// sync with _error_reason's fixed set by hand; the default below is the
// safety net if that ever drifts.
func friendlyError(reason string) string {
	switch reason {
	case "tmdb_unavailable":
		return "Can't reach the film database right now."
	case "model_unavailable":
		return agentUnavailableText
	case "internal":
		return genericErrorText
	default:
		slog.Warn("unrecognized agent error reason", "reason", reason)
		return genericErrorText
	}
}

func summarizePicks(picks []agentPick) string {
	if len(picks) == 0 {
		return ""
	}
	titles := make([]string, len(picks))
	for i, p := range picks {
		titles[i] = p.Title
	}
	return "Suggested: " + strings.Join(titles, ", ")
}

// runTurn loads the context the agent needs, calls it, and streams translated
// events down the socket — the concrete version of ARCHITECTURE.md's "Flow:
// asking a question." Reports exactly once to `done`, always, so
// runChatConnection can clear the active-turn state whether the turn
// succeeded, errored, or was cancelled.
//
// Takes two contexts: turnCtx governs this turn's own work (cancelled by
// supersede, explicit cancel, or turnDeadline expiring — see
// runChatConnection) and gates every write via sendEvent; connCtx governs
// only the deferred send's escape hatch below, and must never be used for
// anything else in this function — see that comment for why the
// distinction matters.
func (h *Handler) runTurn(
	turnCtx context.Context,
	connCtx context.Context,
	conn *websocket.Conn,
	userID, turnID, text string,
	history []historyTurn,
	done chan<- turnRecord,
) {
	rec := turnRecord{turn: turnID}
	defer func() {
		// done is unbuffered, and runChatConnection stops reading it once the
		// connection dies — without an escape hatch, this send blocks forever
		// on the ordinary "client closed the tab mid-answer" path, leaking
		// this goroutine permanently. That escape hatch must be connCtx, not
		// turnCtx: turnCtx also becomes Done on an ordinary supersede or
		// turnDeadline expiry, which happens while the coordinator is very
		// much still alive and reading `done` — racing that against turnCtx
		// would let Go's uniform-random select silently drop a legitimately
		// completed turn's record about as often as it delivers it. connCtx
		// only closes when the connection itself is torn down, which is
		// exactly the "nobody will ever read done again" condition this
		// needs.
		select {
		case done <- rec:
		case <-connCtx.Done():
		}
	}()

	chatCtx, err := h.loadChatCtx(turnCtx, userID)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return
		}
		slog.ErrorContext(turnCtx, "chat context load failed", "turn", turnID, "error", dbError(err))
		h.sendEvent(turnCtx, conn, turnID, outboundEvent{Type: "error", Text: genericErrorText})
		return
	}

	// A second round trip rather than a field on chatContext: see that type's
	// comment for why the other three loadChatCtx callers must not pay for
	// this. Sequential after it, matching providers()/watchlist()'s existing
	// two-transaction shape — the turn is about to spend two model calls and
	// several TMDB round trips, so a BEGIN/COMMIT is not what makes it slow,
	// and running loadChatCtx first means a cancelled turn bails before this.
	//
	// A failure here fails the turn rather than degrading to no verdicts.
	// TASKS.md is explicit that "the chat cannot degrade," and degrading
	// would silently re-show titles the user marked seen — breaking exactly
	// the guarantee T19 exists to make, in the direction users notice.
	// Skipped for a caller with no subscriptions: the agent answers that with
	// its "pick your services" message before it ever looks at verdicts
	// (agent/chat.py's stream_chat), so loading them is work whose result is
	// unused - and a title_verdicts problem must not be the thing that stops a
	// brand-new user, who has no verdicts anyway, from being told to pick some.
	var verdicts []Verdict
	if len(chatCtx.Providers) > 0 {
		verdicts, err = h.loadVerdicts(turnCtx, userID)
		if err != nil {
			if errors.Is(err, context.Canceled) {
				return
			}
			slog.ErrorContext(turnCtx, "chat verdict load failed", "turn", turnID, "error", dbError(err))
			h.sendEvent(turnCtx, conn, turnID, outboundEvent{Type: "error", Text: genericErrorText})
			return
		}
	}

	events, err := h.callAgent(turnCtx, agentChatRequest{
		Message:            text,
		WatchRegion:        chatCtx.Region,
		WatchProviders:     chatCtx.Providers,
		WatchProviderNames: chatCtx.ProviderNames,
		History:            windowHistory(history),
		Verdicts:           verdicts,
	})
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return
		}
		slog.ErrorContext(turnCtx, "agent call failed", "turn", turnID, "error", err.Error())
		h.sendEvent(turnCtx, conn, turnID, outboundEvent{Type: "error", Text: agentUnavailableText})
		return
	}

	for ev := range events {
		if turnCtx.Err() != nil {
			return
		}
		switch ev.Type {
		case "intent":
			var intent agentIntent
			if err := json.Unmarshal(ev.Intent, &intent); err == nil {
				line := interpretingLine(intent, chatCtx.ProviderNames)
				h.sendEvent(turnCtx, conn, turnID, outboundEvent{Type: "interpreting", Text: line})
			} else {
				slog.WarnContext(turnCtx, "chat intent decode failed", "turn", turnID, "error", err.Error())
			}
		case "results":
			h.sendEvent(turnCtx, conn, turnID, outboundEvent{
				Type: "results", Picks: ev.Picks, Relaxed: ev.Relaxed,
			})
			// Guarded on a non-empty summary, not unconditional like
			// "message" below: zero picks is a real, reachable state
			// (summarizePicks returns "" for it) that must not overwrite an
			// ok=true left by an earlier event in this same stream. An empty
			// ev.Text on "message" isn't reachable the same way — every
			// message-type event agent/chat.py emits comes from a non-empty
			// template — so no matching guard is needed there.
			if summary := summarizePicks(ev.Picks); summary != "" {
				rec.ok = true
				rec.userText = text
				rec.assistantText = summary
			}
		case "message":
			h.sendEvent(turnCtx, conn, turnID, outboundEvent{Type: "token", Text: ev.Text})
			rec.ok = true
			rec.userText = text
			rec.assistantText = ev.Text
		case "error":
			h.sendEvent(turnCtx, conn, turnID, outboundEvent{Type: "error", Text: friendlyError(ev.Reason)})
			rec.ok = false
			return
		case "done":
			h.sendEvent(turnCtx, conn, turnID, outboundEvent{Type: "done"})
			return
		default:
			slog.WarnContext(turnCtx, "unknown agent event type", "turn", turnID, "type", ev.Type)
		}
	}

	// Reached only if the channel closed without "done" or "error" ever
	// firing (both return above) — a transport failure (see newAgentCaller's
	// scanner.Err() log) or an agent-side bug that dropped the stream mid-way.
	// TASKS.md: "A dropped stream keeps what arrived, marks it incomplete,
	// and offers a retry — it never just stops mid-sentence looking
	// finished." rec.ok may already be true from an earlier "results"/
	// "message" event in this same stream — that's left as-is, not reset:
	// what the user already saw stays what the user saw (DECISIONS.md:
	// "stored history must match what the user saw"), this fallback only
	// adds the missing "and it didn't finish cleanly" signal on top.
	h.sendEvent(turnCtx, conn, turnID, outboundEvent{Type: "error", Text: genericErrorText})
}

// interpretingLine is templated from the agent's DiscoverIntent, never a
// second model call (TASKS.md T14) — it doubles as a comprehension check, so
// it must be on screen well before the full pipeline finishes.
func interpretingLine(intent agentIntent, providerNames []string) string {
	kind := "movies"
	if intent.MediaType == "tv" {
		kind = "TV shows"
	}
	subject := kind
	if len(intent.Genres) > 0 {
		subject = strings.Join(intent.Genres, "/") + " " + kind
	}

	var clauses []string
	switch {
	case intent.ReleaseYearGte != nil && intent.ReleaseYearLte != nil:
		clauses = append(clauses, fmt.Sprintf("from %d-%d", *intent.ReleaseYearGte, *intent.ReleaseYearLte))
	case intent.ReleaseYearGte != nil:
		clauses = append(clauses, fmt.Sprintf("from %d on", *intent.ReleaseYearGte))
	case intent.ReleaseYearLte != nil:
		clauses = append(clauses, fmt.Sprintf("before %d", *intent.ReleaseYearLte))
	}
	if intent.MaxRuntimeMinutes != nil {
		clauses = append(clauses, fmt.Sprintf("under %d minutes", *intent.MaxRuntimeMinutes))
	}
	if len(intent.Keywords) > 0 {
		clauses = append(clauses, "about "+strings.Join(intent.Keywords, ", "))
	}
	if excl := slices.Concat(intent.WithoutGenres, intent.WithoutKeywords); len(excl) > 0 {
		clauses = append(clauses, "excluding "+strings.Join(excl, ", "))
	}
	if len(intent.Cast) > 0 {
		clauses = append(clauses, "starring "+humanJoin(intent.Cast))
	}
	if len(intent.Crew) > 0 {
		clauses = append(clauses, "from "+humanJoin(intent.Crew))
	}

	line := "Looking for " + subject
	if len(clauses) > 0 {
		line += " " + strings.Join(clauses, ", ")
	}
	// Requires streaming_providers to have a cached row for the caller's
	// country (T9's table; T15 owns keeping it fresh). Until then, or if the
	// user's picks aren't in it yet, providerNames is empty and this clause is
	// dropped — degrade the phrasing, never block on it.
	if len(providerNames) > 0 {
		line += " on " + humanJoin(providerNames)
	}
	return line
}

func humanJoin(items []string) string {
	switch len(items) {
	case 0:
		return ""
	case 1:
		return items[0]
	default:
		return strings.Join(items[:len(items)-1], ", ") + " and " + items[len(items)-1]
	}
}

// --- Loading the agent's context --------------------------------------------

// chatContext is the subscriptions-and-country half of what ARCHITECTURE.md's
// "Flow: asking a question" calls loading "subscriptions, country and every
// verdict." The verdicts half is deliberately not here: three of
// loadChatCtx's four callers (providers.go's two handlers and watchlist.go)
// have no verdict dependency, so loading them here would spend a query those
// three discard and, worse, would 503 them on a title_verdicts problem they
// have nothing to do with. runTurn assembles the third piece itself, from
// verdicts.go's loadVerdicts — see there and T19.
type chatContext struct {
	Region        string
	Providers     []int
	ProviderNames []string // the caller's own Providers, resolved to names where cached
}

// newChatContext returns a chatContext with both slice fields non-nil, for any
// construction site that needs the "[] not null" guarantee providers.go's
// subscriptions handler (and any future direct JSON consumer) depends on — not
// every chatContext{} literal in this package uses it; a few test fixtures set
// only the fields they read and skip it safely.
func newChatContext() chatContext {
	return chatContext{Providers: []int{}, ProviderNames: []string{}}
}

// A minimal, independent decode of streaming_providers.providers — see
// providers.go's Provider, which decodes the same jsonb column for the
// /api/providers picker response. If a field name here changes, check there
// too.
type cachedProvider struct {
	ProviderID   int    `json:"provider_id"`
	ProviderName string `json:"provider_name"`
}

func parseProviderNames(raw []byte, wanted []int) []string {
	var all []cachedProvider
	if err := json.Unmarshal(raw, &all); err != nil {
		return []string{}
	}
	byID := make(map[int]string, len(all))
	for _, p := range all {
		byID[p.ProviderID] = p.ProviderName
	}
	names := make([]string, 0, len(wanted))
	for _, id := range wanted {
		if name, ok := byID[id]; ok {
			names = append(names, name)
		}
	}
	return names
}

// loadChatContext mirrors upsertUser's shape in main.go: a plain function
// closed over the pool, run inside withUser so streaming_subscriptions' and
// users' RLS policies apply. streaming_providers has no per-user policy (it's
// a country-keyed cache, not user data), so reading it inside the same
// transaction is just convenient, not a privilege requirement.
func loadChatContext(db *pgxpool.Pool) func(ctx context.Context, userID string) (chatContext, error) {
	return func(ctx context.Context, userID string) (chatContext, error) {
		cc := newChatContext()
		err := withUser(ctx, db, userID, func(tx pgx.Tx) error {
			if err := tx.QueryRow(ctx, `select country from users where id = $1`, userID).
				Scan(&cc.Region); err != nil {
				return err
			}

			rows, err := tx.Query(ctx,
				`select tmdb_provider_id from streaming_subscriptions where user_id = $1`, userID)
			if err != nil {
				return err
			}
			defer rows.Close()
			for rows.Next() {
				var id int
				if err := rows.Scan(&id); err != nil {
					return err
				}
				cc.Providers = append(cc.Providers, id)
			}
			if err := rows.Err(); err != nil {
				return err
			}

			var providersJSON []byte
			err = tx.QueryRow(ctx,
				`select providers from streaming_providers where country = $1`, cc.Region).
				Scan(&providersJSON)
			switch {
			case errors.Is(err, pgx.ErrNoRows):
				// No cached row for this country yet (T15 populates it lazily
				// on read) — degrade the interpreting line, don't fail the turn.
			case err != nil:
				return err
			default:
				cc.ProviderNames = parseProviderNames(providersJSON, cc.Providers)
			}
			return nil
		})
		return cc, err
	}
}

// --- Calling the agent -------------------------------------------------------

// agentChatRequest is agent/chat.py's ChatRequest — field names must match
// its JSON aliases exactly. omitempty on the slices matters: Pydantic's
// list fields default via default_factory=list and reject an explicit null,
// so an empty slice must be an absent key, not `null`.
type agentChatRequest struct {
	Message        string `json:"message"`
	WatchRegion    string `json:"watch_region"`
	WatchProviders []int  `json:"watch_providers,omitempty"`
	// Same values as chatContext.ProviderNames, already resolved for
	// interpretingLine below — reused here so a capability-question answer
	// (TASKS.md T16.5) can name the caller's services without the agent
	// needing its own TMDB lookup.
	WatchProviderNames []string      `json:"watch_provider_names,omitempty"`
	History            []historyTurn `json:"history,omitempty"`
	// The caller's whole verdict set (T19); agent/catalog_tool.py's search()
	// decides what each value means.
	Verdicts []Verdict `json:"verdicts,omitempty"`
}

// agentIntent mirrors DiscoverIntent.model_dump(exclude_none=True) — see
// agent/catalog_tool.py. Pointers on the fields Pydantic can leave unset.
//
// Deliberately omits sort_by and limit: interpretingLine() is a comprehension
// check on *what* the user asked for, not *how* the search runs — the user
// already knows how many results they asked for and doesn't need it read
// back. Every other DiscoverIntent field is here; an omitted field silently
// vanishes on decode (json.Unmarshal drops unknown keys), so if you add a
// field to DiscoverIntent that should show up in the interpreting line, it
// must be added here too — nothing else catches the gap.
type agentIntent struct {
	MediaType         string   `json:"media_type"`
	Genres            []string `json:"genres"`
	WithoutGenres     []string `json:"without_genres"`
	Keywords          []string `json:"keywords"`
	WithoutKeywords   []string `json:"without_keywords"`
	Cast              []string `json:"cast"`
	Crew              []string `json:"crew"`
	MaxRuntimeMinutes *int     `json:"max_runtime_minutes"`
	ReleaseYearGte    *int     `json:"release_year_gte"`
	ReleaseYearLte    *int     `json:"release_year_lte"`
}

// agentPick mirrors one entry of chat.py's "results" event: a tmdb.Title
// merged with per-title runtime/cast (tmdb.TitleDetails), resolved genre
// names, availability filtered to the caller's own subscriptions, and
// rank()'s blurb. Forwarded to the browser close to verbatim — the
// title-card rendering is TASKS.md T16.5's job, not this one's.
type agentPick struct {
	TMDBID         int      `json:"tmdb_id"`
	MediaType      string   `json:"media_type"`
	Title          string   `json:"title"`
	Year           *int     `json:"year"`
	Overview       string   `json:"overview"`
	PosterURL      *string  `json:"poster_url"`
	VoteAverage    float64  `json:"vote_average"`
	VoteCount      int      `json:"vote_count"`
	GenreIDs       []int    `json:"genre_ids"`
	GenreNames     []string `json:"genre_names"`
	RuntimeMinutes *int     `json:"runtime_minutes"`
	Cast           []string `json:"cast"`
	// nil (JSON null) when the agent's availability check itself failed —
	// distinct from a non-nil empty slice, which means TMDB confirms this
	// title streams on none of the caller's services. No `omitempty`: both
	// states must reach the browser distinguishably, and Go's JSON codec
	// already round-trips null<->nil and []<->non-nil-empty correctly for a
	// plain slice, so no extra type (e.g. a pointer) is needed here.
	AvailableOn []agentProvider `json:"available_on"`
	Blurb       string          `json:"blurb"`
	// True only for a watchlist row (TASKS.md T18.5) whose TMDB lookup
	// failed or the id no longer resolves — every other field on such a
	// row is then a Go zero value, not real data. Absent (false) on every
	// /chat pick.
	Unavailable bool `json:"unavailable,omitempty"`
}

// agentProvider mirrors one entry of tmdb.py's Provider TypedDict, as used in
// agentPick.AvailableOn — a minimal, independent decode, same pattern this
// file already uses twice for overlapping TMDB provider shapes (providers.go's
// Provider and this file's own cachedProvider each note the other in their
// doc comments rather than sharing a type). Not providers.go's Provider: that
// one also carries DisplayPriority with no `omitempty`, which has no meaning
// for one title's availability and would leak a bogus "display_priority": 0
// into every entry sent to the browser. Not cachedProvider either: that one
// has no LogoURL, and a card plausibly wants to show one.
type agentProvider struct {
	ProviderID   int     `json:"provider_id"`
	ProviderName string  `json:"provider_name"`
	LogoURL      *string `json:"logo_url"`
}

// agentEvent is one NDJSON line from POST /chat on the agent — see
// agent/chat.py's module docstring for the full shape of each type.
type agentEvent struct {
	Type    string          `json:"type"`
	Intent  json.RawMessage `json:"intent,omitempty"`
	Picks   []agentPick     `json:"picks,omitempty"`
	Relaxed []string        `json:"relaxed,omitempty"`
	Text    string          `json:"text,omitempty"`
	Reason  string          `json:"reason,omitempty"`
}

// agentCaller is the injection seam for the agent HTTP call — same shape as
// ensureUser in main.go: production wiring is newAgentCaller, tests fake it
// directly rather than standing up a real agent process.
type agentCaller func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error)

// newAgentCaller streams agent/main.py's POST /chat response (chunked NDJSON,
// not a second WebSocket — see ../../DECISIONS.md) and parses it one line at
// a time onto a channel. Cancelling ctx aborts the underlying HTTP request,
// which is the whole of this codebase's cancel story for the agent call — see
// chat.go's runTurn and DECISIONS.md's "One socket per session."
func newAgentCaller(client *http.Client, baseURL string) agentCaller {
	return func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error) {
		body, err := json.Marshal(req)
		if err != nil {
			return nil, err
		}
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/chat", bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(httpReq)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			return nil, fmt.Errorf("agent: unexpected status %d", resp.StatusCode)
		}

		events := make(chan agentEvent)
		go func() {
			defer close(events)
			defer resp.Body.Close()
			scanner := bufio.NewScanner(resp.Body)
			// Titles carry overviews and posters for up to 20 candidates
			// (TASKS.md T13's limit) — comfortably past the 64KiB default,
			// nowhere near unbounded.
			scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)
			for scanner.Scan() {
				var ev agentEvent
				if err := json.Unmarshal(scanner.Bytes(), &ev); err != nil {
					// A malformed line from the agent is its bug, not a reason
					// to abort a turn that may otherwise be fine — drop it.
					continue
				}
				select {
				case events <- ev:
				case <-ctx.Done():
					return
				}
			}
			// Distinguishes an abnormal body close (agent crash, network
			// drop, an oversized line) from a clean EOF — nil here either
			// way. This is safe to *only* log: runTurn's event loop returns
			// early on "done"/"error" and falls through to an unconditional
			// fallback otherwise, so whichever caused the drop, the browser
			// still gets told the turn is incomplete rather than being left
			// hanging (see runTurn and TASKS.md's dropped-stream rule).
			if err := scanner.Err(); err != nil {
				slog.WarnContext(ctx, "agent stream ended abnormally", "error", err.Error())
			}
		}()
		return events, nil
	}
}
