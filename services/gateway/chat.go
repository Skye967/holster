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
	"net"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"golang.org/x/time/rate"
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

type chatTicketEntry struct {
	ticket   chatTicket
	expireAt time.Time
}

type chatTicketStore struct {
	mu      sync.Mutex
	tickets map[string]chatTicketEntry
}

func newChatTicketStore() *chatTicketStore {
	return &chatTicketStore{tickets: make(map[string]chatTicketEntry)}
}

func (s *chatTicketStore) mint(userID string) (string, error) {
	// "" is the guest sentinel inside a connection (see "Guests" below); a
	// ticket must never be able to carry it.
	if userID == "" {
		return "", errNoUserID
	}
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	id := base64.RawURLEncoding.EncodeToString(buf)
	s.mu.Lock()
	s.tickets[id] = chatTicketEntry{ticket: chatTicket{userID: userID}, expireAt: time.Now().Add(chatTicketTTL)}
	s.mu.Unlock()
	s.scheduleExpiry(id)
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
	entry, ok := s.tickets[id]
	if ok {
		delete(s.tickets, id)
	}
	return entry.ticket, ok
}

// scheduleExpiry removes id after chatTicketTTL. Ticket ids are fresh 32-byte
// crypto/rand values and never reused, so there is no second write to the same
// key to defend against: an unconditional delete is correct.
func (s *chatTicketStore) scheduleExpiry(id string) {
	time.AfterFunc(chatTicketTTL, func() {
		s.mu.Lock()
		delete(s.tickets, id)
		s.mu.Unlock()
	})
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

// --- Guests ------------------------------------------------------------------
//
// Why guests exist and what they cost: ARCHITECTURE.md's "Sign-in is optional".
//
// What this file enforces: a guest has no users row, so nothing user-scoped is
// read or written for it; ?guest=1 declares a guest and a ticket declares an
// account, while neither is a 401 — an absent credential never silently
// becomes a different mode; within a connection userID == "" is the guest, and
// mint above refuses to ever issue that value.
//
// A per-IP token bucket gates guest "message" frames (guestRateLimiters
// below), so one client can't be the exhaustion vector. It does not divide the
// quota: several concurrent guests still contend for the same Gemini free
// tier, and a model 429 reaches the user as "try again in a moment"
// (friendlyError) either way.
//
// The bucket gates a frame on an already-open socket, never the upgrade
// itself: a browser can't read a rejected upgrade's status, so nothing here
// refuses a connection.

// Every account is provisioned into users.country's default
// (20260831230634_streaming.sql), so a guest hard-wired to the same value sees
// the same catalog an account does today. Revisit alongside any real country
// setting.
const guestRegion = "US"

// Bounds one message's providers list — the picker offers a few dozen per
// region, so anything past this is a malformed frame, not a real selection.
// Mirrors web/src/lib/guest.ts's MAX_GUEST_PROVIDERS, which stops a longer
// list reaching the wire in the first place.
const maxGuestProviders = 50

// Mirrors web/src/lib/guest.ts's MAX_PROVIDER_ID. TMDB's ids are four digits;
// the ceiling only has to be low enough to keep a nonsense id out of a frame.
const maxProviderID = 1_000_000

// Turns one guest socket may send to the model before it's nudged to sign in.
// Per-socket, and a reconnect starts a fresh one — a nudge in the funnel, not a
// quota control; nothing here bounds the shared LLM spend (see the note above).
const defaultGuestTurnCap = 20

// No number in the copy, so it can't drift from the cap.
const guestTurnCapText = "You've reached the guest limit for this session — sign in to keep chatting."

// guestRateLimiters is a per-IP token bucket shared by every guest socket
// from that IP — deliberately not per-socket, since opening a
// fresh socket must not hand a client a fresh budget. In-memory and
// single-instance, same limitation as chatTicketStore.
//
// IP extraction (see guestIP below) reads r.RemoteAddr, not
// X-Forwarded-For: nothing in this repo documents a reverse proxy in front
// of the gateway (docker-compose.yml exposes it directly), and trusting a
// client-supplied header with no verified hop in front of it would let any
// client spoof its way past the limiter entirely. If a trusted proxy is
// ever placed in front of the gateway, this must switch to reading that
// proxy's header, or every guest collapses into one shared bucket.
type guestRateLimiters struct {
	mu      sync.Mutex
	limit   rate.Limit
	burst   int
	idleTTL time.Duration
	byIP    map[string]*guestRateLimiterEntry
}

type guestRateLimiterEntry struct {
	limiter  *rate.Limiter
	lastSeen time.Time
}

// Sized against the shared LLM quota, not one guest's own budget: Gemini's
// free tier allows 15 requests a minute and a turn costs two calls, so the
// whole app has roughly 7 turns a minute, shared by every guest and
// account. One IP steady-refilling at 2/min leaves headroom for several
// concurrently active guests before the app-wide ceiling is reached, while a
// burst of 3 still covers an opening message plus a couple of quick
// follow-ups ("show me more") before it throttles.
var defaultGuestRateLimit = rate.Every(30 * time.Second)

const defaultGuestRateBurst = 3

// Idle IPs are swept so the map can't grow without bound over the process's
// lifetime: distinct guest IPs arrive far faster than the deletes that grow
// conversationDeletions, and a forgotten limiter costs a guest one free burst
// rather than correctness.
const defaultGuestLimiterIdleTTL = 30 * time.Minute
const defaultGuestLimiterSweepInterval = 10 * time.Minute

// No number in the copy, matching guestTurnCapText.
const guestRateLimitText = "You're sending messages faster than I can keep up — wait a moment and try again."

func newGuestRateLimiters(rootCtx context.Context, limit rate.Limit, burst int, idleTTL, sweepInterval time.Duration) *guestRateLimiters {
	g := &guestRateLimiters{limit: limit, burst: burst, idleTTL: idleTTL, byIP: make(map[string]*guestRateLimiterEntry)}
	go g.janitor(rootCtx, sweepInterval)
	return g
}

// allow reports whether ip may send now, lazily creating its bucket on first
// sight. Only the map lookup/creation needs the mutex — *rate.Limiter is
// already safe for concurrent use, so AllowN() runs outside the lock rather
// than serializing every guest IP's check behind one. AllowN(now, 1) rather
// than Allow(), so lastSeen and the token check read the same instant
// instead of two separate calls to time.Now().
func (g *guestRateLimiters) allow(ip string) bool {
	now := time.Now()
	g.mu.Lock()
	entry, ok := g.byIP[ip]
	if !ok {
		entry = &guestRateLimiterEntry{limiter: rate.NewLimiter(g.limit, g.burst)}
		g.byIP[ip] = entry
	}
	entry.lastSeen = now
	limiter := entry.limiter
	g.mu.Unlock()
	return limiter.AllowN(now, 1)
}

// sweep evicts entries idle past idleTTL, as of now — a parameter rather
// than time.Now() so a test can drive it without a real wait.
func (g *guestRateLimiters) sweep(now time.Time) {
	g.mu.Lock()
	defer g.mu.Unlock()
	for ip, entry := range g.byIP {
		if now.Sub(entry.lastSeen) > g.idleTTL {
			delete(g.byIP, ip)
		}
	}
}

func (g *guestRateLimiters) janitor(rootCtx context.Context, sweepInterval time.Duration) {
	ticker := time.NewTicker(sweepInterval)
	defer ticker.Stop()
	for {
		select {
		case <-rootCtx.Done():
			return
		case now := <-ticker.C:
			g.sweep(now)
		}
	}
}

// guestIP extracts the caller's address for guestRateLimiters. See that
// type's doc comment for why this trusts RemoteAddr and not a header.
func guestIP(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		// Not expected on a direct connection (see the doc comment above) —
		// logged because the fallback silently collapses every guest hitting
		// it into one shared bucket, which would otherwise be invisible.
		slog.WarnContext(r.Context(), "guest IP parse failed", "remoteAddr", r.RemoteAddr, "error", err.Error())
		return r.RemoteAddr
	}
	return host
}

// validGuestProviders rejects a frame whose providers list can't have come
// from the picker — a set of switches, so no duplicates, and the same
// positive-id posture as setSubscription's providerID check.
func validGuestProviders(ids []int) bool {
	if len(ids) > maxGuestProviders {
		return false
	}
	seen := make(map[int]bool, len(ids))
	for _, id := range ids {
		if id <= 0 || id > maxProviderID || seen[id] {
			return false
		}
		seen[id] = true
	}
	return true
}

// loadGuestChatContext is loadChatContext's guest counterpart, injected the
// same way: region fixed, providers from the message, names from the same
// cached row — never loadProviders, whose stale-cache refresh path belongs to
// the picker's page load, not inside a turn's deadline.
func loadGuestChatContext(db *pgxpool.Pool) func(ctx context.Context, providers []int) (chatContext, error) {
	return func(ctx context.Context, providers []int) (chatContext, error) {
		cc := chatContext{Region: guestRegion, Providers: providers, ProviderNames: []string{}}
		// Nothing to resolve, and the agent answers this case from a template
		// without ever reading names — no reason to touch the database.
		if len(providers) == 0 {
			return cc, nil
		}
		names, err := cachedProviderNames(ctx, db, guestRegion, providers)
		if err != nil {
			return cc, err
		}
		cc.ProviderNames = names
		return cc, nil
	}
}

// --- Tracking conversations deleted mid-turn --------------------------------
//
// Deleting a conversation is a plain HTTP request, racing whatever WS turn is
// still in flight on that id. Without this, a straggling finishTurn's persist
// lands after the delete and recreates the row through saveMessages'
// on-conflict-do-nothing insert, which can't tell "exists because it's
// ongoing" from "exists because it was just deleted."
//
// A monotonic per-conversation generation makes that distinction: dispatch
// captures the current generation (runChatConnection) and finishTurn
// suppresses its persist only if the generation moved on since. A boolean
// tombstone can't — it reads a legitimate new turn dispatched after the
// delete (the back button reopening a deleted conversation's URL) the same as
// a straggler.
//
// Never expired. An aged-out generation reverts to 0, "never deleted", and
// nothing anywhere notices — finishTurn stops suppressing and a deleted
// conversation reappears. A straggling turn is bounded (turnDeadline plus the
// persist, ~32s), so finishTurn alone would tolerate a generous TTL; the risk
// is anything longer-lived that reads generation(), for which no TTL is
// demonstrably long enough. Permanence is cheap: one int per conversation ever
// deleted, a low-volume, user-driven event.
//
// In-memory, single-instance — same limitation as chatTicketStore: a delete
// landing on one replica while a straggler lives on another misses this check.
type conversationDeletions struct {
	mu   sync.Mutex
	gens map[string]int
}

func newConversationDeletions() *conversationDeletions {
	return &conversationDeletions{gens: make(map[string]int)}
}

// bump records a real delete of id, advancing its generation. Only ever
// called after a delete that actually removed a row (deleteConversationHandler
// checks RowsAffected first) — an idempotent no-op delete, including one
// attempted against another user's conversation id (rejected by RLS/the
// WHERE clause, matching zero rows), must never reach here.
func (c *conversationDeletions) bump(id string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.gens[id]++
	return c.gens[id]
}

// generation returns id's current generation: 0 if it has never been
// deleted. Never decreases and never forgotten — see this type's doc
// comment.
func (c *conversationDeletions) generation(id string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.gens[id]
}

// --- WebSocket upgrade and per-connection turn loop -------------------------
//
// One socket per session (ARCHITECTURE.md): messages up, curated events
// down, an explicit cancel rather than hanging up and hoping the server
// notices. The gateway owns conversation state for the life of the
// connection, seeded and persisted via conversations.go.

// inboundMessage is what the browser sends up the socket. Conversation is
// required on every "message" frame — the client generates
// each conversation's id itself (see web/src/lib/chat-socket.ts), so a
// connection may carry turns for more than one conversation over its
// lifetime as the user switches between them.
type inboundMessage struct {
	Type         string `json:"type"`
	Turn         string `json:"turn"`
	Text         string `json:"text,omitempty"`
	Conversation string `json:"conversation,omitempty"`
	// A guest's ticked services, sent on every "message" frame since it has
	// no streaming_subscriptions rows to load. Ignored on an authenticated
	// socket: an account's turn always reads its own rows, never the client.
	Providers []int `json:"providers,omitempty"`
}

// maxMessageLength bounds one chat message's text, in runes (counted via
// utf8.RuneCountInString, which walks the UTF-8 bytes without allocating a
// []rune copy just to measure it) — so a multi-byte character is never
// split. Comfortably under the 32 KiB default per-message read limit
// coder/websocket enforces, even once the rest of the envelope and a
// guest's providers list are added — the point is for this check to fire
// first, with a clear message, rather than the message ever getting big
// enough to trip the wire limit and silently kill the whole connection
// (web/src/lib/chat-socket.ts's MAX_MESSAGE_LENGTH mirrors this).
const maxMessageLength = 4000

// outboundEvent is the gateway's curated, public vocabulary — never the
// agent's internal event shape. See chat.go's translation in runTurn and
// agent/chat.py's module docstring on why the two protocols differ.
//
// "conversation_created" is the one type not translated from an agent event
// at all — see finishTurn — sent once persistence of a turn's first message
// on a conversation has actually completed, so the browser has a
// server-confirmed reason to refetch the sidebar's list rather than
// inferring one from "done" (which can arrive before the persist finishes)
// or from client-side turn-id bookkeeping (which can't survive the
// dispatching ChatPanel unmounting first).
type outboundEvent struct {
	Type    string      `json:"type"`
	Turn    string      `json:"turn"`
	Text    string      `json:"text,omitempty"`
	Picks   []agentPick `json:"picks,omitempty"`
	Relaxed []string    `json:"relaxed,omitempty"`
	// The agent's one-line explanation of a widened search, empty
	// whenever nothing was relaxed. Composed there, beside the rest of the
	// relaxation copy, rather than templated here like interpretingLine:
	// splitting it from agent/chat.py's _nothing_found_message would
	// duplicate that file's label table across two services.
	Note string `json:"note,omitempty"`
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
	turn string
	// The conversation this turn was actually dispatched for — fixed at
	// dispatch (see runTurn), independent of whichever conversation the
	// coordinator considers current by the time this reports back. A user
	// may switch conversations while a turn on the old one
	// is still finishing; finishTurn uses this, not the coordinator's
	// current pointer, to decide what to persist and whether to merge into
	// the in-memory history window.
	conversation string
	// The conversation's deletion generation captured at dispatch time
	// (conversationDeletions.generation) — compared in finishTurn against
	// the *current* generation to tell "this turn predates a delete of its
	// own conversation" (suppress) from "this turn is a legitimate new
	// message to a conversation id that happens to have been deleted at
	// some earlier point" (persist normally). See finishTurn.
	deletionGen   int
	ok            bool // false on error/cancel: nothing worth remembering happened
	userText      string
	assistantText string
	// The tmdb_id/media_type pairs shown, from a "results" event; nil for a
	// "message" turn, which showed no picks.
	titleRefs []agentTitleRef
	// Whether those refs should also join the connection's "already shown"
	// set, which is a narrower question than whether they are worth
	// persisting: only a rendered recommendation counts. Both reasons it can
	// be false are recorded where they are known -- see the "results" case in
	// runTurn.
	//
	// Guests only. An account's set is read back from messages.title_refs
	// instead (conversations.go's fetchShownRefs), which cannot see this
	// field -- see there for what that costs.
	countsAsShown bool
}

// Only the last few exchanges reach the agent — see agent/chat.py's
// MAX_HISTORY_TURNS.
//
// Five, not two, because the interpret prompt carries a follow-up's parameters
// forward out of this history ("show me more" repeats the previous request —
// agent/catalog_tool.py's _INTERPRET_SYSTEM_PROMPT). At two, the original
// request fell out of the window on the third exchange and "show me more"
// became "show me anything".
const maxHistoryExchanges = 5

// The history window, in messages: what windowHistory forwards to the agent
// and what loadConversation reads back, named once so the two cannot drift.
// The load has no reason to read deeper — the already-shown set beside it is
// its own query (conversations.go's fetchShownRefs), not a flattening of
// these rows.
//
// Which leaves the windowHistory call in the agent request with nothing to
// trim: an account's history arrives capped by the load, and a guest's was
// capped by the windowHistory call on the way in.
const maxHistoryMessages = maxHistoryExchanges * 2

// The backstop for a turn that never comes back. The agent bounds its own
// widening (catalog_tool.py's LADDER_BUDGET_SECONDS), but not the first
// TMDB call of a search, so this still has to catch a stuck one. Generous
// relative to the documented 4-8s normal case.
//
// A var, not a const, so a test can shorten it without a real 30s wait —
// same reasoning as chatTicketTTL above.
var turnDeadline = 30 * time.Second

// Comfortably inside the 60s idle timeout common to proxies and load
// balancers, so a socket between turns is never quiet long enough to be
// reaped. pingTimeout only has to outlast a round trip — a peer that cannot
// answer in ten seconds is gone, not slow.
const (
	pingInterval = 30 * time.Second
	pingTimeout  = 10 * time.Second
)

// dedupeShown keeps one entry per title, at its most recent position. A title
// is stored once per turn that showed it, so a repeat lookup of the same one
// is several rows; the agent turns this list into a set either way, so what
// this saves is request body rather than correctness.
func dedupeShown(refs []agentTitleRef) []agentTitleRef {
	seen := make(map[agentTitleRef]struct{}, len(refs))
	out := make([]agentTitleRef, 0, len(refs))
	// Newest first, so the surviving copy is the most recent sighting; the
	// caller wants oldest-first, hence the reverse. That end matters:
	// agent/chat.py's MAX_SHOWN backstop cuts from the front.
	for i := len(refs) - 1; i >= 0; i-- {
		if _, dup := seen[refs[i]]; dup {
			continue
		}
		seen[refs[i]] = struct{}{}
		out = append(out, refs[i])
	}
	slices.Reverse(out)
	return out
}

func windowHistory(history []historyTurn) []historyTurn {
	if len(history) <= maxHistoryMessages {
		return history
	}
	return history[len(history)-maxHistoryMessages:]
}

func (h *Handler) chatWS(w http.ResponseWriter, r *http.Request) {
	// Ticket → account, ?guest=1 → guest, neither → 401 (see "Guests" above).
	// A ticket that's present but bad is a rejection, never a downgrade, and
	// a request carrying both is an account — a credential is never ignored
	// in favour of a weaker mode.
	var userID string
	if ticket := r.URL.Query().Get("ticket"); ticket != "" {
		t, ok := h.tickets.consume(ticket)
		if !ok {
			writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "invalid or expired ticket"})
			return
		}
		userID = t.userID
	} else if r.URL.Query().Get("guest") != "1" {
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "missing ticket"})
		return
	}

	// Before Accept: guestIP logs against r.Context() on its error path, and
	// that context is unreliable once Accept returns (see the comment below).
	ip := guestIP(r)

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

	h.runChatConnection(connCtx, conn, userID, ip)
}

// runChatConnection is the single coordinator for one connection: it alone
// tracks the active turn, so that state needs no mutex. A dedicated
// goroutine does the blocking wsjson.Read loop and hands messages over a
// channel; turn goroutines (runTurn) run concurrently and report back over
// `done` — writes to conn are safe from multiple goroutines (coder/websocket
// handles that internally), but only one goroutine may ever call Read, which
// is why the read loop is separate and singular.
func (h *Handler) runChatConnection(ctx context.Context, conn *websocket.Conn, userID, guestIPAddr string) {
	guest := userID == ""

	// Keepalive. A chat socket is idle between turns for as long as the user
	// takes to type, and intermediaries drop a connection that goes quiet.
	// Reconnecting costs an account nothing — loadConversation re-reads every
	// message — but a guest's history lives only in this stack frame, so a
	// silent drop is the one way it can be lost while the panel still shows
	// the thread on screen. Browsers answer a ping frame themselves, so this
	// needs no client half. Its own goroutine because Read is single-threaded
	// below; Ping is safe to call concurrently with it.
	go func() {
		ticker := time.NewTicker(pingInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				pingCtx, cancel := context.WithTimeout(ctx, pingTimeout)
				err := conn.Ping(pingCtx)
				cancel()
				if err != nil {
					// Not proof the peer is gone: Ping waits for a Reader to
					// read the pong (coder/websocket's own contract), and the
					// read loop is outside Read whenever it is handing a
					// message to the coordinator below — so a busy connection
					// can miss one on a healthy socket. Returning here would
					// leave the socket unpinged for the rest of its life,
					// which is the idle-drop this exists to prevent. Retry on
					// the next tick; the read loop still owns teardown.
					continue
				}
			}
		}
	}()

	// A guest's history is this connection's alone — held here, never on the
	// Handler, or two guests naming the same client-minted UUID would read
	// each other's turns. It dies with the socket, which is the whole of a
	// guest's persistence story.
	//
	// One conversation, not a map keyed by conversation id: the id is chosen
	// by the client, so a map has no bound on its number of keys and one
	// socket could grow the heap indefinitely. A guest drives one thread at a
	// time (the panel is keyed by conversation id and remounts on a switch),
	// and this way the gateway forgets exactly what the panel forgets — going
	// back to an earlier conversation shows an empty thread and the model has
	// no memory of it either, rather than answering from turns the guest can
	// no longer see. Windowed on write, so the live thread stays bounded too.
	var guestConv string
	var guestHistory []historyTurn
	var guestTurns int
	// The titles already shown on this connection — guests only. An
	// account's are read back from messages.title_refs on every message
	// instead, beside the history: see the load below for why connection
	// memory is the wrong home for state the database already holds.
	//
	// Keyed on guestConv below rather than a conversation of its own: the
	// panel forgets a thread's cards and its text together, so one reset has
	// to govern both or the two come to disagree about which thread is live.
	//
	// Uncapped, unlike guestHistory: re-offering a title is the bug this set
	// exists to prevent, so nothing in it may age out. What bounds it is
	// h.guestTurnCap — that many turns of at most ten cards each
	// (agent/catalog_tool.py's RESULT_CEILING).
	var shownRefs []agentTitleRef
	// The one place a finished turn is handled, so the guest branch can't be
	// missed at any of the three sites below: an account persists via
	// finishTurn, a guest only remembers.
	finish := func(dbCtx, eventCtx context.Context, rec turnRecord) {
		if !guest {
			h.finishTurn(dbCtx, eventCtx, userID, rec, conn)
			return
		}
		if !rec.ok {
			return
		}
		// Keyed on the turn's own conversation, not the coordinator's current
		// one: a turn that finishes after the guest has moved on belongs to
		// the thread it was dispatched for. History and shown set reset
		// together — the panel forgets both at the same moment.
		if rec.conversation != guestConv {
			guestConv = rec.conversation
			guestHistory = nil
			shownRefs = nil
		}
		// Narrower than the history beside it: only a rendered recommendation
		// was actually offered (see runTurn's "results" case).
		if rec.countsAsShown && len(rec.titleRefs) > 0 {
			shownRefs = dedupeShown(append(shownRefs, rec.titleRefs...))
		}
		guestHistory = windowHistory(append(guestHistory,
			historyTurn{Role: "user", Text: rec.userText},
			historyTurn{Role: "assistant", Text: rec.assistantText}))
	}

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

	var cancelCurrent context.CancelFunc
	var currentTurn string
	// How many dispatched turns haven't yet reported to `done` — a
	// superseded turn's goroutine keeps running (and will still eventually
	// send) after cancelCurrent, so more than one can be outstanding at
	// once. Used by the disconnect drain below to wait for all of them, not
	// just currentTurn, so an earlier-superseded turn isn't dropped in
	// favor of persisting a stale record while the real one goes unread.
	var outstanding int
	done := make(chan turnRecord)

	defer func() {
		if cancelCurrent != nil {
			cancelCurrent()
		}
	}()

	for {
		select {
		case <-ctx.Done():
			if cancelCurrent != nil {
				cancelCurrent()
			}
			// Best-effort, not a guarantee: runTurn's own reply races this
			// same ctx.Done() as its escape hatch (see its deferred send),
			// so a turn already mid-unwind may take that hatch instead of
			// ever reaching `done` — waiting here could then hang forever,
			// which is why this only peeks. Whatever's already queued still
			// gets persisted; anything not yet delivered is an accepted,
			// bounded loss (one turn, self-healing on the user's next
			// message) rather than a reason to give the two cases separate
			// signals. context.Background() for the save: ctx is what just
			// fired, so a save derived from it would fail immediately. ctx
			// itself (not Background()) for the event context: finishTurn's
			// WS send must see this connection's own context as already
			// done, or sendTerminalEvent's cancellation guard would fire a
			// stale conversation_created down a socket that's already being
			// torn down by this same shutdown.
			for {
				select {
				case rec := <-done:
					finish(context.Background(), ctx, rec)
				default:
					return
				}
			}

		case msg, ok := <-inbound:
			if !ok {
				// Cancel, then wait: rec.ok can already be true before
				// "done" arrives (an earlier "results"/"message" event set
				// it), so an ordinary disconnect must still persist it.
				// Draining a full count, not just one peek, so an
				// earlier-superseded turn FIFO-ahead of this one doesn't
				// get skipped. Each wait still escapes via ctx.Done(): ctx
				// isn't cancelled by a mere disconnect, but if shutdown
				// happens to race this same disconnect, runTurn's own reply
				// races that identical signal (see the case above), so
				// waiting past it here would risk the same hang.
				if cancelCurrent != nil {
					cancelCurrent()
				}
				for outstanding > 0 {
					select {
					case rec := <-done:
						finish(ctx, ctx, rec)
						outstanding--
					case <-ctx.Done():
						outstanding = 0
					}
				}
				return // socket closed or read failed
			}
			switch msg.Type {
			case "message":
				// msg.Turn is the sole correlation key between this dispatch
				// and its `done` record (see the "done" case's currentTurn
				// match), and "" doubles as runChatConnection's "no turn
				// active" sentinel — so an empty turn id must never reach it.
				// Rejecting only emptiness, not requiring any particular
				// shape: unlike conversationID below, turn is never a database
				// key or parsed as a uuid, only echoed back and compared by
				// string equality.
				if msg.Turn == "" {
					h.sendTerminalEvent(ctx, conn, msg.Turn, outboundEvent{Type: "error", Text: genericErrorText})
					continue
				}
				if utf8.RuneCountInString(msg.Text) > maxMessageLength {
					h.sendTerminalEvent(ctx, conn, msg.Turn, outboundEvent{Type: "error", Text: "That message is too long — try something shorter."})
					continue
				}
				parsedConv, err := uuid.Parse(msg.Conversation)
				if err != nil {
					// Malformed frame — every real client always supplies a
					// well-formed uuid; conversations.id is
					// a uuid column, so letting anything else reach a query
					// would surface as a database type-cast error rather
					// than a clean signal. Reachable today via /chat/[id]'s
					// route param — now also checked server-side before
					// ChatPanel ever mounts, but this is defence in depth,
					// not the only guard.
					h.sendTerminalEvent(ctx, conn, msg.Turn, outboundEvent{Type: "error", Text: genericErrorText})
					continue
				}
				// Canonicalized once, here, and used for every reference to
				// this conversation below. conversationDeletions is keyed by
				// Go's case-sensitive string equality, unlike Postgres's
				// uuid column — without this, "Abc...1" and "abc...1"
				// collide at the database but silently miss each other in
				// this in-memory map, so a delete recorded under one casing
				// would never be seen by a generation() check keyed under
				// the other.
				conversationID := parsedConv.String()

				// Before the supersede below, like the uuid check above: a
				// frame rejected here must not have cancelled the turn it
				// failed to replace. An account sending providers is the
				// same kind of malformed frame — its services come from its
				// own rows, never the client — rejected so a client bug
				// fails loudly instead of being silently ignored.
				var history []historyTurn
				var providers []int
				if guest {
					if !validGuestProviders(msg.Providers) {
						h.sendTerminalEvent(ctx, conn, msg.Turn, outboundEvent{Type: "error", Text: genericErrorText})
						continue
					}
					// Only a turn that can reach the model counts: with no
					// services picked the agent answers from a template
					// (agent/chat.py's NO_PROVIDERS_MESSAGE), and that nudge
					// toward Connections must never turn into a sign-in wall.
					// Counts attempts, not answers: a turn that fails before
					// or at the model still spent a slot; a reload resets it.
					if len(msg.Providers) > 0 {
						// Cap first — a free per-socket check — so an
						// already-capped socket never spends a shared token.
						if guestTurns >= h.guestTurnCap {
							h.sendTerminalEvent(ctx, conn, msg.Turn, outboundEvent{Type: "error", Text: guestTurnCapText})
							continue
						}
						if !h.guestLimiters.allow(guestIPAddr) {
							h.sendTerminalEvent(ctx, conn, msg.Turn, outboundEvent{Type: "error", Text: guestRateLimitText})
							continue
						}
						guestTurns++
					}
					providers = msg.Providers
					if conversationID == guestConv {
						history = guestHistory
					}
				} else if len(msg.Providers) > 0 {
					h.sendTerminalEvent(ctx, conn, msg.Turn, outboundEvent{Type: "error", Text: genericErrorText})
					continue
				}

				// Captured immediately on receipt, before the load below —
				// not after it returns. Reading this after the load would
				// let a delete that lands while the load is still in flight
				// get captured as though it predated this dispatch, so
				// finishTurn's `>` comparison would fail to suppress the
				// persist and silently resurrect the conversation the
				// delete just removed. See
				// TestChatDeleteDuringInFlightLoadStillSuppressesTheStragglingTurn.
				deletionGen := h.conversationDeletions.generation(conversationID)

				// A new message supersedes whatever is in flight — belt and
				// suspenders alongside the client disabling send while
				// streaming: the gateway never trusts the
				// client alone to keep two turns from overlapping.
				if cancelCurrent != nil {
					cancelCurrent()
				}

				var storedShown []agentTitleRef
				if !guest {
					// Loaded fresh on every message rather than cached: a
					// read is cheap, and a cache here would need
					// invalidation rules the database keeps outrunning.
					// (guestHistory above has no database to diverge from.)
					loadCtx, cancel := context.WithTimeout(ctx, provisionTimeout)
					history, storedShown, err = h.loadConversation(loadCtx, userID, conversationID)
					cancel()
					if err != nil {
						// Logged, not fatal to the turn: loadConversation's
						// own contract already treats "nothing saved yet"
						// as a normal nil result, so a transient read
						// failure degrades to an empty history window
						// instead of blocking the turn — the next message
						// on this conversation simply retries the load.
						// One transaction carries the already-shown set
						// too, so this costs that as well and the turn may
						// re-offer a title; a repeat beats refusing to
						// answer.
						slog.ErrorContext(ctx, "conversation load failed", "error", dbError(err))
						history = nil
						storedShown = nil
					}
				}

				turnCtx, cancel := context.WithTimeout(ctx, turnDeadline)
				cancelCurrent = cancel
				currentTurn = msg.Turn
				outstanding++
				// Only this conversation's own shown set: switching threads
				// must not suppress a title in the new one because the old
				// one showed it. An account's came from the load above (keyed
				// on this conversation by the query itself); a guest's from
				// the in-memory accumulation, which needs the check.
				var shown []agentTitleRef
				if guest {
					if conversationID == guestConv {
						shown = shownRefs
					}
				} else {
					shown = dedupeShown(storedShown)
				}
				go h.runTurn(turnCtx, ctx, conn, userID, msg.Turn, conversationID, msg.Text, providers, history, shown, deletionGen, done)
			case "cancel":
				if cancelCurrent != nil && currentTurn == msg.Turn {
					cancelCurrent()
				}
			}

		case rec := <-done:
			outstanding--
			// cancelCurrent != nil is defence in depth, not load-bearing:
			// the "message" case's emptiness check already keeps msg.Turn
			// from colliding with currentTurn's "no turn active" sentinel.
			// Turn-id uniqueness beyond non-emptiness is client convention
			// (each client mints a crypto.randomUUID() per message —
			// web/src/lib/chat-socket.ts), not enforced here.
			if rec.turn == currentTurn && cancelCurrent != nil {
				// Release turnDeadline's timer now rather than letting it
				// idle until it fires on its own — same reason the
				// "message" and "cancel" cases above call this eagerly.
				cancelCurrent()
				cancelCurrent = nil
				currentTurn = ""
			}
			finish(ctx, ctx, rec)
		}
	}
}

// finishTurn persists rec (if it succeeded) against the conversation it was
// actually run against, rec.conversation — never whatever the coordinator
// considers current now, since the user may have switched to a different
// conversation while this turn was still finishing.
//
// Skips persisting when rec.conversation's generation has moved on since
// dispatch (rec.deletionGen): a straggler would otherwise resurrect the row
// the delete just removed. Only a straggler predating the delete has a
// smaller captured generation — a turn dispatched after one captures the
// post-delete value and is never suppressed.
//
// Two contexts, same split as runTurn's: dbCtx governs the save, eventCtx
// gates the sendTerminalEvent calls below. Identical at every call site but
// runChatConnection's shutdown drain, where dbCtx is context.Background() (ctx
// has fired, so a save derived from it fails immediately) and eventCtx stays
// ctx — already Done, so no stale event is written down a closing connection.
func (h *Handler) finishTurn(dbCtx, eventCtx context.Context, userID string, rec turnRecord, conn *websocket.Conn) {
	if !rec.ok {
		return
	}
	if h.conversationDeletions.generation(rec.conversation) > rec.deletionGen {
		slog.InfoContext(dbCtx, "dropped a turn for a since-deleted conversation", "turn", rec.turn, "conversation", rec.conversation)
		return
	}
	// Blocking, not backgrounded: a two-row insert, not a TMDB round trip
	// — and the simpler design shipped cleanly beats the complex one shipped
	// badly.
	saveCtx, cancel := context.WithTimeout(dbCtx, provisionTimeout)
	created, err := h.saveMessages(saveCtx, userID, rec.conversation, rec.userText, rec.assistantText, rec.titleRefs)
	cancel()
	if err != nil {
		// A conversation id this session can't write into (foreign, or
		// deleted between dispatch and this persist attempt) is not a server
		// failure — it's saveMessages' own explicit ownership check
		// (errConversationNotWritable) or, failing that, RLS's
		// message_isolation policy as a second, independent backstop
		// (isRLSRejection). Either way the browser must be told: the turn's
		// answer was shown but will never be saved, and nothing else will
		// ever tell the user that — a turn always ends in a terminal frame
		// (ARCHITECTURE.md). The agent has already spent its call by the time
		// this is caught; checking ownership earlier would mean either bypassing RLS
		// or duplicating saveMessages' own conflict logic ahead of the turn,
		// both bigger changes than this path warrants.
		//
		// Every other persist failure stays a silent log: what the user
		// already saw stays on screen, and the next message on this
		// conversation self-heals through saveMessages' own conflict
		// handling.
		if errors.Is(err, errConversationNotWritable) || isRLSRejection(err) {
			slog.WarnContext(dbCtx, "message persist rejected: conversation not writable",
				"turn", rec.turn, "conversation", rec.conversation)
			h.sendTerminalEvent(eventCtx, conn, rec.turn, outboundEvent{Type: "error", Text: "This conversation isn't available."})
			return
		}
		slog.ErrorContext(dbCtx, "message persist failed", "turn", rec.turn, "conversation", rec.conversation, "error", dbError(err))
		return
	}
	if created {
		// Tells the sidebar a brand-new conversation now exists, straight
		// from the write that actually created it — not a client guess
		// about which turn was "its own" (that guess can't survive the
		// dispatching panel unmounting: a conversation switch, or an
		// abandoned "New chat") and not tied to the browser ever seeing a
		// "done" event (a dropped stream sends only "error" — see
		// runTurn's post-loop fallback — even though the turn still
		// persisted here). Gated on err == nil, not just created: a
		// failed messages insert rolls back the whole transaction,
		// including the conversations insert that set created, so
		// nothing was actually committed.
		h.sendTerminalEvent(eventCtx, conn, rec.turn, outboundEvent{Type: "conversation_created"})
	}
}

// sendProgressEvent writes a progress event (interpreting/results/token),
// gated on turnCtx: a stale one for a turn that's since been cancelled,
// superseded, or timed out is correctly dropped. The counterpart to
// sendTerminalEvent below — see that function's doc comment for why the two
// must never be interchanged, and never collapsed back into one function
// taking a plain context.Context.
func (h *Handler) sendProgressEvent(turnCtx context.Context, conn *websocket.Conn, turn string, ev outboundEvent) bool {
	return h.writeEvent(turnCtx, conn, turn, ev)
}

// sendTerminalEvent writes a turn's terminal event — done, error, the
// runChatConnection-level rejections that end a "message" frame before any
// turn is even dispatched, and finishTurn's conversation_created/persist-
// failure events — gated on connCtx, never turnCtx. turnCtx being Done is
// exactly the situation a terminal frame exists to report, so gating one on
// turnCtx would silently swallow the very deadline/cancel/supersede signal
// it's supposed to deliver.
//
// Two named functions rather than one sendEvent taking a context: both
// contexts are in scope wherever a turn runs, so a single entry point leaves
// each call site free to pass the wrong one. "sendTerminalEvent(turnCtx, ...)"
// is a visible contradiction; "sendEvent(turnCtx, ...)" is not.
func (h *Handler) sendTerminalEvent(connCtx context.Context, conn *websocket.Conn, turn string, ev outboundEvent) bool {
	return h.writeEvent(connCtx, conn, turn, ev)
}

// writeEvent is sendProgressEvent's and sendTerminalEvent's shared
// implementation, and is not called directly by anything else: it enforces
// nothing about which context it is handed.
//
// Checks ctx first so a write that's no longer wanted never reaches the
// browser after the fact. Real call sites always pass a live connection, so
// conn == nil is reachable only from unit tests calling finishTurn directly —
// it is here so correctness doesn't rest on every fake returning
// created == false, which would otherwise nil-deref the created branch.
// Reports whether the event reached the browser: most callers ignore that,
// but the "results" case needs it, since picks the user never saw must not
// join the already-shown set.
func (h *Handler) writeEvent(ctx context.Context, conn *websocket.Conn, turn string, ev outboundEvent) bool {
	if ctx.Err() != nil || conn == nil {
		return false
	}
	ev.Turn = turn
	writeCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := wsjson.Write(writeCtx, conn, ev); err != nil {
		slog.WarnContext(ctx, "chat event write failed", "turn", turn, "error", err.Error())
		return false
	}
	return true
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

// summarizePicks writes the assistant's side of the turn into stored history.
// That text is both what a reload renders (conversations.go does not surface
// title_refs) and what feeds the next turn's interpret(), so it has to say
// what actually happened: "Suggested" on a turn that answered a lookup would
// claim a recommendation the assistant never made -- and would then be read
// back as one.
func summarizePicks(picks []agentPick, isLookup bool) string {
	if len(picks) == 0 {
		return ""
	}
	// Count first, then qualify only the names that repeat. A lookup answers
	// with every title carrying the name, so "Dune, Dune" and "Fargo, Fargo"
	// are the normal case, not an edge one -- and this text is the whole
	// assistant side of the turn, so an unqualified repeat is what a reload
	// renders and what the next turn's interpret() reads back.
	seen := make(map[string]int, len(picks))
	for _, p := range picks {
		seen[p.Title]++
	}
	titles := make([]string, len(picks))
	for i, p := range picks {
		titles[i] = p.Title
		if seen[p.Title] > 1 && p.Year != nil {
			titles[i] = fmt.Sprintf("%s (%d)", p.Title, *p.Year)
		}
	}
	verb := "Suggested: "
	if isLookup {
		verb = "Looked up: "
	}
	return verb + strings.Join(titles, ", ")
}

// runTurn loads the context the agent needs, calls it, and streams translated
// events down the socket — the concrete version of ARCHITECTURE.md's "Flow:
// asking a question." Reports exactly once to `done`, always, so
// runChatConnection can clear the active-turn state whether the turn
// succeeded, errored, or was cancelled.
//
// Takes two contexts: turnCtx governs this turn's own work (cancelled by
// supersede, explicit cancel, or turnDeadline expiring — see
// runChatConnection) and gates every *progress* write via sendProgressEvent
// (interpreting/results/token — a stale one for an abnormally-ended turn
// must still be dropped). connCtx gates the deferred send's escape hatch
// below and every *terminal* write via sendTerminalEvent (done, error, and
// the post-loop fallback) — turnCtx being Done is exactly the condition a
// terminal frame exists to report, so gating those on turnCtx instead is
// what let a cancelled/superseded/timed-out turn end with no terminal frame
// at all and the composer stuck disabled. See sendTerminalEvent's own
// doc comment for the same split from the other side.
func (h *Handler) runTurn(
	turnCtx context.Context,
	connCtx context.Context,
	conn *websocket.Conn,
	userID, turnID, conversationID, text string,
	guestProviders []int,
	history []historyTurn,
	shown []agentTitleRef,
	deletionGen int,
	done chan<- turnRecord,
) {
	rec := turnRecord{turn: turnID, conversation: conversationID, deletionGen: deletionGen}
	defer func() {
		// done is unbuffered and runChatConnection stops reading it once the
		// connection dies, so this send needs an escape hatch or it leaks the
		// goroutine when the client closes the tab mid-answer. It must be
		// connCtx: turnCtx is also Done on a supersede or deadline expiry,
		// while the coordinator is still reading `done`, and select would then
		// drop a completed turn's record about half the time.
		select {
		case done <- rec:
		case <-connCtx.Done():
		}
	}()

	// A guest (userID == "", see "Guests" above) loads from the message, an
	// account from its rows; both fail the turn the same way. These two
	// branches are the only ones on userID in the turn — everything from
	// callAgent down is identical.
	var chatCtx chatContext
	var err error
	if userID == "" {
		chatCtx, err = h.loadGuestChatCtx(turnCtx, guestProviders)
	} else {
		chatCtx, err = h.loadChatCtx(turnCtx, userID)
	}
	if err != nil {
		if !errors.Is(err, context.Canceled) {
			slog.ErrorContext(turnCtx, "chat context load failed", "turn", turnID, "error", dbError(err))
		}
		h.sendTerminalEvent(connCtx, conn, turnID, outboundEvent{Type: "error", Text: genericErrorText})
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
	// The chat is the one surface that cannot degrade (ARCHITECTURE.md's
	// Failure rules), and degrading would silently re-show titles the user
	// marked seen — breaking the promise that a title marked seen is never
	// suggested again, in the direction users notice.
	// Skipped for a caller with no subscriptions: the agent answers that with
	// its "pick your services" message before it ever looks at verdicts
	// (agent/chat.py's stream_chat), so loading them is work whose result is
	// unused - and a title_verdicts problem must not be the thing that stops a
	// brand-new user, who has no verdicts anyway, from being told to pick some.
	// A guest has no verdicts to load at all.
	var verdicts []Verdict
	if userID != "" && len(chatCtx.Providers) > 0 {
		verdicts, err = h.loadVerdicts(turnCtx, userID)
		if err != nil {
			if !errors.Is(err, context.Canceled) {
				slog.ErrorContext(turnCtx, "chat verdict load failed", "turn", turnID, "error", dbError(err))
			}
			h.sendTerminalEvent(connCtx, conn, turnID, outboundEvent{Type: "error", Text: genericErrorText})
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
		Shown:              shown,
	})
	if err != nil {
		// Unlike the two blocks above, this one still has to pick between
		// two different texts: agentUnavailableText specifically blames the
		// model for a genuine call failure, which would be a wrong (and
		// user-visible) accusation for an ordinary cancel/supersede/timeout,
		// so that case falls back to the same genericErrorText the other
		// blocks use unconditionally.
		// errText, not text: this function's text parameter is the user's
		// own chat message, still referenced elsewhere below — reusing that
		// name here for an unrelated error string would shadow it for the
		// rest of this block.
		errText := agentUnavailableText
		if errors.Is(err, context.Canceled) {
			errText = genericErrorText
		} else {
			slog.ErrorContext(turnCtx, "agent call failed", "turn", turnID, "error", err.Error())
		}
		h.sendTerminalEvent(connCtx, conn, turnID, outboundEvent{Type: "error", Text: errText})
		return
	}

	// turnCtx.Err() is checked inside each *progress* case, never once before
	// the switch: "done" and "error" must not check it at all, so the event
	// that ends the turn can't be discarded by a cancel racing this receive.
	//
	// The race is real either way: `events` is unbuffered and newAgentCaller
	// runs a mirrored select against this receive, so a "results"/"message"
	// can already be in `ev` when turnCtx fires and is then dropped by the
	// `continue` below. That residual is accepted — the turn still ends with a
	// correct terminal frame, but rec.ok stays false and nothing is persisted.
	// A clean "done" with no row in messages is this case, not a bug in
	// finishTurn.
	//
	// "done" and "error" gate on connCtx and return instead, reporting
	// correctly whatever turnCtx is doing. If turnCtx fires before the agent's
	// HTTP call notices, that call's cancellation closes `events`
	// (newAgentCaller) and the post-loop fallback below handles it.
	for ev := range events {
		switch ev.Type {
		case "intent":
			if turnCtx.Err() != nil {
				continue
			}
			var intent agentIntent
			if err := json.Unmarshal(ev.Intent, &intent); err == nil {
				line := interpretingLine(intent, chatCtx.ProviderNames)
				h.sendProgressEvent(turnCtx, conn, turnID, outboundEvent{Type: "interpreting", Text: line})
			} else {
				slog.WarnContext(turnCtx, "chat intent decode failed", "turn", turnID, "error", err.Error())
			}
		case "results":
			if turnCtx.Err() != nil {
				continue
			}
			rendered := h.sendProgressEvent(turnCtx, conn, turnID, outboundEvent{
				Type: "results", Picks: ev.Picks, Relaxed: ev.Relaxed, Note: ev.Note,
			})
			// Guarded on a non-empty summary, not unconditional like
			// "message" below: zero picks is a real, reachable state
			// (summarizePicks returns "" for it) that must not overwrite an
			// ok=true left by an earlier event in this same stream. An empty
			// ev.Text on "message" isn't reachable the same way — every
			// message-type event agent/chat.py emits comes from a non-empty
			// template — so no matching guard is needed there.
			if summary := summarizePicks(ev.Picks, ev.Kind == "lookup"); summary != "" {
				rec.ok = true
				rec.userText = text
				rec.assistantText = summary
				rec.titleRefs = make([]agentTitleRef, len(ev.Picks))
				for i, p := range ev.Picks {
					rec.titleRefs[i] = agentTitleRef{TMDBID: p.TMDBID, MediaType: p.MediaType}
				}
				// Narrower than the persist above: only a rendered
				// recommendation was actually offered. A lookup is exempt for
				// the same reason search() won't filter one ("is Dune on
				// Netflix" is the same question however often it is asked), and
				// cards that never reached the socket showed nothing to
				// suppress.
				rec.countsAsShown = rendered && ev.Kind != "lookup"
			}
		case "message":
			if turnCtx.Err() != nil {
				continue
			}
			h.sendProgressEvent(turnCtx, conn, turnID, outboundEvent{Type: "token", Text: ev.Text})
			rec.ok = true
			rec.userText = text
			rec.assistantText = ev.Text
			// Not reachable today (agent/chat.py emits "results" xor
			// "message" per turn) but cheap to keep correct: without this,
			// a "results" event followed by a "message" event in the same
			// stream would leave rec.titleRefs pointing at picks that don't
			// match rec.assistantText's replacement text.
			rec.titleRefs = nil
			rec.countsAsShown = false
		case "error":
			h.sendTerminalEvent(connCtx, conn, turnID, outboundEvent{Type: "error", Text: friendlyError(ev.Reason)})
			rec.ok = false
			return
		case "done":
			h.sendTerminalEvent(connCtx, conn, turnID, outboundEvent{Type: "done"})
			return
		default:
			slog.WarnContext(turnCtx, "unknown agent event type", "turn", turnID, "type", ev.Type)
		}
	}

	// Reached only if the channel closed without "done" or "error" ever
	// firing (both return above) — a transport failure (see newAgentCaller's
	// scanner.Err() log) or an agent-side bug that dropped the stream mid-way.
	// A dropped stream keeps what arrived, marks it incomplete and offers a
	// retry, rather than stopping mid-sentence looking finished
	// (ARCHITECTURE.md's Failure rules). rec.ok may already be true from an
	// earlier "results"/ "message" event in this same stream — that's left
	// as-is, not reset: what the user already saw stays what the user saw
	// (ARCHITECTURE.md's Failure rules), this fallback only adds the missing
	// "and it didn't finish cleanly" signal on top. connCtx, not turnCtx: on the
	// 30s turnDeadline-expiry path turnCtx is reliably Done by the time this
	// line runs, so gating on it would silently drop the one event this fallback
	// exists to send.
	h.sendTerminalEvent(connCtx, conn, turnID, outboundEvent{Type: "error", Text: genericErrorText})
}

// interpretingLine is templated from the agent's DiscoverIntent, never a
// second model call — it doubles as a comprehension check, so
// it must be on screen well before the full pipeline finishes.
func interpretingLine(intent agentIntent, providerNames []string) string {
	// A named-title lookup answers with that title, not a filtered search, so
	// none of the clauses below apply — "Looking for movies on Netflix and
	// Hulu" would contradict the card that follows it. See the agent's
	// catalog_tool.lookup_title.
	if intent.Title != "" {
		// Typographic quotes, not %q: %q is strconv.Quote, which renders a
		// title containing quotes with literal backslashes on screen.
		return "Looking up \u201c" + intent.Title + "\u201d"
	}

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
		// "or", not humanJoin's "and": tmdb.py sends with_keywords OR-joined,
		// so a title carrying either tag qualifies, and the interpret prompt
		// now actively encourages near-synonyms for one mood ("slow burn" and
		// "tense" together). Read as a conjunction, this line promises a
		// filter the results provably do not satisfy.
		clauses = append(clauses, "about "+strings.Join(intent.Keywords, " or "))
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
	// country, kept fresh by providers.go's lazy cache. Until then, or if the
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
// verdicts.go's loadVerdicts.
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

// cachedProviderNames resolves wanted ids to names from the country's cached
// streaming_providers row — the one read both loadChatContext (inside its
// transaction) and loadGuestChatContext (straight off the pool) share, so it
// takes queryRower (main.go), what pgx.Tx and *pgxpool.Pool have in common.
// No cached row yet (providers.go fills it lazily on read) degrades to no names,
// never an error: the interpreting line drops its "on …" clause, the turn
// goes on.
func cachedProviderNames(ctx context.Context, q queryRower, country string, wanted []int) ([]string, error) {
	var providersJSON []byte
	err := q.QueryRow(ctx,
		`select providers from streaming_providers where country = $1`, country).
		Scan(&providersJSON)
	switch {
	case errors.Is(err, pgx.ErrNoRows):
		return []string{}, nil
	case err != nil:
		return nil, err
	}
	return parseProviderNames(providersJSON, wanted), nil
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

			names, err := cachedProviderNames(ctx, tx, cc.Region, cc.Providers)
			if err != nil {
				return err
			}
			cc.ProviderNames = names
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
	// can name the caller's services without the agent
	// needing its own TMDB lookup.
	WatchProviderNames []string      `json:"watch_provider_names,omitempty"`
	History            []historyTurn `json:"history,omitempty"`
	// The caller's whole verdict set; agent/catalog_tool.py's search()
	// decides what each value means.
	Verdicts []Verdict `json:"verdicts,omitempty"`
	// Everything already put on screen for this conversation, so
	// "show me 10 more" means ten different titles. An account's comes from
	// messages.title_refs, read back on every message beside the history
	// (conversations.go's fetchShownRefs); a guest's, having no rows,
	// accumulates on the connection. Sent whole rather than windowed: the
	// agent pages deeper the more of it there is, and applies its own ceiling
	// (agent/chat.py's MAX_SHOWN).
	Shown []agentTitleRef `json:"shown,omitempty"`
}

// agentIntent mirrors DiscoverIntent.model_dump(exclude_none=True) — see
// agent/catalog_tool.py. Pointers on the fields Pydantic can leave unset.
//
// Deliberately omits sort_by and limit: interpretingLine() is a comprehension
// check on *what* the user asked for, not *how* the search runs — the user
// already knows how many results they asked for and doesn't need it read
// back. Title is the one field here that changes which line gets built at
// all rather than adding a clause to it — a named-title turn skips
// discover() entirely on the agent side. Every other DiscoverIntent field is
// here; an omitted field silently vanishes on decode (json.Unmarshal drops
// unknown keys), so if you add a field to DiscoverIntent that should show up
// in the interpreting line, it must be added here too — nothing else catches
// the gap.
type agentIntent struct {
	MediaType         string   `json:"media_type"`
	Title             string   `json:"title"`
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
// title-card rendering is the browser's job, not this one's.
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
	// True only for a watchlist row whose TMDB lookup
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
	Type string `json:"type"`
	// Which path search() took, on "results" only: "search" or "lookup".
	// The agent records this where the branch is actually taken, so stored
	// history reflects what the turn did rather than what its intent looked
	// like. interpretingLine cannot use it and reads intent.Title instead:
	// the interpreting line is built from the "intent" event, which is
	// emitted before search() has chosen a path (agent/catalog_tool.py fires
	// on_intent first, deliberately). The two agree today; if they ever
	// diverge, the cost is a premature interpreting line, never a wrong verb
	// written into history.
	Kind    string          `json:"kind,omitempty"`
	Intent  json.RawMessage `json:"intent,omitempty"`
	Picks   []agentPick     `json:"picks,omitempty"`
	Relaxed []string        `json:"relaxed,omitempty"`
	Note    string          `json:"note,omitempty"`
	Text    string          `json:"text,omitempty"`
	Reason  string          `json:"reason,omitempty"`
}

// agentCaller is the injection seam for the agent HTTP call — same shape as
// ensureUser in main.go: production wiring is newAgentCaller, tests fake it
// directly rather than standing up a real agent process.
type agentCaller func(ctx context.Context, req agentChatRequest) (<-chan agentEvent, error)

// newAgentCaller streams agent/main.py's POST /chat response (chunked NDJSON,
// not a second WebSocket — ARCHITECTURE.md's "One socket per session") and
// parses it one line at
// a time onto a channel. Cancelling ctx aborts the underlying HTTP request,
// which is the whole of this codebase's cancel story for the agent call — see
// chat.go's runTurn and ARCHITECTURE.md's "One socket per session."
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
			// Titles carry overviews and posters for up to 20 candidates —
			// comfortably past the 64KiB default, nowhere near unbounded.
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
			// hanging — a dropped stream keeps what arrived and marks it
			// incomplete, never stopping mid-sentence looking finished.
			if err := scanner.Err(); err != nil {
				slog.WarnContext(ctx, "agent stream ended abnormally", "error", err.Error())
			}
		}()
		return events, nil
	}
}
