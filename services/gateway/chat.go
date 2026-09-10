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
	"github.com/google/uuid"
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

// chatTicketEntry is one minted ticket paired with its expiry deadline —
// chatTicketStore's own inlined version of the mutex-protected, self-expiring
// map it used to share with conversationDeletions (see that type's doc
// comment for why the two have since diverged). Not generic anymore, now
// that chatTicketStore is the sole user.
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

// scheduleExpiry removes id after chatTicketTTL. Every ticket id is a fresh
// 32-byte crypto/rand value (mint, above) and is never reused, so — unlike
// the old shared expiringMap this once wrapped — there is no second write to
// the same key to defend against here; a plain unconditional delete is
// correct.
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
// Why guests exist and what they cost: DECISIONS.md "Optional sign-in".
//
// What this file enforces: a guest has no users row, so nothing user-scoped is
// read or written for it; ?guest=1 declares a guest and a ticket declares an
// account, while neither is a 401 — an absent credential never silently
// becomes a different mode; within a connection userID == "" is the guest, and
// mint above refuses to ever issue that value.
//
// No rate limit here, per-IP or global: a model 429 already reaches the user as
// "try again in a moment" (friendlyError), so throttling would only move the
// refusal without bounding the shared quota.

// Every account is provisioned into users.country's default
// (20260831230634_streaming.sql), so a guest hard-wired to the same value sees
// the same catalog an account does today. Revisit alongside any real country
// setting.
const guestRegion = "US"

// Bounds one message's providers list — the picker offers a few dozen per
// region, so anything past this is a malformed frame, not a real selection.
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
// Deleting a conversation (conversations.go's deleteConversationHandler) is a
// plain HTTP request, racing whatever WS turn might still be in flight on
// that same conversation id. Without this, a straggling turn's own persist
// (finishTurn, via saveMessages) can land after the delete and silently
// recreate the conversation row through saveMessages' own
// on-conflict-do-nothing insert — that insert can't tell "already exists
// because it's ongoing" from "already exists because it was just deleted."
//
// A boolean "was this deleted recently" tombstone can't make that
// distinction from "a brand-new, legitimate turn was dispatched after the
// delete, against the same conversation id" (e.g. the browser's back button
// reopening a since-deleted conversation's URL and sending a fresh message
// to it) — both look identical to a plain within-TTL check, and the second
// case would be wrongly dropped. A monotonic per-conversation generation
// counter fixes that: dispatch captures the conversation's current
// generation (runChatConnection), and finishTurn only suppresses persistence
// if the generation has moved on since — i.e. a delete actually happened
// strictly after this turn was dispatched, not merely at some point within
// the TTL window.
//
// Deliberately never expired. An earlier version wrapped an expiringMap and
// aged each id's generation out after a 40s TTL, sized to outlast a
// straggling turn's own finishTurn call — safe for that alone (a turn can
// only still be in flight for roughly turnDeadline+provisionTimeout after
// dispatch, comfortably under 40s), but a since-removed per-connection
// history cache also read this same generation to decide whether its own
// long-lived in-memory entry was still trustworthy, and could live far past
// 40 seconds. Once the TTL passed with nothing to notice the gap,
// generation() reverted to 0 ("never deleted"), letting that cache silently
// serve pre-deletion history as agent context indefinitely — the bug this
// permanent map exists to prevent. It stays permanent even with that cache
// gone: a real delete must stay remembered for as long as anything might
// still hold pre-delete data, and nothing bounds how long a turn dispatched
// against a long-open connection might straggle. Growth is bounded by how
// many conversations are ever actually deleted across the app's whole
// lifetime — a low-volume, user-driven event — so an unbounded map is the
// simpler, correct choice at this app's scale.
//
// In-memory, single-instance — same limitation as chatTicketStore: if this
// ever runs more than one gateway replica, a delete landing on one instance
// while a straggler or a cache entry lives on another would miss this check
// entirely.
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
// One socket per session (../../DECISIONS.md): messages up, curated events
// down, an explicit cancel rather than hanging up and hoping the server
// notices. The gateway owns conversation state for the life of the
// connection, seeded and persisted via conversations.go (TASKS.md T20).

// inboundMessage is what the browser sends up the socket. Conversation is
// required on every "message" frame (TASKS.md T20.5) — the client generates
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
	// may switch conversations (TASKS.md T20.5) while a turn on the old one
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

// Comfortably inside the 60s idle timeout common to proxies and load
// balancers, so a socket between turns is never quiet long enough to be
// reaped. pingTimeout only has to outlast a round trip — a peer that cannot
// answer in ten seconds is gone, not slow.
const (
	pingInterval = 30 * time.Second
	pingTimeout  = 10 * time.Second
)

func windowHistory(history []historyTurn) []historyTurn {
	n := maxHistoryExchanges * 2
	if len(history) <= n {
		return history
	}
	return history[len(history)-n:]
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

	h.runChatConnection(connCtx, conn, userID)
}

// runChatConnection is the single coordinator for one connection: it alone
// tracks the active turn, so that state needs no mutex. A dedicated
// goroutine does the blocking wsjson.Read loop and hands messages over a
// channel; turn goroutines (runTurn) run concurrently and report back over
// `done` — writes to conn are safe from multiple goroutines (coder/websocket
// handles that internally), but only one goroutine may ever call Read, which
// is why the read loop is separate and singular.
func (h *Handler) runChatConnection(ctx context.Context, conn *websocket.Conn, userID string) {
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
		// the thread it was dispatched for.
		if rec.conversation != guestConv {
			guestConv = rec.conversation
			guestHistory = nil
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
			// done, or sendEvent's cancellation guard would fire a stale
			// conversation_created down a socket that's already being torn
			// down by this same shutdown.
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
				parsedConv, err := uuid.Parse(msg.Conversation)
				if err != nil {
					// Malformed frame — every real client always supplies a
					// well-formed uuid (TASKS.md T20.5); conversations.id is
					// a uuid column, so letting anything else reach a query
					// would surface as a database type-cast error rather
					// than a clean signal. Reachable today via /chat/[id]'s
					// route param — now also checked server-side before
					// ChatPanel ever mounts, but this is defence in depth,
					// not the only guard.
					h.sendEvent(ctx, conn, msg.Turn, outboundEvent{Type: "error", Text: genericErrorText})
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
						h.sendEvent(ctx, conn, msg.Turn, outboundEvent{Type: "error", Text: genericErrorText})
						continue
					}
					// Only a turn that can reach the model counts: with no
					// services picked the agent answers from a template
					// (agent/chat.py's NO_PROVIDERS_MESSAGE), and that nudge
					// toward Connections must never turn into a sign-in wall.
					// Counts attempts, not answers: a turn that fails before
					// or at the model still spent a slot; a reload resets it.
					if len(msg.Providers) > 0 {
						if guestTurns >= h.guestTurnCap {
							h.sendEvent(ctx, conn, msg.Turn, outboundEvent{Type: "error", Text: guestTurnCapText})
							continue
						}
						guestTurns++
					}
					providers = msg.Providers
					if conversationID == guestConv {
						history = guestHistory
					}
				} else if len(msg.Providers) > 0 {
					h.sendEvent(ctx, conn, msg.Turn, outboundEvent{Type: "error", Text: genericErrorText})
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
				// streaming (TASKS.md T14): the gateway never trusts the
				// client alone to keep two turns from overlapping.
				if cancelCurrent != nil {
					cancelCurrent()
				}

				if !guest {
					// Loaded fresh on every message rather than cached: a
					// per-connection history cache used to live here, but
					// its invalidation rules were the source of most of
					// this feature's bugs, for a database read cheap enough
					// to just repeat. (guestHistory above is that cache's
					// shape again, but with no database to fall out of sync
					// with, none of those rules apply to it.)
					loadCtx, cancel := context.WithTimeout(ctx, provisionTimeout)
					history, err = h.loadConversation(loadCtx, userID, conversationID)
					cancel()
					if err != nil {
						// Logged, not fatal to the turn: loadConversation's
						// own contract already treats "nothing saved yet"
						// as a normal nil result, so a transient read
						// failure degrades to an empty history window
						// instead of blocking the turn — the next message
						// on this conversation simply retries the load.
						slog.ErrorContext(ctx, "conversation load failed", "error", dbError(err))
						history = nil
					}
				}

				turnCtx, cancel := context.WithTimeout(ctx, turnDeadline)
				cancelCurrent = cancel
				currentTurn = msg.Turn
				outstanding++
				go h.runTurn(turnCtx, ctx, conn, userID, msg.Turn, conversationID, msg.Text, providers, history, deletionGen, done)
			case "cancel":
				if cancelCurrent != nil && currentTurn == msg.Turn {
					cancelCurrent()
				}
			}

		case rec := <-done:
			outstanding--
			if rec.turn == currentTurn {
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
// conversation (TASKS.md T20.5) while this turn was still finishing.
//
// Skips persisting when rec.conversation's generation has moved on since
// this turn was dispatched (rec.deletionGen) — meaning a real delete landed
// strictly after dispatch, so this turn's own persist would otherwise
// resurrect the conversation the delete just removed via saveMessages'
// on-conflict-do-nothing insert. A turn dispatched *after* a delete captures
// the post-delete generation itself, so this comparison never suppresses it
// — only a genuine straggler that predates the delete has a strictly
// smaller captured generation than the current one.
//
// Takes two contexts, same split as runTurn's turnCtx/connCtx: dbCtx governs
// the save to the database, and eventCtx gates the "conversation_created"
// send via sendEvent's own cancellation guard. They're the same context at
// every call site except runChatConnection's shutdown drain, where dbCtx is
// context.Background() (ctx has already fired, so a save derived from it
// would fail immediately) but eventCtx stays ctx — deliberately already
// Done, so sendEvent's guard skips writing a stale event down a connection
// that's already being torn down by that same shutdown.
func (h *Handler) finishTurn(dbCtx, eventCtx context.Context, userID string, rec turnRecord, conn *websocket.Conn) {
	if !rec.ok {
		return
	}
	if h.conversationDeletions.generation(rec.conversation) > rec.deletionGen {
		slog.InfoContext(dbCtx, "dropped a turn for a since-deleted conversation", "turn", rec.turn, "conversation", rec.conversation)
		return
	}
	// Blocking, not backgrounded: a two-row insert, not a TMDB round trip
	// — DECISIONS.md's bar for this file ("shipping the simpler design
	// cleanly beats the complex one badly").
	saveCtx, cancel := context.WithTimeout(dbCtx, provisionTimeout)
	created, err := h.saveMessages(saveCtx, userID, rec.conversation, rec.userText, rec.assistantText, rec.titleRefs)
	cancel()
	if err != nil {
		// Logged, not shown to the browser: what the user already saw
		// stays on screen. The next message on this conversation
		// self-heals through saveMessages' own conflict handling either
		// way.
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
		h.sendEvent(eventCtx, conn, rec.turn, outboundEvent{Type: "conversation_created"})
	}
}

// sendEvent writes one curated event down the socket, tagged with turn. It
// checks turnCtx first so a turn that has been cancelled, superseded, or has
// timed out never writes a stale event after the fact — see
// runChatConnection for the full set of reasons turnCtx becomes Done. This
// check is both necessary and sufficient; no shared "is this still current"
// state is needed beyond it. conn == nil is a second, defensive guard: the
// three real call sites always pass a live connection, but finishTurn is
// also called directly (with no connection) from unit tests, where safety
// shouldn't depend on every test fake correctly returning created == false.
func (h *Handler) sendEvent(turnCtx context.Context, conn *websocket.Conn, turn string, ev outboundEvent) {
	if turnCtx.Err() != nil || conn == nil {
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
// runChatConnection) and gates every write via sendEvent; connCtx governs
// only the deferred send's escape hatch below, and must never be used for
// anything else in this function — see that comment for why the
// distinction matters.
func (h *Handler) runTurn(
	turnCtx context.Context,
	connCtx context.Context,
	conn *websocket.Conn,
	userID, turnID, conversationID, text string,
	guestProviders []int,
	history []historyTurn,
	deletionGen int,
	done chan<- turnRecord,
) {
	rec := turnRecord{turn: turnID, conversation: conversationID, deletionGen: deletionGen}
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
	// A guest has no verdicts to load at all.
	var verdicts []Verdict
	if userID != "" && len(chatCtx.Providers) > 0 {
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
			if summary := summarizePicks(ev.Picks, ev.Kind == "lookup"); summary != "" {
				rec.ok = true
				rec.userText = text
				rec.assistantText = summary
				rec.titleRefs = make([]agentTitleRef, len(ev.Picks))
				for i, p := range ev.Picks {
					rec.titleRefs[i] = agentTitleRef{TMDBID: p.TMDBID, MediaType: p.MediaType}
				}
			}
		case "message":
			h.sendEvent(turnCtx, conn, turnID, outboundEvent{Type: "token", Text: ev.Text})
			rec.ok = true
			rec.userText = text
			rec.assistantText = ev.Text
			// Not reachable today (agent/chat.py emits "results" xor
			// "message" per turn) but cheap to keep correct: without this,
			// a "results" event followed by a "message" event in the same
			// stream would leave rec.titleRefs pointing at picks that don't
			// match rec.assistantText's replacement text.
			rec.titleRefs = nil
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

// cachedProviderNames resolves wanted ids to names from the country's cached
// streaming_providers row — the one read both loadChatContext (inside its
// transaction) and loadGuestChatContext (straight off the pool) share, so
// the interface is what pgx.Tx and *pgxpool.Pool have in common. No cached
// row yet (T15 populates it lazily on read) degrades to no names, never an
// error: the interpreting line drops its "on …" clause, the turn goes on.
func cachedProviderNames(ctx context.Context, q interface {
	QueryRow(context.Context, string, ...any) pgx.Row
}, country string, wanted []int) ([]string, error) {
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
