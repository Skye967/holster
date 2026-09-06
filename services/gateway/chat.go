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

func windowHistory(history []historyTurn) []historyTurn {
	n := maxHistoryExchanges * 2
	if len(history) <= n {
		return history
	}
	return history[len(history)-n:]
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
// tracks the active turn, so that state needs no mutex. A dedicated
// goroutine does the blocking wsjson.Read loop and hands messages over a
// channel; turn goroutines (runTurn) run concurrently and report back over
// `done` — writes to conn are safe from multiple goroutines (coder/websocket
// handles that internally), but only one goroutine may ever call Read, which
// is why the read loop is separate and singular.
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
					h.finishTurn(context.Background(), ctx, userID, rec, conn)
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
						h.finishTurn(ctx, ctx, userID, rec, conn)
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

				// Loaded fresh on every message rather than cached: a
				// per-connection history cache used to live here, but its
				// invalidation rules were the source of most of this
				// feature's bugs, for a database read cheap enough to just
				// repeat.
				loadCtx, cancel := context.WithTimeout(ctx, provisionTimeout)
				history, err := h.loadConversation(loadCtx, userID, conversationID)
				cancel()
				if err != nil {
					// Logged, not fatal to the turn: loadConversation's own
					// contract already treats "nothing saved yet" as a
					// normal nil result, so a transient read failure
					// degrades to an empty history window instead of
					// blocking the turn — the next message on this
					// conversation simply retries the load.
					slog.ErrorContext(ctx, "conversation load failed", "error", dbError(err))
					history = nil
				}

				turnCtx, cancel := context.WithTimeout(ctx, turnDeadline)
				cancelCurrent = cancel
				currentTurn = msg.Turn
				outstanding++
				go h.runTurn(turnCtx, ctx, conn, userID, msg.Turn, conversationID, msg.Text, history, deletionGen, done)
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
			h.finishTurn(ctx, ctx, userID, rec, conn)
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
	userID, turnID, conversationID, text string,
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
