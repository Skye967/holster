// One WebSocket per chat session (ARCHITECTURE.md's "One socket per session"),
// opened with a single-use ticket minted from POST /api/chat/ticket —
// browsers can't set an Authorization header on a WS upgrade, so this is
// the only way the connection gets authenticated (services/gateway/chat.go).

import { useAuth } from "@clerk/nextjs"
import { useCallback, useEffect, useRef, useState } from "react"

import {
  GATEWAY_CALL_TIMEOUT_MS,
  gatewayErrorText,
  gatewayFetch,
  withTimeout,
  type GetToken,
} from "@/lib/gateway"

export interface AgentProvider {
  provider_id: number
  provider_name: string
  logo_url: string | null
}

// Mirrors services/gateway/chat.go's agentPick — forwarded to the browser
// close to verbatim, so field names must match its JSON tags exactly.
export interface AgentPick {
  tmdb_id: number
  media_type: "movie" | "tv"
  title: string
  year: number | null
  overview: string
  poster_url: string | null
  vote_average: number
  vote_count: number
  genre_ids: number[]
  genre_names: string[]
  runtime_minutes: number | null
  cast: string[]
  // null when the agent's own availability check failed — distinct from an
  // empty array (confirmed available on none of the caller's services).
  // Must render differently: null means "unknown," not "not here."
  available_on: AgentProvider[] | null
  blurb: string
  // True only for a watchlist row whose TMDB lookup failed
  // or the id no longer resolves — every other field is then a zero value,
  // not real data. Absent (falsy) on every /chat pick.
  unavailable?: boolean
}

// The gateway's curated event vocabulary (outboundEvent in chat.go) — every
// event carries `turn`, since sendEvent always sets it before writing.
// relaxed is optional, not just possibly-empty: services/gateway/chat.go's
// outboundEvent tags it `json:"relaxed,omitempty"`, so the key is absent
// from the wire entirely whenever nothing was relaxed (the common case) —
// typing it as always-present string[] would be a lie the JSON.parse cast
// below can't catch. conversation_created carries nothing beyond turn: it
// isn't rendered inline in any turn's UI, and chat-session-provider.tsx
// intercepts it — telling the sidebar to refetch — before it ever reaches a
// ChatPanel (see that file and services/gateway/chat.go's finishTurn).
// `reply` is the agent's model-written opener, sent ahead of `interpreting`
// and rendered above it. Separate from `token` because a token renders below
// the cards and a later `message` on the same turn replaces it; the gateway
// also never stores this one, so a reloaded conversation shows no opener.
// `note` is `relaxed` said in a sentence — the agent composes it (chat.py's
// _relaxed_note) and it is what gets rendered; `relaxed` stays alongside as
// the machine-readable form. Both are `omitempty`, so both are optional here.
export type ChatEvent =
  | { type: "reply"; turn: string; text: string }
  | { type: "interpreting"; turn: string; text: string }
  | {
      type: "results"
      turn: string
      picks: AgentPick[]
      relaxed?: string[]
      note?: string
    }
  | { type: "token"; turn: string; text: string }
  | { type: "error"; turn: string; text: string }
  | { type: "done"; turn: string }
  | { type: "conversation_created"; turn: string }

// A ticket opens an account socket; `guest=1` opens a guest one
// (services/gateway/chat.go's "Guests"); the gateway refuses anything else.
function wsURL(query: string): string {
  const base = process.env.NEXT_PUBLIC_GATEWAY_URL
  if (!base) throw new Error("NEXT_PUBLIC_GATEWAY_URL is not set")
  return `${base.replace(/^http/, "ws")}/ws/chat?${query}`
}

async function mintTicket(getToken: GetToken): Promise<string> {
  const res = await withTimeout(
    gatewayFetch(
      "/api/chat/ticket",
      { method: "POST", signal: AbortSignal.timeout(GATEWAY_CALL_TIMEOUT_MS) },
      getToken,
    ),
    GATEWAY_CALL_TIMEOUT_MS,
  )
  if (!res.ok) throw new Error(`status ${res.status}`)
  const data: { ticket: string } = await res.json()
  return data.ticket
}

type ConnectResult = { socket: WebSocket } | { error: string }

const UNREACHABLE_TEXT = "Can't reach the server right now — try again"

// Mirrors services/gateway/chat.go's maxMessageLength — checked here so an
// over-long message never leaves the browser at all, rather than reaching
// the gateway and being rejected round-trip. Counted in code points, not
// UTF-16 length, so a multi-byte character is never split — same reasoning
// as the gateway's own rune count.
export const MAX_MESSAGE_LENGTH = 4000

// Reports whether text has more than max code points, without counting past
// max — the composer has no maxLength, so a pasted string can be arbitrarily
// large and `[...text].length` would walk all of it just to reject it. Drives
// the iterator directly rather than `for...of` because an unused loop binding
// trips this project's lint config.
function exceedsCodePointLength(text: string, max: number): boolean {
  let count = 0
  const iter = text[Symbol.iterator]()
  while (!iter.next().done) {
    if (++count > max) return true
  }
  return false
}

// Bounds the WebSocket open handshake — same order of magnitude as
// GATEWAY_CALL_TIMEOUT_MS but a distinct concern (no HTTP retry involved),
// so it gets its own constant rather than borrowing that one's semantics.
const SOCKET_OPEN_TIMEOUT_MS = 10000

// guest is decided by the server (chat/layout.tsx's getSession()) and passed
// down —
// never inferred here from a failed ticket mint, which must stay a visible
// error for an account rather than a silent downgrade to a guest socket.
async function connect(
  getToken: GetToken,
  guest: boolean,
): Promise<ConnectResult> {
  let query = "guest=1"
  if (!guest) {
    try {
      query = `ticket=${encodeURIComponent(await mintTicket(getToken))}`
    } catch (err) {
      return { error: gatewayErrorText(err, UNREACHABLE_TEXT) }
    }
  }

  let socket: WebSocket
  try {
    socket = new WebSocket(wsURL(query))
  } catch {
    // wsURL() throws when NEXT_PUBLIC_GATEWAY_URL is unset, and the WebSocket
    // constructor can throw synchronously for a malformed URL. connect()'s
    // contract is that it always resolves and never rejects: ensureSocket()
    // has no .catch(), so a rejection would leave connectingRef pointing at a
    // dead promise and wedge every later send() for the session.
    return { error: UNREACHABLE_TEXT }
  }
  const opened = await new Promise<boolean>((resolve) => {
    // Without this, a connection attempt the browser silently black-holes
    // (TCP succeeds, the WS upgrade never completes) leaves this promise —
    // and every future ensureSocket()/send() call chained onto it — hanging
    // forever with no visible error.
    const timer = setTimeout(() => resolve(false), SOCKET_OPEN_TIMEOUT_MS)
    socket.addEventListener(
      "open",
      () => {
        clearTimeout(timer)
        resolve(true)
      },
      { once: true },
    )
    socket.addEventListener(
      "error",
      () => {
        clearTimeout(timer)
        resolve(false)
      },
      { once: true },
    )
  })
  if (!opened) {
    socket.close()
    return { error: UNREACHABLE_TEXT }
  }
  return { socket }
}

// send() returns a turn id immediately (the caller renders the user's own
// message and a pending state right away) and resolves the actual socket
// work in the background — including, if needed, a fresh connect. That
// reconnect-on-next-use is deliberately not a background retry loop: it
// mirrors gateway.ts's own reasoning for an expired auth token ("an idle
// tab's next request can hit a 401 that isn't a real error"), applied to an
// idle tab's socket instead.
//
// guestProviders is null for an account and the guest's ticked services
// otherwise — sent on every message frame, since a guest has no rows for
// the gateway to load (services/gateway/chat.go's inboundMessage).
export function useChatSocket(
  onEvent: (ev: ChatEvent) => void,
  onDisconnect: (text: string) => void,
  guestProviders: number[] | null,
) {
  const { getToken, isLoaded } = useAuth()
  const guest = guestProviders !== null
  const socketRef = useRef<WebSocket | null>(null)
  const connectingRef = useRef<Promise<ConnectResult> | null>(null)
  // A ref, like onEventRef below, so a re-rendered provider with a fresh
  // array doesn't rebuild send() and reconnect the socket.
  const guestProvidersRef = useRef(guestProviders)
  // A plain effect-local `cancelled` flag (streaming-picker.tsx's pattern)
  // doesn't reach here: the async continuation that needs to check it lives
  // inside connect()'s .then(), shared across the mount effect and any
  // later send() call via connectingRef — not owned by one effect run.
  const cancelledRef = useRef(false)
  // Set by the close listener the moment a guest's socket drops, which is
  // when the history is actually gone — not on the next connect, which only
  // happens inside send() and so would land after the follow-up it needs to
  // warn about. Cleared by send(), whose turn starts the new thread.
  const [guestContextLost, setGuestContextLost] = useState(false)
  const onEventRef = useRef(onEvent)
  const onDisconnectRef = useRef(onDisconnect)
  useEffect(() => {
    onEventRef.current = onEvent
    onDisconnectRef.current = onDisconnect
    guestProvidersRef.current = guestProviders
  })

  const ensureSocket = useCallback((): Promise<ConnectResult> => {
    const current = socketRef.current
    if (current && current.readyState === WebSocket.OPEN) {
      return Promise.resolve({ socket: current })
    }
    if (!connectingRef.current) {
      connectingRef.current = connect(getToken, guest).then((result) => {
        connectingRef.current = null
        if (cancelledRef.current) {
          if ("socket" in result) result.socket.close()
          // Not the stale `result`: a caller chained onto this same
          // promise (send()) must see a connection it can't use as a
          // failure, not as a socket that's already closing.
          return { error: UNREACHABLE_TEXT }
        }
        if ("socket" in result) {
          const { socket } = result
          socketRef.current = socket
          socket.addEventListener("message", (e) => {
            try {
              onEventRef.current(JSON.parse(e.data) as ChatEvent)
            } catch {
              // A malformed frame is the gateway's bug, not a reason to
              // wedge the connection — drop it (mirrors the gateway's own
              // tolerance for a bad line from the agent).
            }
          })
          socket.addEventListener("close", () => {
            // True only for an unexpected drop of the still-current socket:
            // our own deliberate close (unmount cleanup) already nulls
            // socketRef synchronously before this event fires, so this
            // branch never runs for that case.
            if (socketRef.current === socket) {
              socketRef.current = null
              // A guest's thread lives only on the socket that carried it, so
              // it is gone as of now — say so before the guest types a
              // follow-up the replacement socket would answer blind. The
              // panel still shows the turns on screen.
              if (guest) setGuestContextLost(true)
              onDisconnectRef.current(UNREACHABLE_TEXT)
            }
          })
        }
        return result
      })
    }
    return connectingRef.current
  }, [getToken, guest])

  // A guest needs nothing from Clerk — waiting on its script here would make
  // "no account needed" depend on that script loading. Derived rather than
  // tested in the effect body: isLoaded still flips false->true underneath a
  // guest, and as a dependency it re-runs the effect, closing the open socket
  // and dialling a second one. That close is our own, so the listener below
  // stays silent and an in-flight turn is never failed.
  const ready = guest || isLoaded

  useEffect(() => {
    if (!ready) return
    cancelledRef.current = false
    ensureSocket()
    return () => {
      cancelledRef.current = true
      socketRef.current?.close()
      socketRef.current = null
    }
  }, [ready, ensureSocket])

  const send = useCallback(
    (text: string, conversationId: string): string => {
      const turn = crypto.randomUUID()
      if (exceedsCodePointLength(text, MAX_MESSAGE_LENGTH)) {
        // Deferred, not fired synchronously: chat-session-provider.tsx marks
        // the turn active *after* this call returns, so an error dispatched
        // before that would clear nothing and leave the composer disabled
        // forever. This matches the "fires after send() returns" timing
        // every other failure path here has, without opening a connection.
        queueMicrotask(() => {
          onEventRef.current({
            type: "error",
            turn,
            text: "That message is too long — try something shorter.",
          })
        })
        return turn
      }
      ensureSocket().then((result) => {
        if ("error" in result) {
          onEventRef.current({ type: "error", turn, text: result.error })
          return
        }
        // Cleared here, not at call time: a reconnect that failed leaves the
        // history just as lost, so the warning must outlive a failed send.
        setGuestContextLost(false)
        // providers only on a guest frame: the gateway rejects it on an
        // account socket as malformed, since an account's services come
        // from its own rows, never the client.
        const providers = guestProvidersRef.current
        result.socket.send(
          JSON.stringify({
            type: "message",
            turn,
            text,
            conversation: conversationId,
            ...(providers !== null && { providers }),
          }),
        )
      })
      return turn
    },
    [ensureSocket],
  )

  return { send, guestContextLost }
}
