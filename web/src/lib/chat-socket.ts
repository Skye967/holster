// One WebSocket per chat session (DECISIONS.md "One socket per session"),
// opened with a single-use ticket minted from POST /api/chat/ticket —
// browsers can't set an Authorization header on a WS upgrade, so this is
// the only way the connection gets authenticated (services/gateway/chat.go).

import { useAuth } from "@clerk/nextjs"
import { useCallback, useEffect, useRef } from "react"

import {
  GATEWAY_CALL_TIMEOUT_MS,
  GatewaySessionExpiredError,
  SESSION_EXPIRED_TEXT,
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
}

// The gateway's curated event vocabulary (outboundEvent in chat.go) — every
// event carries `turn`, since sendEvent always sets it before writing.
// relaxed is optional, not just possibly-empty: services/gateway/chat.go's
// outboundEvent tags it `json:"relaxed,omitempty"`, so the key is absent
// from the wire entirely whenever nothing was relaxed (the common case) —
// typing it as always-present string[] would be a lie the JSON.parse cast
// below can't catch.
export type ChatEvent =
  | { type: "interpreting"; turn: string; text: string }
  | { type: "results"; turn: string; picks: AgentPick[]; relaxed?: string[] }
  | { type: "token"; turn: string; text: string }
  | { type: "error"; turn: string; text: string }
  | { type: "done"; turn: string }

function wsURL(ticket: string): string {
  const base = process.env.NEXT_PUBLIC_GATEWAY_URL
  if (!base) throw new Error("NEXT_PUBLIC_GATEWAY_URL is not set")
  return `${base.replace(/^http/, "ws")}/ws/chat?ticket=${encodeURIComponent(ticket)}`
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

// Bounds the WebSocket open handshake — same order of magnitude as
// GATEWAY_CALL_TIMEOUT_MS but a distinct concern (no HTTP retry involved),
// so it gets its own constant rather than borrowing that one's semantics.
const SOCKET_OPEN_TIMEOUT_MS = 10000

async function connect(getToken: GetToken): Promise<ConnectResult> {
  let ticket: string
  try {
    ticket = await mintTicket(getToken)
  } catch (err) {
    return {
      error:
        err instanceof GatewaySessionExpiredError
          ? SESSION_EXPIRED_TEXT
          : UNREACHABLE_TEXT,
    }
  }

  const socket = new WebSocket(wsURL(ticket))
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
export function useChatSocket(
  onEvent: (ev: ChatEvent) => void,
  onDisconnect: (text: string) => void,
) {
  const { getToken, isLoaded } = useAuth()
  const socketRef = useRef<WebSocket | null>(null)
  const connectingRef = useRef<Promise<ConnectResult> | null>(null)
  // A plain effect-local `cancelled` flag (streaming-picker.tsx's pattern)
  // doesn't reach here: the async continuation that needs to check it lives
  // inside connect()'s .then(), shared across the mount effect and any
  // later send() call via connectingRef — not owned by one effect run.
  const cancelledRef = useRef(false)
  const onEventRef = useRef(onEvent)
  const onDisconnectRef = useRef(onDisconnect)
  useEffect(() => {
    onEventRef.current = onEvent
    onDisconnectRef.current = onDisconnect
  })

  const ensureSocket = useCallback((): Promise<ConnectResult> => {
    const current = socketRef.current
    if (current && current.readyState === WebSocket.OPEN) {
      return Promise.resolve({ socket: current })
    }
    if (!connectingRef.current) {
      connectingRef.current = connect(getToken).then((result) => {
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
              onDisconnectRef.current(UNREACHABLE_TEXT)
            }
          })
        }
        return result
      })
    }
    return connectingRef.current
  }, [getToken])

  useEffect(() => {
    if (!isLoaded) return
    cancelledRef.current = false
    ensureSocket()
    return () => {
      cancelledRef.current = true
      socketRef.current?.close()
      socketRef.current = null
    }
  }, [isLoaded, ensureSocket])

  const send = useCallback(
    (text: string): string => {
      const turn = crypto.randomUUID()
      ensureSocket().then((result) => {
        if ("error" in result) {
          onEventRef.current({ type: "error", turn, text: result.error })
          return
        }
        result.socket.send(JSON.stringify({ type: "message", turn, text }))
      })
      return turn
    },
    [ensureSocket],
  )

  return { send }
}
