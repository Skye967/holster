// A fetch call to the gateway, hardened against a stale token: Clerk session
// tokens live 60 seconds, so any idle tab's next request can
// hit a 401 that isn't a real error. This catches it, fetches a fresh token,
// and retries once — only a failed retry becomes a visible error.

export class GatewaySessionExpiredError extends Error {}

// Shared by every call site needing a bound on a gateway round trip that may
// include gatewayFetch's own 401-retry (subscriptions.ts, chat-socket.ts).
// Sized against the worst case: subscriptions.ts's /api/subscriptions call
// can block on providersRefreshTimeout (15s, services/gateway/providers.go).
// chat-socket.ts's mintTicket has no such I/O — it just rides along at the
// same bound.
export const GATEWAY_CALL_TIMEOUT_MS = 10000

// The one user-facing message for GatewaySessionExpiredError, shared so it
// can't drift between call sites (streaming-picker.tsx, chat-socket.ts).
export const SESSION_EXPIRED_TEXT = "Your session ended — reload the page"

// Shared fallback for a gatewayFetch call's catch block. Not a general
// error-to-text mapper: a call site with its own typed errors
// (VerdictLockedError, VerdictStaleError) checks those first and passes the
// result in as fallback.
export function gatewayErrorText(err: unknown, fallback: string): string {
  return err instanceof GatewaySessionExpiredError
    ? SESSION_EXPIRED_TEXT
    : fallback
}

// The browser and this container reach the gateway at different addresses. A
// browser resolves NEXT_PUBLIC_GATEWAY_URL on the host; server components and
// route handlers run inside the container, where that URL's localhost is the
// container itself. GATEWAY_SERVICE_URL is the compose-network address, set
// only for the containerised web service — unset (bun run dev on the host),
// both callers share an origin and the public URL is right for each.
//
// Deliberately not NEXT_PUBLIC_: that prefix would inline it into the client
// bundle, the one place it must never appear.
function gatewayBase(): string | undefined {
  return typeof window === "undefined"
    ? (process.env.GATEWAY_SERVICE_URL ?? process.env.NEXT_PUBLIC_GATEWAY_URL)
    : process.env.NEXT_PUBLIC_GATEWAY_URL
}

export type GetToken = (opts: {
  template: string
  skipCache?: boolean
}) => Promise<string | null>

// Bounds an arbitrary promise, not just a fetch(): getToken() (Clerk's SDK)
// has no cancellation of its own, so an AbortSignal passed to gatewayFetch
// only reaches the fetch() half. A caller wanting the fetch torn down too
// passes its own AbortSignal.timeout as gatewayFetch's init.signal.
//
// The .finally is load-bearing, not tidy-up: without it a fast response leaves
// the loser's timer pending for the rest of the window — one per call, and in
// Node that keeps the event loop alive past the request.
export function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout>
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error("gateway call timed out")), ms)
  })
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer))
}

// getToken is null for the gateway's guest surface (/guest/*): no session, so
// no Authorization header and no 401 retry — a call site chooses null from
// the server's own auth() result, never because a token failed to mint.
export async function gatewayFetch(
  path: string,
  init: RequestInit,
  getToken: GetToken | null,
): Promise<Response> {
  const doFetch = (t: string | null) => {
    // Headers, not a plain-object spread: init.headers may legally be a
    // Headers instance or a tuple array, either of which would silently
    // spread into {} or numeric keys instead of real header entries.
    const headers = new Headers(init.headers)
    if (t) headers.set("Authorization", `Bearer ${t}`)
    return fetch(`${gatewayBase()}${path}`, {
      ...init,
      headers,
    })
  }

  if (getToken === null) return doFetch(null)

  const token = await getToken({ template: "gateway" })
  if (!token) throw new GatewaySessionExpiredError()

  let res = await doFetch(token)
  if (res.status === 401) {
    // skipCache: true so this isn't handed back the same about-to-expire
    // token that just failed.
    const fresh = await getToken({ template: "gateway", skipCache: true })
    if (!fresh) throw new GatewaySessionExpiredError()
    res = await doFetch(fresh)
    if (res.status === 401) throw new GatewaySessionExpiredError()
  }
  return res
}
