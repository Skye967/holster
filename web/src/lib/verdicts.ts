import {
  GATEWAY_CALL_TIMEOUT_MS,
  gatewayFetch,
  withTimeout,
  type GetToken,
} from "@/lib/gateway"

// Mirrors services/gateway/verdicts.go's Verdict — field names must match its
// JSON tags exactly.
export type Verdict =
  "liked" | "disliked" | "seen" | "not_interested" | "want_to_watch"

export interface VerdictEntry {
  tmdb_id: number
  media_type: "movie" | "tv"
  verdict: Verdict
}

// One verdict map entry per title, not per media type + id pair as two
// separate keys — matches title_verdicts' composite primary key, and lets a
// title repeated across turns (e.g. "show me more" re-showing a result), or
// shown on both chat and the watchlist, share the same entry.
export function verdictKey(tmdbId: number, mediaType: "movie" | "tv"): string {
  return `${mediaType}:${tmdbId}`
}

// The single builder for PUT/DELETE /api/verdicts/{mediaType}/{tmdbId} — a
// DELETE's ?expect is the gateway's required compare-and-delete guard
// (verdicts.go's saveVerdict); omit it for a PUT. One place so chat-panel.tsx
// and watchlist-view.tsx can't drift on the query-string shape.
export function verdictUrl(
  mediaType: "movie" | "tv",
  tmdbId: number,
  expect?: Verdict,
): string {
  return `/api/verdicts/${mediaType}/${tmdbId}${expect ? `?expect=${expect}` : ""}`
}

// Thrown when the gateway 409s a verdict write (verdicts.go's errVerdictLocked
// path) — the judgment was already locked by a prior write, so retrying will
// never succeed and the caller must say so, not invite a retry. Lives here
// rather than in the calling component, the same way GatewaySessionExpiredError
// lives in lib/gateway.ts rather than being redefined per caller.
export class VerdictLockedError extends Error {}

// Shown by both chat-panel.tsx and watchlist-view.tsx on a VerdictLockedError
// — one source of truth so the two never drift.
export const VERDICT_LOCKED_TEXT =
  "That title's already been rated — clear it first to change your answer"

// Thrown when the gateway 409s a DELETE (verdicts.go's errVerdictStale path)
// — the row is still there, but no longer holds the verdict this card's
// ?expect claimed, so the clear didn't happen. Distinct from
// VerdictLockedError (a PUT rejected by the lock): here the caller's local
// state was simply stale, not blocked by a rule, so the message and the
// recovery are different — reload, don't "clear it first" (they just tried).
export class VerdictStaleError extends Error {}

export const VERDICT_STALE_TEXT =
  "That title's rating changed since you last saw it — reload and try again"

// One round trip for the caller's whole verdict set, so chat-panel.tsx can
// hydrate every title card's saved/judged state without a request per card.
export async function fetchVerdicts(
  getToken: GetToken,
): Promise<VerdictEntry[]> {
  const res = await withTimeout(
    gatewayFetch(
      "/api/verdicts",
      { signal: AbortSignal.timeout(GATEWAY_CALL_TIMEOUT_MS) },
      getToken,
    ),
    GATEWAY_CALL_TIMEOUT_MS,
  )
  if (!res.ok) throw new Error(`status ${res.status}`)
  return res.json()
}

