import type { AgentPick } from "@/lib/chat-socket"
import { gatewayFetch, type GetToken } from "@/lib/gateway"

// No withTimeout/AbortSignal.timeout here, unlike verdicts.ts's fetchVerdicts
// or subscriptions.ts's needsOnboarding — those endpoints are DB-only and
// fast. GET /api/watchlist can legitimately block up to
// services/gateway/watchlist.go's own watchlistTimeout (20s, TMDB-backed),
// so this follows streaming-picker.tsx's pattern for the same reason: a
// client-side bound shorter than the server's own would abort a request
// that was always going to succeed.
export async function fetchWatchlist(getToken: GetToken): Promise<AgentPick[]> {
  const res = await gatewayFetch("/api/watchlist", {}, getToken)
  if (!res.ok) throw new Error(`status ${res.status}`)
  return res.json()
}
