"use client"

import { useAuth } from "@clerk/nextjs"
import { useCallback, useEffect, useState } from "react"

import { TitleCard } from "@/components/chat/title-card"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import type { AgentPick } from "@/lib/chat-socket"
import {
  GATEWAY_CALL_TIMEOUT_MS,
  GatewaySessionExpiredError,
  SESSION_EXPIRED_TEXT,
  gatewayFetch,
  withTimeout,
} from "@/lib/gateway"
import {
  VERDICT_LOCKED_TEXT,
  VERDICT_STALE_TEXT,
  VerdictLockedError,
  VerdictStaleError,
  verdictKey,
  verdictUrl,
  type Verdict,
} from "@/lib/verdicts"
import { fetchWatchlist } from "@/lib/watchlist"

interface RowStatus {
  pending: boolean
  error?: string
}

// A saved title whose base TMDB lookup failed or no longer resolves
// (pick.unavailable — services/gateway/chat.go's agentPick.Unavailable).
// title_verdicts is stored data (TASKS.md's cross-cutting "degrade rather
// than fail" rule), so the row still has to appear with a way to clear it,
// not vanish — kept as its own small component rather than teaching
// TitleCard (shared with chat) a degraded-data branch it otherwise has no
// use for.
function UnavailableRow({
  pending,
  error,
  onRemove,
}: {
  pending: boolean
  error?: string
  onRemove: () => void
}) {
  return (
    <div className="flex w-64 shrink-0 flex-col gap-3 rounded-lg border p-4">
      <p className="text-sm text-muted-foreground">
        Can&apos;t load this title right now — try reloading.
      </p>
      <Button variant="outline" size="sm" disabled={pending} onClick={onRemove}>
        Remove
      </Button>
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  )
}

export function WatchlistView() {
  const { getToken, isLoaded } = useAuth()
  const [items, setItems] = useState<AgentPick[] | null>(null)
  const [loadError, setLoadError] = useState(false)
  const [rowStatus, setRowStatus] = useState<Record<string, RowStatus>>({})

  useEffect(() => {
    if (!isLoaded) return
    let cancelled = false
    fetchWatchlist(getToken)
      .then((data) => {
        if (!cancelled) setItems(data)
      })
      .catch(() => {
        if (!cancelled) setLoadError(true)
      })
    return () => {
      cancelled = true
    }
  }, [isLoaded, getToken])

  // Clears a title_verdicts row — the shared action behind both the
  // bookmark's "remove from watchlist" and an UnavailableRow's "Remove".
  // Deliberately non-optimistic, unlike chat-panel.tsx's writeVerdict: there
  // the mutated element is an icon flipping in place with an obvious spot to
  // show a rollback error; here the whole card leaves the grid, which would
  // need index-preserving reinsertion just to show one. Simpler to disable
  // the row while in flight and remove it only once the gateway confirms —
  // one extra round trip's latency, no new failure mode.
  // Takes only (tmdbId, mediaType) — fewer parameters than TitleCard's
  // onClearVerdict type, which TypeScript allows a function value to omit.
  // Every row in this view is want_to_watch (line ~215 below), so the
  // gateway's required ?expect= is hardcoded rather than threaded in: the
  // judgment scope (title-card.tsx's "Clear rating") never renders for a
  // want_to_watch card, so there's never another value this could be.
  const removeItem = useCallback(
    async (tmdbId: number, mediaType: "movie" | "tv") => {
      const key = verdictKey(tmdbId, mediaType)
      setRowStatus((prev) => ({ ...prev, [key]: { pending: true } }))
      try {
        const signal = AbortSignal.timeout(GATEWAY_CALL_TIMEOUT_MS)
        const res = await withTimeout(
          gatewayFetch(
            verdictUrl(mediaType, tmdbId, "want_to_watch"),
            { method: "DELETE", signal },
            getToken,
          ),
          GATEWAY_CALL_TIMEOUT_MS,
        )
        // 409 here means errVerdictStale (verdicts.go): the row is still
        // want_to_watch elsewhere in its lifecycle but no longer matches
        // what this row believed — e.g. judged from another tab between this
        // page's load and this click. Non-optimistic already keeps the row
        // in place until the gateway confirms, so this can't silently drift;
        // it's still worth its own message rather than the generic one.
        if (res.status === 409) throw new VerdictStaleError()
        if (!res.ok) throw new Error(`status ${res.status}`)
        setItems(
          (prev) =>
            prev?.filter((p) => verdictKey(p.tmdb_id, p.media_type) !== key) ??
            prev,
        )
      } catch (err) {
        setRowStatus((prev) => ({
          ...prev,
          [key]: {
            pending: false,
            error:
              err instanceof GatewaySessionExpiredError
                ? SESSION_EXPIRED_TEXT
                : err instanceof VerdictStaleError
                  ? VERDICT_STALE_TEXT
                  : "Couldn't remove that — try again",
          },
        }))
      }
    },
    [getToken],
  )

  // Locks a judgment in place — the same write chat-panel.tsx's dropdown
  // makes, reachable here too since a watchlist row also carries the
  // smaller judgment control. want_to_watch never appears as a target: the
  // bookmark (verdict="want_to_watch" on every row here) only ever clears.
  const setJudgment = useCallback(
    async (tmdbId: number, mediaType: "movie" | "tv", verdict: Verdict) => {
      const key = verdictKey(tmdbId, mediaType)
      setRowStatus((prev) => ({ ...prev, [key]: { pending: true } }))
      try {
        const signal = AbortSignal.timeout(GATEWAY_CALL_TIMEOUT_MS)
        const res = await withTimeout(
          gatewayFetch(
            verdictUrl(mediaType, tmdbId),
            {
              method: "PUT",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ verdict }),
              signal,
            },
            getToken,
          ),
          GATEWAY_CALL_TIMEOUT_MS,
        )
        if (res.status === 409) throw new VerdictLockedError()
        if (!res.ok) throw new Error(`status ${res.status}`)
        // A judgment (e.g. "seen") doesn't clear want_to_watch server-side —
        // T17's verdict is one row, one value — so the row leaves the
        // watchlist the same way a bookmark-clear does.
        setItems(
          (prev) =>
            prev?.filter((p) => verdictKey(p.tmdb_id, p.media_type) !== key) ??
            prev,
        )
      } catch (err) {
        setRowStatus((prev) => ({
          ...prev,
          [key]: {
            pending: false,
            error:
              err instanceof GatewaySessionExpiredError
                ? SESSION_EXPIRED_TEXT
                : err instanceof VerdictLockedError
                  ? VERDICT_LOCKED_TEXT
                  : "Couldn't save that — try again",
          },
        }))
      }
    },
    [getToken],
  )

  if (loadError) {
    return (
      <p className="text-sm text-muted-foreground">
        Can&apos;t reach your watchlist right now — try reloading.
      </p>
    )
  }

  if (!items) {
    return (
      <div className="grid grid-cols-[repeat(auto-fill,minmax(16rem,1fr))] gap-4">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="aspect-2/3 w-64" />
        ))}
      </div>
    )
  }

  if (items.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        Nothing saved yet — bookmark a title from chat to see it here.
      </p>
    )
  }

  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(16rem,1fr))] gap-4">
      {items.map((pick) => {
        const key = verdictKey(pick.tmdb_id, pick.media_type)
        const status = rowStatus[key]
        if (pick.unavailable) {
          return (
            <UnavailableRow
              key={key}
              pending={status?.pending ?? false}
              error={status?.error}
              onRemove={() => removeItem(pick.tmdb_id, pick.media_type)}
            />
          )
        }
        return (
          <TitleCard
            key={key}
            pick={pick}
            verdict="want_to_watch"
            pending={status?.pending}
            error={status?.error}
            onSetVerdict={setJudgment}
            onClearVerdict={removeItem}
          />
        )
      })}
    </div>
  )
}
