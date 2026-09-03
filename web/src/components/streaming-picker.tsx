"use client"

import { useAuth } from "@clerk/nextjs"
import { Search } from "lucide-react"
import Image from "next/image"
import { useCallback, useEffect, useState } from "react"

import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { Switch } from "@/components/ui/switch"
import {
  GatewaySessionExpiredError,
  SESSION_EXPIRED_TEXT,
  gatewayFetch,
} from "@/lib/gateway"

// Always-visible providers, ranked by TMDB's own display_priority — everything
// past this is reachable only through search. TASKS.md T15: "surface the
// common providers and put the long tail behind search."
const TOP_PROVIDER_COUNT = 8

interface ProviderEntry {
  provider_id: number
  provider_name: string
  logo_url: string | null
  display_priority: number
  subscribed: boolean
}

interface RowStatus {
  pending: boolean
  error?: string
}

export function StreamingPicker() {
  const { getToken, isLoaded } = useAuth()
  const [providers, setProviders] = useState<ProviderEntry[] | null>(null)
  const [loadError, setLoadError] = useState(false)
  const [search, setSearch] = useState("")
  // Per-row mutation status, keyed by provider_id — pending disables that
  // row's Switch so a second click can't race the first write, error shows
  // an inline message on revert.
  const [rowStatus, setRowStatus] = useState<Record<number, RowStatus>>({})

  useEffect(() => {
    if (!isLoaded) return
    let cancelled = false

    async function load() {
      try {
        const res = await gatewayFetch("/api/providers", {}, getToken)
        if (!res.ok) throw new Error(`status ${res.status}`)
        const data: ProviderEntry[] = await res.json()
        if (!cancelled) setProviders(data)
      } catch {
        if (!cancelled) setLoadError(true)
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [isLoaded, getToken])

  const toggle = useCallback(
    async (providerId: number, subscribed: boolean) => {
      // Starting fresh with {pending: true} also clears any stale error from
      // a previous failed attempt on this row — set before the optimistic
      // update / first await so a second click is already disabled by the
      // time React re-renders.
      setRowStatus((prev) => ({ ...prev, [providerId]: { pending: true } }))
      // Optimistic: flip immediately, revert with an inline message if the
      // write fails (TASKS.md's cross-cutting rule — never show a status code).
      setProviders(
        (prev) =>
          prev?.map((p) =>
            p.provider_id === providerId ? { ...p, subscribed } : p,
          ) ?? prev,
      )

      try {
        const res = await gatewayFetch(
          `/api/subscriptions/${providerId}`,
          { method: subscribed ? "PUT" : "DELETE" },
          getToken,
        )
        if (!res.ok) throw new Error(`status ${res.status}`)
        setRowStatus((prev) => ({ ...prev, [providerId]: { pending: false } }))
      } catch (err) {
        setProviders(
          (prev) =>
            prev?.map((p) =>
              p.provider_id === providerId
                ? { ...p, subscribed: !subscribed }
                : p,
            ) ?? prev,
        )
        setRowStatus((prev) => ({
          ...prev,
          [providerId]: {
            pending: false,
            error:
              err instanceof GatewaySessionExpiredError
                ? SESSION_EXPIRED_TEXT
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
        Can&apos;t reach your streaming services right now — try reloading.
      </p>
    )
  }

  if (!providers) {
    return (
      <div className="space-y-2">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="h-12 w-full" />
        ))}
      </div>
    )
  }

  const term = search.trim().toLowerCase()
  const sorted = [...providers].sort(
    (a, b) => a.display_priority - b.display_priority,
  )
  const visible = term
    ? sorted.filter((p) => p.provider_name.toLowerCase().includes(term))
    : sorted.slice(0, TOP_PROVIDER_COUNT)

  return (
    <div className="space-y-4">
      <div className="relative">
        <Search className="absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search for a streaming service…"
          className="h-9 pl-8"
        />
      </div>

      {visible.length === 0 ? (
        term ? (
          <p className="text-sm text-muted-foreground">No matches.</p>
        ) : (
          // Nothing was searched — an empty list here means the region has no
          // cached catalog and the agent couldn't fetch one, not "0 results
          // for your search." Must not read as the latter (TASKS.md's
          // cross-cutting rule: "nothing matched" vs "something broke").
          <p className="text-sm text-muted-foreground">
            Can&apos;t reach your streaming services right now — try reloading.
          </p>
        )
      ) : (
        <ul className="divide-y divide-border">
          {visible.map((p) => (
            <li key={p.provider_id} className="flex items-center gap-3 py-3">
              {p.logo_url ? (
                <Image
                  src={p.logo_url}
                  alt=""
                  width={32}
                  height={32}
                  className="size-8 shrink-0 rounded-md object-cover"
                />
              ) : (
                <div className="size-8 shrink-0 rounded-md bg-muted" />
              )}
              <div className="min-w-0 flex-1">
                <span className="block truncate text-sm">
                  {p.provider_name}
                </span>
                {rowStatus[p.provider_id]?.error ? (
                  <span className="block text-xs text-destructive">
                    {rowStatus[p.provider_id]?.error}
                  </span>
                ) : null}
              </div>
              <Switch
                checked={p.subscribed}
                disabled={rowStatus[p.provider_id]?.pending ?? false}
                onCheckedChange={(checked) => toggle(p.provider_id, checked)}
                aria-label={`Subscribed to ${p.provider_name}`}
              />
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
