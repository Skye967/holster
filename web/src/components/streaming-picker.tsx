"use client"

import { useAuth } from "@clerk/nextjs"
import { Search } from "lucide-react"
import Image from "next/image"
import { useCallback, useEffect, useRef, useState } from "react"

import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { Switch } from "@/components/ui/switch"
import { useRowStatus } from "@/hooks/use-row-status"
import { gatewayErrorText, gatewayFetch } from "@/lib/gateway"
import { MAX_GUEST_PROVIDERS } from "@/lib/guest"
import { setGuestProvider } from "@/lib/guest-actions"
import { setSubscription } from "@/lib/subscriptions"

const GUEST_CAP_TEXT = `Guests can pick up to ${MAX_GUEST_PROVIDERS} services — sign in to add more`

// setGuestProvider reports a refusal as a value, not a throw, so this carries
// it into the shared catch below where every other write failure is handled.
class GuestWriteError extends Error {
  constructor(readonly reason: string) {
    super(reason)
  }
}

// Always-visible providers, picked by hand rather than TMDB's display_priority
// — that ranking surfaces channel add-ons and split paid tiers (Paramount+ and
// Peacock each list as separate "Essential"/"Premium" entries with no plain
// "Paramount Plus" or "Peacock" of their own). Everything else is reachable
// only through search — the common providers surfaced, the long tail behind
// search.
const DEFAULT_PROVIDER_IDS = [
  8, // Netflix
  15, // Hulu
  9, // Amazon Prime Video
  350, // Apple TV
  2616, // Paramount Plus Essential
  1899, // HBO Max
  337, // Disney Plus
  386, // Peacock Premium
]

interface ProviderEntry {
  provider_id: number
  provider_name: string
  logo_url: string | null
  display_priority: number
  subscribed: boolean
}

// Hand-picked defaults in DEFAULT_PROVIDER_IDS order, plus anything else the
// user has subscribed to via search — otherwise a service added through
// search vanishes the moment the search box is cleared. Split out of the
// component body so the ternary that calls it only pays for this when
// there's no search term.
function defaultProviders(
  providers: ProviderEntry[],
  sorted: ProviderEntry[],
): ProviderEntry[] {
  const byId = new Map(providers.map((p) => [p.provider_id, p]))
  const defaults = DEFAULT_PROVIDER_IDS.map((id) => byId.get(id)).filter(
    (p): p is ProviderEntry => p !== undefined,
  )
  const extras = sorted.filter(
    (p) => p.subscribed && !DEFAULT_PROVIDER_IDS.includes(p.provider_id),
  )
  return [...defaults, ...extras]
}

// guestProviders is null for an account (picks live in
// streaming_subscriptions) and the cookie's ids for a guest (lib/guest.ts),
// as the server read them — the same shape ChatSessionProvider takes.
export function StreamingPicker({
  guestProviders,
}: {
  guestProviders: number[] | null
}) {
  const { getToken, isLoaded } = useAuth()
  const guest = guestProviders !== null
  // Read once, when the catalog lands, to seed each row's switch. A ref rather
  // than the prop directly because setGuestProvider revalidates this layout on
  // every toggle: as a dependency of the load effect below, the fresh prop
  // would refetch the catalog and rebuild the list under an in-flight toggle.
  // Never written — the cookie is the source of truth for a guest's picks, and
  // it is maintained server-side.
  const initialGuestIds = useRef(guestProviders)
  const [providers, setProviders] = useState<ProviderEntry[] | null>(null)
  const [loadError, setLoadError] = useState(false)
  const [search, setSearch] = useState("")
  // Per-row mutation status, keyed by provider_id — pending disables that
  // row's Switch so a second click can't race the first write, error shows
  // an inline message on revert.
  const {
    status: rowStatus,
    setPending,
    setSuccess,
    setFailure,
  } = useRowStatus<number>()

  // A guest needs nothing from Clerk, so a guest's `ready` doesn't wait on
  // isLoaded; a signed-in user's does, since the load needs a token. Derived
  // rather than tested inside the load: isLoaded still flips false->true
  // underneath a guest, and as a dependency that re-runs the load, refetching
  // the catalog and rebuilding the list out from under an in-flight toggle.
  // chat-socket.ts derives its own `ready` the same way, for the same reason.
  const ready = guest || isLoaded

  // Only used for the cap message below, so the rendered rows are close enough
  // — the cookie, not this, is what the cap is actually enforced against.
  const subscribedCount =
    providers?.reduce((n, p) => (p.subscribed ? n + 1 : n), 0) ?? 0

  useEffect(() => {
    if (!ready) return
    let cancelled = false

    async function load() {
      try {
        // /guest/providers carries no subscribed flag — the browser is the
        // only thing that knows a guest's picks, so it merges them here.
        const res = guest
          ? await gatewayFetch("/guest/providers", {}, null)
          : await gatewayFetch("/api/providers", {}, getToken)
        if (!res.ok) throw new Error(`status ${res.status}`)
        const data: ProviderEntry[] = await res.json()
        if (cancelled) return
        setProviders(
          guest
            ? data.map((p) => ({
                ...p,
                subscribed: (initialGuestIds.current ?? []).includes(
                  p.provider_id,
                ),
              }))
            : data,
        )
      } catch {
        if (!cancelled) setLoadError(true)
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [ready, getToken, guest])

  const toggle = useCallback(
    async (providerId: number, subscribed: boolean) => {
      // Refused up front so the switch never flips on only to bounce back.
      // Counted from the rendered rows, which can lag a write still in flight;
      // that only costs a redundant round trip, since setGuestProvider enforces
      // the same cap server-side and its "cap" result lands on the same copy.
      if (guest && subscribed && subscribedCount >= MAX_GUEST_PROVIDERS) {
        setFailure(providerId, GUEST_CAP_TEXT)
        return
      }
      // Starting fresh with {pending: true} also clears any stale error from a
      // previous failed attempt on this row — set before the optimistic update
      // and the first await, so a second click is already disabled by the time
      // React re-renders.
      setPending(providerId)
      // Optimistic: flip immediately, revert with an inline message if the
      // write fails, and never showing a status code while doing it.
      setProviders(
        (prev) =>
          prev?.map((p) =>
            p.provider_id === providerId ? { ...p, subscribed } : p,
          ) ?? prev,
      )

      try {
        if (guest) {
          // A Server Function can fail like a fetch, so it gets the same
          // optimistic-then-revert treatment as the account path. One provider
          // per call: the cookie is read and rewritten server-side, so nothing
          // here holds a set that could go stale against it.
          const result = await setGuestProvider(providerId, subscribed)
          if (!result.ok) throw new GuestWriteError(result.reason)
        } else {
          await setSubscription(getToken, providerId, subscribed)
        }
        setSuccess(providerId)
      } catch (err) {
        setProviders(
          (prev) =>
            prev?.map((p) =>
              p.provider_id === providerId
                ? { ...p, subscribed: !subscribed }
                : p,
            ) ?? prev,
        )
        setFailure(
          providerId,
          err instanceof GuestWriteError && err.reason === "cap"
            ? GUEST_CAP_TEXT
            : gatewayErrorText(err, "Couldn't save that — try again"),
        )
      }
    },
    [getToken, guest, subscribedCount, setPending, setSuccess, setFailure],
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
    : defaultProviders(providers, sorted)

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
        ) : providers.length === 0 ? (
          // Nothing was searched and the catalog itself is empty — the
          // region has no cached catalog and the agent couldn't fetch one.
          // Gated on providers.length, not visible.length, so it can't fire
          // just because none of the 8 defaults matched a loaded catalog
          // ("nothing matched" and "something broke" must never look alike —
          // ARCHITECTURE.md's Failure rules).
          <p className="text-sm text-muted-foreground">
            Can&apos;t reach your streaming services right now — try reloading.
          </p>
        ) : null
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
