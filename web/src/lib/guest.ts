// A guest is a browser with no Clerk session. Its ticked
// streaming services live in this cookie rather than streaming_subscriptions,
// written only through guest-actions.ts so Next's client cache is invalidated
// on every change. No next/headers import here: this module is shared with
// client components, which only need the parser and the sign-in link.

export const GUEST_PROVIDERS_COOKIE = "guest_providers"

// Mirrors services/gateway/chat.go's maxGuestProviders — the gateway rejects
// a longer list as a malformed frame, so never let one reach the wire.
export const MAX_GUEST_PROVIDERS = 50

// Mirrors services/gateway/chat.go's maxProviderID. TMDB's ids are four
// digits; the ceiling only has to be low enough that Number.isInteger's
// acceptance of 1e21 can't put a nonsense id in the cookie and, from there,
// into setSubscription at sign-in.
export const MAX_PROVIDER_ID = 1_000_000

// The one definition of a usable provider id, shared by the parser below and
// guest-actions.ts's write path.
export function isValidProviderId(id: number): boolean {
  return Number.isSafeInteger(id) && id > 0 && id <= MAX_PROVIDER_ID
}

// The only parser for the cookie's value, on both the read and the write
// side: positive integers, deduplicated, capped. Anything else is dropped
// rather than failing — a corrupt cookie degrades to "nothing picked."
export function parseGuestProviders(value: string | undefined): number[] {
  if (!value) return []
  const ids: number[] = []
  for (const part of value.split(",")) {
    const id = Number(part)
    if (isValidProviderId(id) && !ids.includes(id)) ids.push(id)
    if (ids.length === MAX_GUEST_PROVIDERS) break
  }
  return ids
}

// Structural, so the store from next/headers' cookies() fits without this
// module importing it.
export function readGuestProviders(store: {
  get(name: string): { value: string } | undefined
}): number[] {
  return parseGuestProviders(store.get(GUEST_PROVIDERS_COOKIE)?.value)
}

// The counterpart to signInHref: what /signed-in accepts back as `next`.
// Resolved with the same parser that will resolve the redirect, then compared
// by origin — a character-level guard misses what URL parsing strips, so
// "/\t/evil.com" would read as "//evil.com" and leave the site. Anything that
// isn't a same-origin path goes home. No hash: a fragment never reaches the
// server anyway.
export function safeNext(value: string | null, base: URL): string {
  if (!value) return "/"
  try {
    const url = new URL(value, base)
    return url.origin === base.origin ? url.pathname + url.search : "/"
  } catch {
    return "/"
  }
}

// Clerk returns to redirect_url after sign-in (or sign-up via its own link).
// That's always /signed-in (app/signed-in/route.ts), which carries the
// guest's picks into the account and then sends them on to `next` — so a
// guest who taps a signed-in-only control lands back where they were.
export function signInHref(returnTo: string): string {
  const landing = `/signed-in?next=${encodeURIComponent(returnTo)}`
  return `/sign-in?redirect_url=${encodeURIComponent(landing)}`
}
