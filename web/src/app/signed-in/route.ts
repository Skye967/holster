import { auth } from "@clerk/nextjs/server"
import type { NextRequest } from "next/server"

import {
  GUEST_PROVIDERS_COOKIE,
  readGuestProviders,
  safeNext,
} from "@/lib/guest"
import { redirectTo } from "@/lib/redirect"
import { setSubscription } from "@/lib/subscriptions"

// Where every sign-in and sign-up lands (the fallbackRedirectUrl on both
// Clerk components, and what signInHref points at). Carries a guest's ticked
// services into the account (TASKS.md T29) before any page renders, so the
// picker and the chat gate see the rows on their first load — a migration
// beside the page would race it. A Route Handler because it's the one place
// that can both write and clear the cookie. Cookie picks are always added,
// never removed — an account's existing subscriptions are not the guest
// session's to drop.
export async function GET(request: NextRequest) {
  const next = safeNext(
    request.nextUrl.searchParams.get("next"),
    request.nextUrl,
  )
  const { userId, getToken } = await auth()
  const ids = userId ? readGuestProviders(request.cookies) : []
  if (ids.length === 0) {
    // Cleared here too, not only after a migration: reaching this branch with
    // a cookie still set means picks that were never carried over, and the
    // next person to sign in on this device would inherit them. Losing a
    // guest's own picks beats writing them into someone else's account.
    const res = redirectTo(next)
    res.cookies.delete(GUEST_PROVIDERS_COOKIE)
    return res
  }

  // setSubscription carries the module's timeout, so a hung gateway can't
  // strand the user on this callback URL. allSettled rather than all: one
  // rejection must not skip the rest, and whether any failed is what picks
  // the redirect below.
  const results = await Promise.allSettled(
    ids.map((id) => setSubscription(getToken, id, true)),
  )
  const failed = results.filter((r) => r.status === "rejected")
  for (const r of failed) {
    console.error("guest subscription migration failed", r.reason)
  }

  // Cleared whatever happened, which is what TASKS.md T29 specifies: this
  // route only runs at sign-in, so a kept cookie is never read again — it
  // just waits to carry this device's picks into whoever signs in next.
  // A partial failure is made visible instead: /onboarding is the picker,
  // account-backed by now, showing exactly which picks landed so the rest
  // can be re-ticked in one screen rather than silently lost or inherited.
  const res = redirectTo(failed.length > 0 ? "/onboarding" : next)
  res.cookies.delete(GUEST_PROVIDERS_COOKIE)
  return res
}
