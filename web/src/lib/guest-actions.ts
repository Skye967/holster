"use server"

import { revalidatePath } from "next/cache"
import { cookies } from "next/headers"

import {
  GUEST_PROVIDERS_COOKIE,
  MAX_GUEST_PROVIDERS,
  isValidProviderId,
  readGuestProviders,
} from "@/lib/guest"

// "cap" is the only rejection the picker has copy for; "invalid" can only come
// from a hand-rolled POST, and reads as a generic failure there.
export type GuestProviderResult = { ok: true } | { ok: false; reason: string }

// One provider per call, mirroring the account path's PUT/DELETE
// /api/subscriptions/{id}. The cookie is read, changed and written here, so the
// browser never sends a whole list — a list is snapshotted when the call is
// made, and a second toggle's snapshot can carry a pick the first one has since
// failed and rolled back. Next dispatches Server Functions one at a time per
// client, so this read-modify-write cannot lose an update. Next's "Mutating
// data" guide calls that an implementation detail that may change: if it ever
// dispatches in parallel, two toggles racing here silently drop one pick.
//
// A Server Function, not document.cookie: only a server-side cookies().set can
// write an httpOnly cookie, and the revalidatePath below is what drops the
// already-rendered layouts that read it — without it, Next's back/forward
// navigation can still serve a stale guest state after the guest ticked
// something. Every action is a public POST, so the id is validated here rather
// than trusted, and the cap is enforced here rather than in the caller.
export async function setGuestProvider(
  id: unknown,
  subscribed: unknown,
): Promise<GuestProviderResult> {
  const providerId = Number(id)
  if (!isValidProviderId(providerId)) return { ok: false, reason: "invalid" }

  const store = await cookies()
  const picked = new Set(readGuestProviders(store))
  if (subscribed === true) {
    // Only a new id can breach the cap — re-ticking one already in the set is
    // a no-op that must not fail.
    if (!picked.has(providerId) && picked.size >= MAX_GUEST_PROVIDERS) {
      return { ok: false, reason: "cap" }
    }
    picked.add(providerId)
  } else {
    picked.delete(providerId)
  }

  store.set(GUEST_PROVIDERS_COOKIE, [...picked].join(","), {
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    // 30 days, not a year: /signed-in and /signed-out both clear this, so a
    // long life only widens the window for picks to outlive the visit on a
    // shared browser.
    maxAge: 60 * 60 * 24 * 30,
  })
  // Every page under the root layout branches on this cookie (the picker, the
  // chat gate, and "/"'s own redirect), so the whole tree is what goes stale.
  revalidatePath("/", "layout")
  return { ok: true }
}
