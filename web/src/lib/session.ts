import "server-only"

import { auth } from "@clerk/nextjs/server"
import { cookies } from "next/headers"

import { readGuestProviders } from "@/lib/guest"

// No session means guest, deliberately — it is not an auth failure to
// recover from. Why: ARCHITECTURE.md's "Sign-in is optional".
//
// The one place a server component decides who it's rendering for: an
// account (userId, getToken) or a guest (guestProviders from the cookie —
// null for an account). Every page passes the pieces it needs straight
// through; nothing client-side re-decides this.
export async function getSession() {
  const { userId, getToken } = await auth()
  const guestProviders = userId ? null : readGuestProviders(await cookies())
  return { userId, getToken, guestProviders }
}

export type Session = Awaited<ReturnType<typeof getSession>>
