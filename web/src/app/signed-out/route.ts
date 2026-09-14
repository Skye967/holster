import { cookies } from "next/headers"
import { redirect } from "next/navigation"

import { GUEST_PROVIDERS_COOKIE } from "@/lib/guest"

// The counterpart to /signed-in (user-menu.tsx's signOut lands here). Clearing
// the guest cookie is the whole job, and it needs a Route Handler for the same
// reason writing it did: the cookie is httpOnly, so nothing client-side can
// touch it. /signed-in already clears it on every sign-in, so this is the
// belt-and-braces half: it drops picks made and never signed in with, rather
// than letting them outlive the visit by the cookie's full year.
//
// Lands on /chat, not /sign-in: guest mode's whole point is that there is no
// wall, and chat/layout.tsx now lets a guest with nothing picked straight
// through. Signing out drops you into the product, signed out — not onto the
// onboarding picker, which read as though the account had been wiped.
//
// redirect(), never NextResponse.redirect(new URL("/chat",
// request.nextUrl)) — see signed-in/route.ts's comment; this is the second
// time that exact mistake was caught in review here.
export async function GET() {
  ;(await cookies()).delete(GUEST_PROVIDERS_COOKIE)
  redirect("/chat")
}
