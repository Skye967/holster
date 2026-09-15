import { cookies } from "next/headers"
import { redirect } from "next/navigation"

import { GUEST_PROVIDERS_COOKIE } from "@/lib/guest"

// The counterpart to /signed-in (user-menu.tsx's signOut lands here). Clearing
// the guest cookie is the whole job, and it needs a Route Handler for the same
// reason writing it did: the cookie is httpOnly, so nothing client-side can
// touch it. /signed-in already clears it on every sign-in, so this is the
// belt-and-braces half — it drops picks made and never signed in with, rather
// than letting them outlive the visit by the cookie's 30 days.
//
// Lands on /chat, not /sign-in: guest mode's whole point is that there is no
// wall, and chat/layout.tsx lets a guest with nothing picked straight through.
// Signing out drops you into the product, signed out — not onto the onboarding
// picker, which read as though the account had been wiped.
//
// redirect(), never NextResponse.redirect(new URL("/chat", request.nextUrl)):
// an absolute URL trusts the server's own Host, which this app's self-hosted
// runtime can get wrong, and /chat is same-origin anyway. signed-in/route.ts
// redirects under the same rule.
export async function GET() {
  ;(await cookies()).delete(GUEST_PROVIDERS_COOKIE)
  redirect("/chat")
}
