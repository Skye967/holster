import { clerkMiddleware } from "@clerk/nextjs/server"

// Attaches auth state to requests. Route protection is NOT done here —
// path matching can diverge from how Next routes requests. Each page or
// handler that touches user-scoped data calls auth.protect() itself, or
// branches on the session (lib/session.ts) and touches none of it for a
// guest.
export default clerkMiddleware()

export const config = {
  matcher: [
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    "/(api|trpc)(.*)",
  ],
}
