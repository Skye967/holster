import Link from "next/link"

// Shown in place of the chat panel for an *account* with zero subscriptions —
// chat/layout.tsx's needsOnboarding check renders this for both the index
// route and every conversation route before either page's own body runs. A
// guest never sees it: it has nothing to reload the picks from, so the nudge
// belongs in the turn itself (agent/chat.py's NO_PROVIDERS_MESSAGE).
export function MissingServicesNotice() {
  return (
    <div className="p-6">
      <h1 className="text-lg font-semibold tracking-tight">Chat</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        You haven&apos;t picked any streaming services yet —{" "}
        <Link
          href="/connections"
          className="text-primary underline underline-offset-4"
        >
          add some in Connections
        </Link>{" "}
        to get useful recommendations here.
      </p>
    </div>
  )
}
