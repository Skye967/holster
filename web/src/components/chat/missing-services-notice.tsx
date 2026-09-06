import Link from "next/link"

// Shown in place of the chat panel for a caller with zero subscriptions —
// chat/layout.tsx's needsOnboarding check renders this for both the index
// route and every conversation route before either page's own body runs.
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
