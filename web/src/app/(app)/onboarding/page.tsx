import Link from "next/link"

import { StreamingPicker } from "@/components/streaming-picker"
import { Button } from "@/components/ui/button"
import { getSession } from "@/lib/session"

export default async function OnboardingPage() {
  // No needsOnboarding() check here on purpose: page.tsx already resolved this
  // to redirect here, so re-checking would both redo that gateway call (a
  // redundant round trip) and, on any transient error, evict a genuinely-new
  // user via needsOnboarding's fail-open false — worse than letting someone who
  // already subscribed and lands here directly (bookmark, back button) just see
  // the picker again.
  const { guestProviders } = await getSession()

  return (
    <div className="p-6">
      <h1 className="text-lg font-semibold tracking-tight">
        What do you subscribe to?
      </h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Pick your streaming services so recommendations only show what you can
        actually watch.
      </p>
      <div className="mt-6">
        <StreamingPicker guestProviders={guestProviders} />
      </div>
      <Button
        render={<Link href="/chat" />}
        variant="link"
        className="mt-4 px-0"
      >
        Skip for now
      </Button>
    </div>
  )
}
