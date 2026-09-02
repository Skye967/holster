import { auth } from "@clerk/nextjs/server"
import Link from "next/link"

import { needsOnboarding } from "@/lib/subscriptions"

export default async function ChatPage() {
  const { getToken } = await auth.protect()
  const missingServices = await needsOnboarding(getToken)

  return (
    <div className="p-6">
      <h1 className="text-lg font-semibold tracking-tight">Chat</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        {missingServices ? (
          <>
            You haven&apos;t picked any streaming services yet —{" "}
            <Link
              href="/connections"
              className="text-primary underline underline-offset-4"
            >
              add some in Connections
            </Link>{" "}
            to get useful recommendations here.
          </>
        ) : (
          "The agent isn't wired up yet."
        )}
      </p>
    </div>
  )
}
