import Link from "next/link"

import { Button } from "@/components/ui/button"
import { WatchlistView } from "@/components/watchlist/watchlist-view"
import { signInHref } from "@/lib/guest"
import { getSession } from "@/lib/session"

export default async function WatchlistPage() {
  const { userId } = await getSession()

  return (
    <div className="p-6">
      <h1 className="text-lg font-semibold tracking-tight">Watchlist</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Titles you&apos;ve saved, and where to watch them right now.
      </p>
      <div className="mt-6">
        {userId ? (
          <WatchlistView />
        ) : (
          // In place, not a redirect: a nav item that bounces to sign-in
          // reads as a wall, and the point of guest mode is that there
          // isn't one. The page still shows what signing in unlocks.
          <div className="space-y-3">
            <p className="text-sm text-muted-foreground">
              Sign in to save titles from chat and see them here.
            </p>
            <Button render={<Link href={signInHref("/watchlist")} />}>
              Sign in
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}
