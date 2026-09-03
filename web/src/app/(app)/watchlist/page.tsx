import { auth } from "@clerk/nextjs/server"

import { WatchlistView } from "@/components/watchlist/watchlist-view"

export default async function WatchlistPage() {
  await auth.protect()

  return (
    <div className="p-6">
      <h1 className="text-lg font-semibold tracking-tight">Watchlist</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Titles you&apos;ve saved, and where to watch them right now.
      </p>
      <div className="mt-6">
        <WatchlistView />
      </div>
    </div>
  )
}
