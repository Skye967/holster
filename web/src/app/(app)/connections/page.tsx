import { auth } from "@clerk/nextjs/server"

import { StreamingPicker } from "@/components/streaming-picker"

export default async function ConnectionsPage() {
  await auth.protect()

  return (
    <div className="p-6">
      <h1 className="text-lg font-semibold tracking-tight">Connections</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Tell us what you subscribe to so recommendations only show what you can
        actually watch.
      </p>
      <div className="mt-6">
        <StreamingPicker />
      </div>
    </div>
  )
}
