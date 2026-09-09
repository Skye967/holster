import { StreamingPicker } from "@/components/streaming-picker"
import { getSession } from "@/lib/session"

export default async function ConnectionsPage() {
  const { guestProviders } = await getSession()

  return (
    <div className="p-6">
      <h1 className="text-lg font-semibold tracking-tight">Connections</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        Tell us what you subscribe to so recommendations only show what you can
        actually watch.
      </p>
      <div className="mt-6">
        <StreamingPicker guestProviders={guestProviders} />
      </div>
    </div>
  )
}
