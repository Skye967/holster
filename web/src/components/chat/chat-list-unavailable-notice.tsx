import { SESSION_EXPIRED_TEXT } from "@/lib/gateway"

// Shown in place of a redirect when GET /api/conversations fails — distinct
// from MissingServicesNotice's "zero conversations, nothing wrong" case.
// chat/page.tsx used to conflate the two, silently minting a fresh orphaned
// conversation id on every load failure.
export function ChatListUnavailableNotice({
  sessionExpired,
}: {
  sessionExpired: boolean
}) {
  return (
    <div className="p-6">
      <h1 className="text-lg font-semibold tracking-tight">Chat</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        {sessionExpired
          ? SESSION_EXPIRED_TEXT
          : "Can't reach your conversations right now — try reloading."}
      </p>
    </div>
  )
}
