import { SESSION_EXPIRED_TEXT } from "@/lib/gateway"

// Shown in place of chat/page.tsx's redirect when GET /api/conversations
// fails — a load failure, not an account with no conversations, which
// redirects to a fresh id like any other.
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
