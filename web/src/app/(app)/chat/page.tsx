import { auth } from "@clerk/nextjs/server"
import { redirect } from "next/navigation"

import { ChatListUnavailableNotice } from "@/components/chat/chat-list-unavailable-notice"
import { fetchConversations } from "@/lib/conversations"
import { GatewaySessionExpiredError } from "@/lib/gateway"

// The bare /chat route resolves "where was I" from the conversations table
// itself (TASKS.md T15.5's "no new flag, derive it" precedent) rather than
// storing a separate pointer: the most recent conversation if one exists,
// otherwise a fresh id — the sidebar's "New chat" button generates the id
// client-side, so a brand-new account gets the identical URL shape either
// way. needsOnboarding is chat/layout.tsx's job now, not this page's — it
// runs before this page ever does.
export default async function ChatIndexPage() {
  const { getToken } = await auth.protect()

  // A failed list load is not the same as "zero conversations" — silently
  // minting a fresh id on failure would strand a user away from
  // conversations that do exist, with nothing to explain why.
  let conversations
  try {
    conversations = await fetchConversations(getToken)
  } catch (err) {
    return (
      <ChatListUnavailableNotice
        sessionExpired={err instanceof GatewaySessionExpiredError}
      />
    )
  }
  // Known limitation: two racing requests for a zero-conversation account
  // (two tabs, or two quick reloads before the first message) can each mint
  // a different id here, forking that account's history. Accepted for now —
  // narrow window, brand-new accounts only.
  redirect(`/chat/${conversations[0]?.id ?? crypto.randomUUID()}`)
}
