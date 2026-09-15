import { redirect } from "next/navigation"

import { ChatPanel } from "@/components/chat/chat-panel"
import { isValidConversationId } from "@/lib/conversations"

// No auth call: this page reads no user data itself — chat/layout.tsx has
// already decided guest vs. account for the socket, and ChatPanel's own
// fetches go through the gateway, which is the authorization boundary.
export default async function ChatConversationPage({
  params,
}: PageProps<"/chat/[id]">) {
  const { id } = await params
  // A malformed route param (hand-edited, or a stale bookmark) must never
  // reach the socket — redirect to a fresh conversation, the same fallback
  // /chat's own index page uses.
  if (!isValidConversationId(id)) redirect(`/chat/${crypto.randomUUID()}`)

  // Keyed by conversationId so switching conversations
  // remounts ChatPanel with fresh state, rather than relying on manual
  // resets when a prop changes — the standard React idiom, and simpler
  // than reproducing what a remount already gives for free.
  return <ChatPanel key={id} conversationId={id} />
}
