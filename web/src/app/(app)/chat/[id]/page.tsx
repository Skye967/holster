import { auth } from "@clerk/nextjs/server"
import { redirect } from "next/navigation"

import { ChatPanel } from "@/components/chat/chat-panel"
import { isValidConversationId } from "@/lib/conversations"

export default async function ChatConversationPage({
  params,
}: PageProps<"/chat/[id]">) {
  const { id } = await params
  await auth.protect()
  // A malformed route param (hand-edited, or a stale bookmark) must never
  // reach the socket — redirect to a fresh conversation, the same fallback
  // /chat's own index page uses. needsOnboarding is chat/layout.tsx's job
  // now, not this page's — it runs before this page ever does.
  if (!isValidConversationId(id)) redirect(`/chat/${crypto.randomUUID()}`)

  // Keyed by conversationId so switching conversations (TASKS.md T20.5)
  // remounts ChatPanel with fresh state, rather than relying on manual
  // resets when a prop changes — the standard React idiom, and simpler
  // than reproducing what a remount already gives for free.
  return <ChatPanel key={id} conversationId={id} />
}
