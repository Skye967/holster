import { auth } from "@clerk/nextjs/server"

import { ChatSessionProvider } from "@/components/chat/chat-session-provider"
import { MissingServicesNotice } from "@/components/chat/missing-services-notice"
import { needsOnboarding } from "@/lib/subscriptions"

// Scoped to /chat and /chat/[id] only — see ChatSessionProvider's own
// comment for why the socket lives here rather than the app-wide layout.
// needsOnboarding is checked here, not in each page, so a caller with no
// streaming services never mounts ChatSessionProvider at all — its
// useChatSocket opens the chat WebSocket unconditionally on mount, and this
// is the one place both /chat and /chat/[id] pass through before either
// page's own logic runs.
export default async function ChatLayout({ children }: LayoutProps<"/chat">) {
  const { getToken } = await auth.protect()
  if (await needsOnboarding(getToken)) return <MissingServicesNotice />

  return <ChatSessionProvider>{children}</ChatSessionProvider>
}
