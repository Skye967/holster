import { ChatSessionProvider } from "@/components/chat/chat-session-provider"
import { MissingServicesNotice } from "@/components/chat/missing-services-notice"
import { getSession } from "@/lib/session"
import { needsOnboarding } from "@/lib/subscriptions"

// Scoped to /chat and /chat/[id] only — see ChatSessionProvider's own
// comment for why the socket lives here rather than the app-wide layout.
// needsOnboarding is checked here, not in each page, so a caller with no
// streaming services never mounts ChatSessionProvider at all — its
// useChatSocket opens the chat WebSocket unconditionally on mount, and this
// is the one place both /chat and /chat/[id] pass through before either
// page's own logic runs. guestProviders is null for an account, so the
// provider's guest mode is decided by the server, from the session.
//
// Accounts only. A guest with nothing picked goes straight into chat: the
// agent already answers that turn from a template pointing at Connections
// (agent/chat.py's NO_PROVIDERS_MESSAGE), so the nudge arrives in context
// instead of as a wall — and a wall here would strand onboarding's "Skip for
// now", which links to /chat. It would also be the first thing an account
// sees when its session ends, claiming it has no services when it has several.
export default async function ChatLayout({ children }: LayoutProps<"/chat">) {
  const session = await getSession()
  if (session.guestProviders === null && (await needsOnboarding(session))) {
    return <MissingServicesNotice />
  }

  return (
    <ChatSessionProvider guestProviders={session.guestProviders}>
      {children}
    </ChatSessionProvider>
  )
}
