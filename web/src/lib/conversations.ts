import {
  GATEWAY_CALL_TIMEOUT_MS,
  gatewayFetch,
  withTimeout,
  type GetToken,
} from "@/lib/gateway"

// Mirrors services/gateway/conversations.go's conversationSummary — field
// names must match its JSON tags exactly.
export interface ConversationSummary {
  id: string
  title: string
}

// Fired on `window` by chat-session-provider.tsx whenever the gateway
// confirms (a "conversation_created" event on the chat socket — see
// chat.go's finishTurn) that a turn's persist just created a brand-new
// conversation row. app-sidebar.tsx listens for this to refetch and pick up
// the new conversation's just-derived title without the
// user navigating away and back. A plain event, not a shared React context:
// the sidebar and the chat session provider don't otherwise need to talk to
// each other, and this is the one signal between them. Dispatched by
// chat-session-provider.tsx, not any one ChatPanel: the provider is what
// survives the life of the chat session, so this still reaches the sidebar
// even if the ChatPanel that dispatched the turn has since unmounted.
export const CONVERSATIONS_CHANGED_EVENT = "holster:conversations-changed"

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

// A basic shape check on a client-generated conversation id, used by
// /chat/[id]/page.tsx to redirect a malformed route param to
// a fresh conversation before it ever reaches the socket. The gateway
// (chat.go's "message" case, via uuid.Parse) is the actual authority on
// well-formedness — this is defence in depth.
export function isValidConversationId(id: string): boolean {
  return UUID_RE.test(id)
}

// The sidebar's list — the gateway already orders these
// newest first.
export async function fetchConversations(
  getToken: GetToken,
): Promise<ConversationSummary[]> {
  const res = await withTimeout(
    gatewayFetch(
      "/api/conversations",
      { signal: AbortSignal.timeout(GATEWAY_CALL_TIMEOUT_MS) },
      getToken,
    ),
    GATEWAY_CALL_TIMEOUT_MS,
  )
  if (!res.ok) throw new Error(`status ${res.status}`)
  return res.json()
}

export async function deleteConversation(
  getToken: GetToken,
  conversationId: string,
): Promise<void> {
  const res = await withTimeout(
    gatewayFetch(
      `/api/conversations/${conversationId}`,
      {
        method: "DELETE",
        signal: AbortSignal.timeout(GATEWAY_CALL_TIMEOUT_MS),
      },
      getToken,
    ),
    GATEWAY_CALL_TIMEOUT_MS,
  )
  if (!res.ok) throw new Error(`status ${res.status}`)
}
