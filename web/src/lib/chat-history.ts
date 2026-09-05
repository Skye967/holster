import {
  GATEWAY_CALL_TIMEOUT_MS,
  gatewayFetch,
  withTimeout,
  type GetToken,
} from "@/lib/gateway"

// Mirrors services/gateway/conversations.go's conversationTurn — field names
// must match its JSON tags exactly.
export interface ConversationTurn {
  user_text: string
  assistant_text: string
}

// One round trip for the caller's stored conversation, so chat-panel.tsx can
// re-render the last exchanges on mount instead of starting blank on a page
// reload (TASKS.md T20). Picks aren't part of this shape — only text is
// persisted for reload today, so a rehydrated turn renders as plain text.
export async function fetchChatHistory(
  getToken: GetToken,
): Promise<ConversationTurn[]> {
  const res = await withTimeout(
    gatewayFetch(
      "/api/chat/history",
      { signal: AbortSignal.timeout(GATEWAY_CALL_TIMEOUT_MS) },
      getToken,
    ),
    GATEWAY_CALL_TIMEOUT_MS,
  )
  if (!res.ok) throw new Error(`status ${res.status}`)
  return res.json()
}
