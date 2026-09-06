"use client"

import { createContext, useCallback, useContext, useRef } from "react"

import { type ChatEvent, useChatSocket } from "@/lib/chat-socket"
import { CONVERSATIONS_CHANGED_EVENT } from "@/lib/conversations"

interface ChatListener {
  onEvent: (ev: ChatEvent) => void
  onDisconnect: (text: string) => void
}

interface ChatSessionContextValue {
  send: (text: string, conversationId: string) => string
  setListener: (listener: ChatListener) => void
  clearListener: () => void
}

const ChatSessionContext = createContext<ChatSessionContextValue | null>(null)

// Lifts useChatSocket above the per-conversation page (chat/[id]/page.tsx)
// so the one socket (DECISIONS.md "One socket per session") survives
// switching between conversations (TASKS.md T20.5) instead of tearing down
// and reconnecting on every navigation — mintTicket's round trip plus the WS
// handshake (chat-socket.ts's SOCKET_OPEN_TIMEOUT_MS bound) is real cost to
// pay per click. Scoped to chat/layout.tsx, not the app-wide layout, so the
// socket still only opens once the user is actually in chat.
//
// Only one ChatPanel is ever mounted at a time, so a single mutable listener
// slot — not a subscriber list — is enough: the panel for the conversation
// currently on screen registers itself and hands events on to whichever
// turn they belong to.
export function ChatSessionProvider({
  children,
}: {
  children: React.ReactNode
}) {
  const listenerRef = useRef<ChatListener | null>(null)

  // The gateway's own confirmation that this turn's persist just created a
  // new conversation row (chat.go's finishTurn) — handled here, not
  // forwarded to whichever ChatPanel is mounted now, because the panel that
  // dispatched the turn may already be gone (a conversation switch, or an
  // abandoned "New chat"). This provider is the one thing that outlives
  // every panel for the life of a chat session (see the module comment
  // above), so it's the only reliable place for a signal that must survive
  // the dispatching panel's unmount.
  const handleEvent = useCallback((ev: ChatEvent) => {
    if (ev.type === "conversation_created") {
      window.dispatchEvent(new CustomEvent(CONVERSATIONS_CHANGED_EVENT))
      return
    }
    listenerRef.current?.onEvent(ev)
  }, [])
  const handleDisconnect = useCallback((text: string) => {
    listenerRef.current?.onDisconnect(text)
  }, [])

  const { send } = useChatSocket(handleEvent, handleDisconnect)

  const setListener = useCallback((listener: ChatListener) => {
    listenerRef.current = listener
  }, [])
  const clearListener = useCallback(() => {
    listenerRef.current = null
  }, [])

  return (
    <ChatSessionContext.Provider value={{ send, setListener, clearListener }}>
      {children}
    </ChatSessionContext.Provider>
  )
}

export function useChatSession() {
  const ctx = useContext(ChatSessionContext)
  if (!ctx) {
    throw new Error("useChatSession must be used within a ChatSessionProvider")
  }
  return ctx
}
