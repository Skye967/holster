"use client"

import { createContext, useCallback, useContext, useRef, useState } from "react"

import { type ChatEvent, useChatSocket } from "@/lib/chat-socket"
import { CONVERSATIONS_CHANGED_EVENT } from "@/lib/conversations"

interface ChatListener {
  onEvent: (ev: ChatEvent) => void
}

interface ChatSessionContextValue {
  send: (text: string, conversationId: string) => string
  setListener: (listener: ChatListener) => void
  clearListener: () => void
  // Decided once, server-side, in chat/layout.tsx — the panel reads it from
  // here rather than from Clerk's client state so there is one source.
  guest: boolean
  // True after a dropped guest socket has been replaced: the turns are still
  // on screen but the new socket carries none of them (chat.go's "Guests").
  guestContextLost: boolean
  // The turn id currently in flight on the shared socket, or null — see this
  // file's own comment on why this lives here rather than in ChatPanel.
  activeTurn: string | null
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
  guestProviders,
  children,
}: {
  // null for an account; a guest's ticked services otherwise (lib/guest.ts),
  // as chat/layout.tsx read them from the cookie.
  guestProviders: number[] | null
  children: React.ReactNode
}) {
  const listenerRef = useRef<ChatListener | null>(null)

  // Lives here, not in ChatPanel, because only one turn may ever be in
  // flight on the shared socket (TASKS.md T14: "disable send while
  // streaming so two are never in flight") — a session-wide invariant, not
  // a per-conversation one. ChatPanel is keyed by conversationId and
  // remounts on every switch, so component-local busy state there used to
  // reset on a switch even while a turn dispatched from the *previous*
  // panel was still genuinely running — leaving nothing to ever disable the
  // composer for it, and (once a stale turn's terminal frame stopped being
  // silently dropped, see chat.go's T32 fix) nothing to clear it either.
  // This provider is the one thing that outlives every panel for the life
  // of a chat session (see the module comment above), so it's the only
  // reliable owner.
  //
  // A ref alongside the state, kept in lockstep, for the same reason
  // ChatPanel used to pair activeTurnIdRef with activeTurnId: handleDisconnect
  // is invoked from chat-socket.ts's WebSocket "close" listener, outside
  // React's render cycle, and needs a value it can read synchronously that's
  // never stale.
  const [activeTurn, setActiveTurn] = useState<string | null>(null)
  const activeTurnRef = useRef<string | null>(null)

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
    if (
      (ev.type === "done" || ev.type === "error") &&
      activeTurnRef.current === ev.turn
    ) {
      // A stale/superseded turn's terminal frame now always arrives
      // (chat.go's T32 fix) but carries a different turn id than whatever's
      // active, so the match above is what makes this correctly a no-op
      // for it — only the currently-tracked turn's own terminal frame
      // clears busy.
      activeTurnRef.current = null
      setActiveTurn(null)
    }
    listenerRef.current?.onEvent(ev)
  }, [])

  // Fires when the socket drops out from under an in-flight turn (network
  // blip, proxy idle-kill, server restart) — chat-socket.ts's own
  // "close" listener has no turn id for this, since it isn't about any one
  // turn. Synthesizes an "error" event for whichever turn was active, the
  // same shape ChatPanel used to build itself, routed through the normal
  // onEvent path so ChatPanel's own applyEvent attaches it without needing
  // to track a turn id of its own. Nothing to do when no turn was active:
  // there's no per-turn error to synthesize, and no other listener hook
  // exists for a bare disconnect.
  const handleDisconnect = useCallback((text: string) => {
    const id = activeTurnRef.current
    if (id !== null) {
      activeTurnRef.current = null
      setActiveTurn(null)
      listenerRef.current?.onEvent({ type: "error", turn: id, text })
    }
  }, [])

  const { send: rawSend, guestContextLost } = useChatSocket(
    handleEvent,
    handleDisconnect,
    guestProviders,
  )

  // Wraps chat-socket.ts's send so dispatch and marking the turn active
  // happen together — a caller can't do one without the other.
  const send = useCallback(
    (text: string, conversationId: string): string => {
      const turn = rawSend(text, conversationId)
      activeTurnRef.current = turn
      setActiveTurn(turn)
      return turn
    },
    [rawSend],
  )

  const setListener = useCallback((listener: ChatListener) => {
    listenerRef.current = listener
  }, [])
  const clearListener = useCallback(() => {
    listenerRef.current = null
  }, [])

  return (
    <ChatSessionContext.Provider
      value={{
        send,
        setListener,
        clearListener,
        guest: guestProviders !== null,
        guestContextLost,
        activeTurn,
      }}
    >
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
