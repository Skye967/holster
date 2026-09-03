"use client"

import { Send } from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"

import { TitleCard } from "@/components/chat/title-card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  type AgentPick,
  type ChatEvent,
  useChatSocket,
} from "@/lib/chat-socket"

interface Turn {
  id: string
  userText: string
  interpreting?: string
  picks?: AgentPick[]
  tokenText?: string
  error?: string
}

function applyEvent(turn: Turn, ev: ChatEvent): Turn {
  switch (ev.type) {
    case "interpreting":
      return { ...turn, interpreting: ev.text }
    case "results":
      return { ...turn, picks: ev.picks }
    case "token":
      return { ...turn, tokenText: ev.text }
    case "error":
      return { ...turn, error: ev.text }
    case "done":
      return turn
    default:
      // gateway and frontend deploy independently (ARCHITECTURE.md) — a
      // new event type shipped on one side before the other lands here.
      // `satisfies never` re-creates the compile-time exhaustiveness check
      // a plain `default: return turn` would otherwise silently drop.
      console.warn("unrecognized chat event type", ev satisfies never)
      return turn
  }
}

function ChatInput({
  value,
  onChange,
  onSubmit,
  disabled,
  autoFocus,
}: {
  value: string
  onChange: (value: string) => void
  onSubmit: () => void
  disabled: boolean
  autoFocus?: boolean
}) {
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault()
        onSubmit()
      }}
      className="flex gap-2"
    >
      <Input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="What do you feel like watching?"
        disabled={disabled}
        autoFocus={autoFocus}
      />
      <Button
        type="submit"
        size="icon"
        disabled={disabled || !value.trim()}
        aria-label="Send"
      >
        <Send />
      </Button>
    </form>
  )
}

function TurnView({ turn, onRetry }: { turn: Turn; onRetry: () => void }) {
  // No field has arrived yet — "Thinking…" rather than a blank screen.
  // Update alongside applyEvent if Turn gains a new optional field.
  const pending =
    !turn.interpreting &&
    !turn.picks &&
    turn.tokenText === undefined &&
    !turn.error

  return (
    <div className="space-y-3">
      <p className="ml-auto w-fit max-w-[85%] rounded-2xl bg-muted px-3 py-2 text-sm">
        {turn.userText}
      </p>
      {turn.interpreting && (
        <p className="text-xs text-muted-foreground italic">
          {turn.interpreting}
        </p>
      )}
      {pending && <p className="text-xs text-muted-foreground">Thinking…</p>}
      {turn.picks && turn.picks.length > 0 && (
        <div className="-mx-4 flex snap-x gap-3 overflow-x-auto px-4 pb-2">
          {turn.picks.map((pick) => (
            <TitleCard key={pick.tmdb_id} pick={pick} />
          ))}
        </div>
      )}
      {turn.tokenText && <p className="text-sm">{turn.tokenText}</p>}
      {turn.error && (
        <div className="flex items-center gap-2">
          <p className="text-sm text-destructive">{turn.error}</p>
          <Button variant="outline" size="sm" onClick={onRetry}>
            Retry
          </Button>
        </div>
      )}
    </div>
  )
}

export function ChatPanel() {
  const [turns, setTurns] = useState<Turn[]>([])
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null)
  const [inputValue, setInputValue] = useState("")
  const listRef = useRef<HTMLDivElement>(null)
  // Mirrors activeTurnId for handleDisconnect's benefit: that callback is
  // invoked from chat-socket.ts's WebSocket "close" listener, outside
  // React's render cycle, so it needs a value it can read synchronously
  // and that's never stale — a ref, not a closure over state. Every write
  // to activeTurnId has a paired write here.
  const activeTurnIdRef = useRef<string | null>(null)

  const handleEvent = useCallback((ev: ChatEvent) => {
    setTurns((prev) =>
      prev.map((t) => (t.id === ev.turn ? applyEvent(t, ev) : t)),
    )
    if (
      (ev.type === "done" || ev.type === "error") &&
      activeTurnIdRef.current === ev.turn
    ) {
      activeTurnIdRef.current = null
      setActiveTurnId(null)
    }
  }, [])

  // Fires when the socket drops out from under an in-flight turn (network
  // blip, proxy idle-kill, server restart) — none of chat-socket.ts's own
  // events carry a turn id for this, since it isn't about any one turn.
  // Synthesizes an "error" event for whatever turn is currently active, so
  // handleEvent stays the single place that fails a turn and clears
  // activeTurnId.
  const handleDisconnect = useCallback(
    (text: string) => {
      const id = activeTurnIdRef.current
      if (id !== null) {
        handleEvent({ type: "error", turn: id, text })
      }
    },
    [handleEvent],
  )

  const { send } = useChatSocket(handleEvent, handleDisconnect)

  // The only send path — typing a new message and retrying a failed turn
  // both call this with the text to (re)send. A retry is nothing more than
  // sending the same text as a new turn; the gateway already guarantees
  // whatever the failed turn already rendered (e.g. results before a
  // mid-stream drop) stays on screen underneath it.
  const submit = useCallback(
    (text: string) => {
      const trimmed = text.trim()
      if (!trimmed || activeTurnId) return
      const turnId = send(trimmed)
      activeTurnIdRef.current = turnId
      setTurns((prev) => [...prev, { id: turnId, userText: trimmed }])
      setActiveTurnId(turnId)
    },
    [activeTurnId, send],
  )

  const handleSubmit = useCallback(() => {
    submit(inputValue)
    setInputValue("")
  }, [submit, inputValue])

  useEffect(() => {
    listRef.current?.scrollTo({
      top: listRef.current.scrollHeight,
      behavior: "smooth",
    })
  }, [turns])

  const disabled = activeTurnId !== null

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* sr-only: the empty-state and populated layouts below are visually
          self-explanatory (a bare input, then a message thread), but the
          page otherwise has no heading or landmark identifying it. */}
      <h1 className="sr-only">Chat</h1>
      {turns.length === 0 ? (
        <div className="flex min-h-0 flex-1 flex-col items-center justify-center p-6">
          <div className="w-full max-w-md">
            <ChatInput
              value={inputValue}
              onChange={setInputValue}
              onSubmit={handleSubmit}
              disabled={disabled}
              autoFocus
            />
          </div>
        </div>
      ) : (
        <>
          <div ref={listRef} className="min-h-0 flex-1 overflow-y-auto">
            <div className="mx-auto max-w-2xl space-y-6 p-4">
              {turns.map((turn) => (
                <TurnView
                  key={turn.id}
                  turn={turn}
                  onRetry={() => submit(turn.userText)}
                />
              ))}
            </div>
          </div>
          <div className="border-t p-4">
            <ChatInput
              value={inputValue}
              onChange={setInputValue}
              onSubmit={handleSubmit}
              disabled={disabled}
            />
          </div>
        </>
      )}
    </div>
  )
}
