"use client"

import { useAuth } from "@clerk/nextjs"
import { Send } from "lucide-react"
import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { useCallback, useEffect, useRef, useState } from "react"

import { useChatSession } from "@/components/chat/chat-session-provider"
import { TitleCard } from "@/components/chat/title-card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { useRowStatus, type RowStatus } from "@/hooks/use-row-status"
import { fetchChatHistory } from "@/lib/chat-history"
import { type AgentPick, type ChatEvent } from "@/lib/chat-socket"
import {
  GATEWAY_CALL_TIMEOUT_MS,
  gatewayErrorText,
  gatewayFetch,
  withTimeout,
} from "@/lib/gateway"
import { signInHref } from "@/lib/guest"
import {
  VERDICT_LOCKED_TEXT,
  VERDICT_STALE_TEXT,
  VerdictLockedError,
  VerdictStaleError,
  fetchVerdicts,
  verdictKey,
  verdictUrl,
  type Verdict,
} from "@/lib/verdicts"

interface Turn {
  id: string
  userText: string
  // The agent's one-line opener, above the interpreting line. Never
  // rehydrated on reload — the gateway does not store it.
  reply?: string
  interpreting?: string
  picks?: AgentPick[]
  // Why the search was widened, when it was — see the agent's _relaxed_note.
  note?: string
  tokenText?: string
  error?: string
}

function applyEvent(turn: Turn, ev: ChatEvent): Turn {
  switch (ev.type) {
    case "reply":
      return { ...turn, reply: ev.text }
    case "interpreting":
      return { ...turn, interpreting: ev.text }
    case "results":
      return { ...turn, picks: ev.picks, note: ev.note }
    case "token":
      return { ...turn, tokenText: ev.text }
    case "error":
      return { ...turn, error: ev.text }
    case "done":
      return turn
    case "conversation_created":
      // Fully handled upstream in chat-session-provider.tsx — this case
      // exists only so the exhaustiveness check below keeps compiling.
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

function TurnView({
  turn,
  onRetry,
  verdicts,
  verdictStatus,
  onSetVerdict,
  onClearVerdict,
}: {
  turn: Turn
  onRetry: () => void
  verdicts: Record<string, Verdict>
  verdictStatus: Record<string, RowStatus>
  onSetVerdict: (
    tmdbId: number,
    mediaType: "movie" | "tv",
    verdict: Verdict,
  ) => void
  onClearVerdict: (
    tmdbId: number,
    mediaType: "movie" | "tv",
    expectedVerdict: Verdict,
  ) => void
}) {
  // No field has arrived yet — "Thinking…" rather than a blank screen. `note`
  // is left out because it only ever arrives with `picks`; a Turn field that
  // can arrive alone belongs here, added alongside applyEvent.
  const pending =
    !turn.reply &&
    !turn.interpreting &&
    !turn.picks &&
    turn.tokenText === undefined &&
    !turn.error

  return (
    <div className="space-y-3">
      <p className="ml-auto w-fit max-w-[85%] rounded-2xl bg-muted px-3 py-2 text-sm">
        {turn.userText}
      </p>
      {turn.reply && <p className="text-sm">{turn.reply}</p>}
      {turn.interpreting && (
        <p className="text-xs text-muted-foreground italic">
          {turn.interpreting}
        </p>
      )}
      {pending && <p className="text-xs text-muted-foreground">Thinking…</p>}
      {turn.note && (
        <p className="text-xs text-muted-foreground">{turn.note}</p>
      )}
      {turn.picks && turn.picks.length > 0 && (
        <div className="-mx-4 flex snap-x gap-3 overflow-x-auto px-4 pb-2">
          {turn.picks.map((pick) => {
            const key = verdictKey(pick.tmdb_id, pick.media_type)
            return (
              <TitleCard
                key={key}
                pick={pick}
                verdict={verdicts[key]}
                pending={verdictStatus[key]?.pending}
                error={verdictStatus[key]?.error}
                onSetVerdict={onSetVerdict}
                onClearVerdict={onClearVerdict}
              />
            )
          })}
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

export function ChatPanel({ conversationId }: { conversationId: string }) {
  const { getToken, isLoaded } = useAuth()
  const {
    send,
    setListener,
    clearListener,
    guest,
    guestContextLost,
    activeTurn,
  } = useChatSession()
  const router = useRouter()
  const pathname = usePathname()
  const [turns, setTurns] = useState<Turn[]>([])
  const [inputValue, setInputValue] = useState("")
  const listRef = useRef<HTMLDivElement>(null)

  // Every verdict the caller has set, hydrated once so a title's saved/judged
  // state is correct on first render rather than only after it's touched
  // this session — e.g. a title marked in an earlier conversation that
  // resurfaces here. One GET, not one request per card.
  const [verdicts, setVerdicts] = useState<Record<string, Verdict>>({})
  // One entry per title, not per control: the bookmark button and the
  // judgment menu both write the same title_verdicts row, so both must
  // disable on the same pending flag or a double-click across the two
  // controls could race.
  const {
    status: verdictStatus,
    setPending: setVerdictPending,
    setSuccess: setVerdictSuccess,
    setFailure: setVerdictFailure,
  } = useRowStatus<string>()

  // Keys writeVerdict has ever touched this session (set or cleared) — the
  // hydration GET below must never let its (possibly stale-by-the-time-it-
  // resolves) snapshot re-introduce a value the user already changed, in
  // either direction. A plain merge can't express "actively cleared," only
  // "not yet touched," so this tracks touches explicitly.
  const touchedVerdictsRef = useRef<Set<string>>(new Set())

  // Both hydration effects skip a guest: it has no verdicts and nothing
  // persisted to reload, and either fetch would just 401 at the gateway.
  useEffect(() => {
    if (!isLoaded || guest) return
    let cancelled = false
    fetchVerdicts(getToken)
      .then((entries) => {
        if (cancelled) return
        setVerdicts((prev) => {
          const next = { ...prev }
          for (const e of entries) {
            const key = verdictKey(e.tmdb_id, e.media_type)
            if (!touchedVerdictsRef.current.has(key)) next[key] = e.verdict
          }
          return next
        })
      })
      .catch(() => {
        // Best-effort hydration: cards just render unmarked if this fails,
        // same as any other title with no verdict yet.
      })
    return () => {
      cancelled = true
    }
  }, [isLoaded, getToken, guest])

  // Rehydrates conversationId's last exchanges on mount, so a page reload
  // keeps the conversation. A switch to a different
  // conversation is a fresh mount, not a change this
  // effect reacts to — [id]/page.tsx keys ChatPanel by conversationId, so
  // `turns` genuinely starts empty here, which is what the
  // "sent a message before the GET resolves" race below relies on. No
  // picks/interpreting on a rehydrated turn — only text is persisted, so
  // it renders as chat-panel.tsx's plain tokenText case.
  useEffect(() => {
    if (!isLoaded || guest) return
    let cancelled = false
    fetchChatHistory(getToken, conversationId)
      .then((history) => {
        if (cancelled) return
        setTurns((prev) =>
          prev.length === 0
            ? history.map((t) => ({
                id: crypto.randomUUID(),
                userText: t.user_text,
                tokenText: t.assistant_text,
              }))
            : prev,
        )
      })
      .catch(() => {
        // Best-effort hydration: an empty thread on failure is the same
        // experience a brand-new conversation already has.
      })
    return () => {
      cancelled = true
    }
  }, [isLoaded, getToken, conversationId, guest])

  // Shared by setVerdict/clearVerdict: optimistic update, then the write,
  // reverting on failure — streaming-picker.tsx's toggle() pattern. Optimistic
  // UI needs a rollback path (ARCHITECTURE.md's Failure rules).
  //
  // expectedVerdict only matters when verdict is undefined (a DELETE): it's
  // the verdict this card is currently showing, sent as ?expect=<verdict> so
  // the gateway only clears the row when it still matches — a
  // compare-and-delete, the same idiom the PUT path's own conditional upsert
  // already uses. See verdicts.go's saveVerdict for why: a stale client (a
  // second tab that hasn't seen a write made elsewhere) then gets a safe
  // no-op instead of erasing whatever's actually there now.
  const writeVerdict = useCallback(
    async (
      tmdbId: number,
      mediaType: "movie" | "tv",
      verdict: Verdict | undefined,
      expectedVerdict?: Verdict,
    ) => {
      // expectedVerdict is only meaningless for a PUT (verdict set); a DELETE
      // (verdict undefined) always needs it — the gateway's ?expect is
      // required, and building the URL below with it missing would silently
      // send the literal string "undefined" instead of failing loudly here.
      if (verdict === undefined && expectedVerdict === undefined) {
        throw new Error(
          "writeVerdict: expectedVerdict is required to clear a verdict",
        )
      }
      // The card's save/rate icons stay visible for a guest — they're the
      // reason to sign in — and a tap goes to sign-in and back here, rather
      // than a disabled control with a tooltip nobody can hover on a phone.
      if (guest) {
        router.push(signInHref(pathname))
        return
      }
      const key = verdictKey(tmdbId, mediaType)
      touchedVerdictsRef.current.add(key)
      // Captured inside the functional updater rather than read from the
      // `verdicts` closure, so this callback doesn't need `verdicts` in its
      // dependency array — its identity (and everything downstream that
      // takes it as a prop) would otherwise be rebuilt on every verdict
      // change instead of only when getToken changes.
      let previous: Verdict | undefined

      setVerdictPending(key)
      setVerdicts((prev) => {
        previous = prev[key]
        const next = { ...prev }
        if (verdict === undefined) delete next[key]
        else next[key] = verdict
        return next
      })

      try {
        const signal = AbortSignal.timeout(GATEWAY_CALL_TIMEOUT_MS)
        const res = await withTimeout(
          gatewayFetch(
            verdictUrl(
              mediaType,
              tmdbId,
              verdict === undefined ? expectedVerdict : undefined,
            ),
            verdict === undefined
              ? { method: "DELETE", signal }
              : {
                  method: "PUT",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ verdict }),
                  signal,
                },
            getToken,
          ),
          GATEWAY_CALL_TIMEOUT_MS,
        )
        // 409 means the gateway rejected the write — a distinct case from a
        // transient failure, since retrying can never succeed as-is. Two
        // different causes share this status: a PUT hitting the judgment
        // lock (verdicts.go's errVerdictLocked, code "verdict_locked") or a
        // DELETE whose ?expect no longer matches (errVerdictStale, code
        // "verdict_stale"). Read from the body's stable code, not inferred
        // from which request this was — inferring from the HTTP method
        // would silently misattribute the moment either path grows a second
        // 409 cause.
        if (res.status === 409) {
          const body: { code?: string } = await res.json().catch(() => ({}))
          if (body.code === "verdict_stale") throw new VerdictStaleError()
          if (body.code === "verdict_locked") throw new VerdictLockedError()
          throw new Error(`status 409, unrecognized code ${body.code}`)
        }
        if (!res.ok) throw new Error(`status ${res.status}`)
        setVerdictSuccess(key)
      } catch (err) {
        // Abandon the touch on failure, not just the optimistic value: if the
        // mount-time hydration GET is still in flight, its (correct) answer
        // for this key must still be allowed to land once it resolves,
        // rather than being permanently skipped for the rest of the session.
        touchedVerdictsRef.current.delete(key)
        setVerdicts((prev) => {
          const next = { ...prev }
          if (previous === undefined) delete next[key]
          else next[key] = previous
          return next
        })
        setVerdictFailure(
          key,
          gatewayErrorText(
            err,
            err instanceof VerdictLockedError
              ? VERDICT_LOCKED_TEXT
              : err instanceof VerdictStaleError
                ? VERDICT_STALE_TEXT
                : "Couldn't save that — try again",
          ),
        )
      }
    },
    [
      getToken,
      guest,
      router,
      pathname,
      setVerdictPending,
      setVerdictSuccess,
      setVerdictFailure,
    ],
  )

  const handleSetVerdict = useCallback(
    (tmdbId: number, mediaType: "movie" | "tv", verdict: Verdict) =>
      writeVerdict(tmdbId, mediaType, verdict),
    [writeVerdict],
  )
  const handleClearVerdict = useCallback(
    (tmdbId: number, mediaType: "movie" | "tv", expectedVerdict: Verdict) =>
      writeVerdict(tmdbId, mediaType, undefined, expectedVerdict),
    [writeVerdict],
  )

  // Only renders the event into the turn it belongs to; tracking which turn
  // is active is chat-session-provider.tsx's job. A superseded turn's
  // terminal event still lands here and still renders — it just doesn't
  // touch the composer's disabled state.
  const handleEvent = useCallback((ev: ChatEvent) => {
    setTurns((prev) =>
      prev.map((t) => (t.id === ev.turn ? applyEvent(t, ev) : t)),
    )
  }, [])

  // Registers this mounted panel as the shared socket's current listener —
  // ChatSessionProvider forwards every event to whichever ChatPanel last
  // called this, always the one on screen. No onDisconnect: a disconnect
  // mid-turn reaches this panel as an ordinary "error" event on its turn,
  // synthesized by chat-session-provider.tsx, which is what knows which turn
  // was active.
  useEffect(() => {
    setListener({ onEvent: handleEvent })
    return () => clearListener()
  }, [setListener, clearListener, handleEvent])

  // The only send path — typing a new message and retrying a failed turn
  // both call this with the text to (re)send. A retry is nothing more than
  // sending the same text as a new turn; the gateway already guarantees
  // whatever the failed turn already rendered (e.g. results before a
  // mid-stream drop) stays on screen underneath it.
  const submit = useCallback(
    (text: string) => {
      const trimmed = text.trim()
      if (!trimmed || activeTurn) return
      const turnId = send(trimmed, conversationId)
      setTurns((prev) => [...prev, { id: turnId, userText: trimmed }])
    },
    [activeTurn, send, conversationId],
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

  const disabled = activeTurn !== null

  // One line, not a banner or a modal: the only thing a guest gives up in
  // chat is persistence, and this says so where the input is.
  //
  // After a drop it says the sharper version of the same thing. The turns
  // above are still on screen but nothing holds them any more, so a follow-up
  // like "more like the second one" would be answered blind — better to say
  // so than to let the model appear to have forgotten.
  const guestNotice = guest && (
    <p className="mt-2 text-xs text-muted-foreground">
      {guestContextLost ? (
        <>
          Disconnected — I&apos;ve lost track of this chat so far.{" "}
          <Link
            href={signInHref(pathname)}
            className="text-primary underline underline-offset-4"
          >
            Sign in
          </Link>{" "}
          to save future chats.
        </>
      ) : (
        <>
          Chats aren&apos;t saved until you{" "}
          <Link
            href={signInHref(pathname)}
            className="text-primary underline underline-offset-4"
          >
            sign in
          </Link>
          .
        </>
      )}
    </p>
  )

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
            {guestNotice}
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
                  verdicts={verdicts}
                  verdictStatus={verdictStatus}
                  onSetVerdict={handleSetVerdict}
                  onClearVerdict={handleClearVerdict}
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
            {guestNotice}
          </div>
        </>
      )}
    </div>
  )
}
