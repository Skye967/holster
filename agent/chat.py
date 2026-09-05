"""Chat pipeline — turns one message into a stream of NDJSON events.

Bridges catalog_tool.search()'s callback hook (on_intent) into an async
generator via a queue: search() runs as a background task and pushes events
onto the queue as they become available (intent as soon as interpret()
resolves, the final result once discover()/rank() finish); this generator
drains the queue and yields each event to its caller in order. That is what
lets an "intent" event reach the caller in ~1-2s while the full pipeline
(up to four sequential TMDB calls, plus rank()) is still running.

Event shapes on the wire (one JSON object per line):

    {"type": "intent", "intent": {...DiscoverIntent fields...}}
    {"type": "results", "relaxed": [...], "picks": [{...Title fields, "genre_names":
        [...], "runtime_minutes": int | None, "cast": [...], "available_on":
        [...Provider fields] | None, "blurb": str}]}
        -- available_on is None only when the availability check itself
           failed, distinct from [] (confirmed available nowhere the caller
           subscribes) — see catalog_tool.py's _safe_availability
    {"type": "message", "text": str}   -- a plain reply with no candidates
                                           (no subscriptions, or nothing found)
    {"type": "error", "reason": str}   -- one of a fixed set, see _error_reason
    {"type": "done"}

This is the agent-internal protocol, not the browser-facing one. The gateway
owns the public event vocabulary (interpreting/results/token/done/error) and
templates it from these — see ../DECISIONS.md and ../CLAUDE.md on the gateway
deciding what leaves. Nothing here assumes anything about WebSockets; this
module has no transport opinion beyond "an async stream of dicts."

Cancellation is plain asyncio cancellation, nothing bespoke: the caller
(main.py's StreamingResponse) stops iterating when the client disconnects,
which cancels this generator at its current `await queue.get()`, which the
`finally` block turns into cancelling the background search() task — so an
in-flight TMDB or model call unwinds the same way any cancelled asyncio call
does. See ../DECISIONS.md "One socket per session" for why the agent needs no
cancel protocol of its own.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from catalog_tool import (
    CatalogToolError,
    DiscoverIntent,
    Interpreter,
    Ranker,
    TitleVerdict,
    search,
)
from tmdb import TMDBClient, TMDBUnavailable

logger = logging.getLogger("holster.chat")

# Defensive cap on history length even though the gateway is the one that
# decides the window (ARCHITECTURE.md: "the gateway assembles the context...
# the agent receives what it needs") — a bug upstream should cost a truncated
# prompt, not an unbounded one.
MAX_HISTORY_TURNS = 6

NO_PROVIDERS_MESSAGE = (
    "You haven't picked any streaming services yet. Head to Connections to "
    "set them up and I'll be able to find something for you."
)

_RELAXED_LABELS = {"runtime": "how long", "year": "the year", "keywords": "the mood"}


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str


class ChatRequest(BaseModel):
    """The gateway's request shape. watch_region/watch_providers are the
    real user's, loaded by the gateway from the database — the model never
    sees or chooses them (see catalog_tool.py, CLAUDE.md)."""

    message: str
    watch_region: str
    watch_providers: list[int] = Field(default_factory=list)
    # Names for the same ids, resolved by the gateway from its own
    # streaming_providers cache (chat.go's chatContext.ProviderNames) —
    # reused here rather than re-resolved, so a capability-question answer
    # can name the caller's services without a second TMDB lookup.
    watch_provider_names: list[str] = Field(default_factory=list)
    # The last few exchanges only — the gateway's job to window. A messages
    # table persists the full conversation now (TASKS.md T20), but this agent
    # still never reads it directly; it only ever sees this windowed slice,
    # handed to it fresh on every call. Oldest-first, current message not
    # included here.
    history: list[HistoryTurn] = Field(default_factory=list)
    # The caller's whole title_verdicts set, loaded by the gateway with the
    # message (TASKS.md T19). Passed straight through; catalog_tool.search()
    # decides what each value means.
    verdicts: list[TitleVerdict] = Field(default_factory=list)


def _with_history(message: str, history: list[HistoryTurn]) -> str:
    """Fold recent turns into the text handed to interpret()/rank(), rather
    than changing catalog_tool.py's message: str contract. Keeps that
    shipped, tested module untouched for a concern that is still settling.

    One of two folding sites now: catalog_tool's _with_taste prepends the
    liked-title hint to the same string, for rank() only. It prepends
    precisely because this function leaves "Current message: ..." at the
    end, and that has to stay the last thing before the candidate list."""
    if not history:
        return message
    lines = [
        f"{'User' if t.role == 'user' else 'Assistant'}: {t.text}" for t in history
    ]
    return (
        "Recent conversation:\n" + "\n".join(lines) + f"\n\nCurrent message: {message}"
    )


def _human_join(items: Sequence[str]) -> str:
    """"Netflix, Hulu and Max" — mirrors services/gateway/chat.go's
    humanJoin so the same kind of list reads the same way wherever it
    reaches the user (the interpreting line, a capability answer, here)."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _nothing_found_message(relaxed: Sequence[str], all_judged: bool) -> str:
    """Why nothing came back, in the caller's words. all_judged takes
    precedence over relaxed: when every title found was already rated, the
    relaxation story is beside the point. The copy names no constraint,
    deliberately - the ladder may already have dropped runtime or year to get
    this page, so "drop the runtime" can be advice to redo what was already
    done; and this text becomes assistant history feeding the next turn's
    interpret(), the same hazard _capability_message documents below."""
    if all_judged:
        return (
            "I found things, but you've rated all of them already. Try "
            "asking for something different."
        )
    if not relaxed:
        return "Nothing matched that. Try loosening what you're looking for."
    labels = _human_join([_RELAXED_LABELS.get(r, r) for r in relaxed])
    return f"Nothing matched, even after loosening {labels}. Try a different request."


def _capability_message(provider_names: Sequence[str]) -> str:
    """Capability-question answer: names the caller's own services (if
    resolved — provider_names can be empty while the per-country name
    cache warms up) plus example phrasing, never a feature list
    (TASKS.md T16.5). No concrete numbers in the example: this text
    becomes conversation history and feeds the next turn's interpret()
    (_with_history, below).
    """
    where = f" on {_human_join(provider_names)}" if provider_names else ""
    return (
        f"I can find something to watch{where} — try something like "
        '"something funny and short" or "a slow-burn thriller with '
        '[actor]." Tell me what you\'re in the mood for.'
    )


def _error_reason(exc: Exception) -> str:
    """A fixed, closed set of reason codes — never the raw exception text,
    which could carry TMDB/model response detail. Mirrors main.go's
    authFailure(): the gateway (or here, the caller of stream_chat) owns the
    user-facing copy; this only classifies.

    TMDBError that is not TMDBUnavailable (a malformed query) means the
    *caller* passed something invalid — watch_region did not come from the
    model, so this is a bug, not a "try later" case — hence "internal", not
    a TMDB-specific code.

    Keep this set in sync by hand with ../services/gateway/chat.go's
    friendlyError(), which maps each of these three strings to user-facing
    copy — its own default case logs anything it doesn't recognize, so a
    drift here surfaces as a gateway log line, not silent degradation.
    """
    if isinstance(exc, TMDBUnavailable):
        return "tmdb_unavailable"
    if isinstance(exc, CatalogToolError):
        return "model_unavailable"
    return "internal"


async def stream_chat(
    req: ChatRequest,
    *,
    client: TMDBClient,
    interpret_model: Interpreter,
    rank_model: Ranker,
) -> AsyncGenerator[dict[str, Any]]:
    # services/gateway's runTurn skips loading verdicts entirely for a caller
    # with no ticked services, on the strength of this early return.
    if not req.watch_providers:
        yield {"type": "message", "text": NO_PROVIDERS_MESSAGE}
        yield {"type": "done"}
        return

    message = _with_history(req.message, req.history[-MAX_HISTORY_TURNS:])
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def on_intent(intent: DiscoverIntent) -> None:
        # is_capability_question never reaches here in practice (search()
        # returns early for it — catalog_tool.py) but excluded on principle:
        # not a search parameter, meaningless in the gateway's template.
        await queue.put(
            {
                "type": "intent",
                "intent": intent.model_dump(
                    exclude_none=True, exclude={"is_capability_question"}
                ),
            }
        )

    async def run() -> None:
        try:
            result = await search(
                message,
                client=client,
                watch_region=req.watch_region,
                watch_providers=req.watch_providers,
                interpret_model=interpret_model,
                rank_model=rank_model,
                on_intent=on_intent,
                verdicts=req.verdicts,
            )
        except Exception as exc:
            # Not `except BaseException` — a cancelled turn (asyncio.CancelledError)
            # must propagate to the caller's `finally`, not be reported as an error.
            reason = _error_reason(exc)
            if reason == "internal":
                logger.exception("chat pipeline failed")
            else:
                logger.info("chat pipeline failed", extra={"reason": reason})
            await queue.put({"type": "error", "reason": reason})
            await queue.put(None)
            return

        if result.intent is not None and result.intent.is_capability_question:
            await queue.put(
                {
                    "type": "message",
                    "text": _capability_message(req.watch_provider_names),
                }
            )
        elif result.picks:
            await queue.put(
                {
                    "type": "results",
                    "relaxed": result.relaxed,
                    "picks": result.picks,
                }
            )
        else:
            await queue.put(
                {
                    "type": "message",
                    "text": _nothing_found_message(result.relaxed, result.all_judged),
                }
            )
        await queue.put({"type": "done"})
        await queue.put(None)

    task = asyncio.ensure_future(run())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
