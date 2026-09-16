"""Chat pipeline — turns one message into a stream of NDJSON events.

Bridges catalog_tool.search()'s callback hook (on_intent) into an async
generator via a queue: search() runs as a background task and pushes events
onto the queue as they become available (intent as soon as interpret()
resolves, the final result once discover()/rank() finish); this generator
drains the queue and yields each event to its caller in order. That is what
lets an "intent" event reach the caller in ~1-2s while the full pipeline
(a deep search is several TMDB round trips, plus rank()) is still running.

Event shapes on the wire (one JSON object per line):

    {"type": "intent", "intent": {...DiscoverIntent fields...}}
    {"type": "results", "kind": "search" | "lookup", "relaxed": [...],
        "note": str,
        "picks": [{...Title fields, "genre_names":
        [...], "runtime_minutes": int | None, "cast": [...], "available_on":
        [...Provider fields] | None, "blurb": str}]}
        -- available_on is None only when the availability check itself
           failed, distinct from [] (confirmed available nowhere the caller
           subscribes) — see catalog_tool.py's _safe_availability
        -- kind is which path search() took, recorded where the branch is
           taken rather than inferred from intent fields afterwards; the
           gateway writes it into stored history (chat.go's summarizePicks)
        -- note is the one-line explanation of a widened search, empty
           whenever nothing was relaxed; `relaxed` stays alongside it as the
           machine-readable form
    {"type": "message", "text": str}   -- a plain reply with no candidates
                                           (no subscriptions, capability
                                           question, clarifying question, a
                                           named title that didn't resolve,
                                           or nothing found)
    {"type": "error", "reason": str}   -- one of a fixed set, see _error_reason
    {"type": "done"}

This is the agent-internal protocol, not the browser-facing one. The gateway
owns the public event vocabulary (interpreting/results/token/done/error) and
templates it from these — see chat.go's outboundEvent for the vocabulary the
browser actually sees. Nothing here assumes anything about WebSockets; this
module has no transport opinion beyond "an async stream of dicts."

Cancellation is plain asyncio cancellation, nothing bespoke: the caller
(main.py's StreamingResponse) stops iterating when the client disconnects,
which cancels this generator at its current `await queue.get()`, which the
`finally` block turns into cancelling the background search() task — so an
in-flight TMDB or model call unwinds the same way any cancelled asyncio call
does. See ARCHITECTURE.md's "One socket per session" for why the agent needs no
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
    TitleRef,
    TitleVerdict,
    search,
)
from tmdb import TMDBClient, TMDBUnavailable

logger = logging.getLogger("holster.chat")

# Defensive cap on history length even though the gateway is the one that
# decides the window (ARCHITECTURE.md: "the gateway assembles the context...
# the agent receives what it needs") — a bug upstream should cost a truncated
# prompt, not an unbounded one.
#
# Counted in messages, not exchanges: HistoryTurn is one row. Deliberately well
# clear of services/gateway's windowHistory (maxHistoryExchanges*2 = 10), which
# is the real policy — a backstop sitting at the operating point stops being a
# backstop and silently becomes a second, tighter window.
MAX_HISTORY_TURNS = 20

# The same shape as the backstop above, but not the same job: nothing upstream
# bounds `shown` any more — the gateway sends a conversation's whole
# already-shown set — and this runs long after the body was parsed, so it
# guards no prompt and no request size. What earns its keep is the warning at
# the slice below: the gateway believing it suppresses titles the search will
# not exclude has no other symptom.
#
# That slice keeps the newest refs, so firing this drops the oldest — the
# titles "never re-offer one" is most about. Sized so a real conversation does
# not reach it.
MAX_SHOWN = 2000

NO_PROVIDERS_MESSAGE = (
    "You haven't picked any streaming services yet. Head to Connections to "
    "set them up and I'll be able to find something for you."
)

# Keep the keys in sync with catalog_tool.RelaxedConstraint by hand — .get()
# below falls back to the raw code, so a missed label reads oddly rather than
# raising, the same trade friendlyError() makes against _error_reason().
_RELAXED_LABELS = {
    "runtime": "how long",
    "year": "the year",
    "keywords": "the mood",
    "genres": "the genre",
    "rating": "the rating bar",
}


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
    # table persists the full conversation now, but this agent
    # still never reads it directly; it only ever sees this windowed slice,
    # handed to it fresh on every call. Oldest-first, current message not
    # included here.
    history: list[HistoryTurn] = Field(default_factory=list)
    # The caller's whole title_verdicts set, loaded by the gateway with the
    # message. Passed straight through; catalog_tool.search()
    # decides what each value means.
    verdicts: list[TitleVerdict] = Field(default_factory=list)
    # Everything the gateway has already shown in this conversation, so "show
    # me 10 more" means ten different titles. Unwindowed, unlike `history`
    # above: a title re-offered is a visible bug where an old turn falling out
    # of the prompt is not. This agent holds nothing between turns.
    shown: list[TitleRef] = Field(default_factory=list)


def _with_history(message: str, history: list[HistoryTurn]) -> str:
    """Fold recent turns into the text handed to interpret()/rank(), rather
    than changing catalog_tool.py's message: str contract.

    Leaves "Current message: ..." last, which is why catalog_tool's _with_taste
    prepends its liked-title hint. _with_target_count then appends one line
    after it — the only thing that comes between the current message and the
    candidate list."""
    if not history:
        return message
    lines = [
        f"{'User' if t.role == 'user' else 'Assistant'}: {t.text}" for t in history
    ]
    return (
        "Recent conversation:\n" + "\n".join(lines) + f"\n\nCurrent message: {message}"
    )


def _human_join(items: Sequence[str]) -> str:
    """ "Netflix, Hulu and Max" — mirrors services/gateway/chat.go's
    humanJoin so the same kind of list reads the same way wherever it
    reaches the user (the interpreting line, a capability answer, here)."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _relaxed_labels(relaxed: Sequence[str]) -> str:
    """What the ladder dropped, in the caller's words. The only reader of
    _RELAXED_LABELS, so the two sentences below can't come to describe the
    same relaxation differently."""
    return _human_join([_RELAXED_LABELS.get(r, r) for r in relaxed])


def _title_not_found_message(title: str) -> str:
    """A named title nothing carried — TMDB returned no rows, or none of the
    rows it did return actually bear the name. Distinct from
    _nothing_found_message because the advice differs: there is nothing to
    loosen, and "try loosening what you're looking for" reads as nonsense in
    reply to a name. Quotes the name back so a misheard or misspelled one is
    visible to the user, which is the likeliest cause -- and saying so is
    honest where a card for the nearest real title would assert an answer
    about something they never asked about."""
    return (
        f'I couldn\'t find anything called "{title}". Check the spelling, or '
        "tell me what you're in the mood for instead."
    )


def _nothing_found_message(
    relaxed: Sequence[str], all_judged: bool, all_shown: bool
) -> str:
    """Why nothing came back, in the caller's words. all_judged and all_shown
    both take precedence over relaxed: when every title found was already
    rated, or already put on screen this conversation, the relaxation story is
    beside the point. They are separate cases because the advice differs and
    because telling someone they rated titles they were only shown is false.
    Otherwise the copy names no constraint, deliberately - the ladder may
    already have dropped runtime or year to get this page, so "drop the
    runtime" can be advice to redo what was already done; and this text
    becomes assistant history feeding the next turn's interpret(), the same
    hazard _capability_message documents below."""
    if all_judged:
        return (
            "I found things, but you've rated all of them already. Try "
            "asking for something different."
        )
    if all_shown:
        # "everything I found", not "everything there is": the read can stop at
        # the page ceiling, the clock, or a failed call, and every query is
        # filtered by a vote floor the user never stated (tmdb.py's
        # MIN_VOTE_COUNT), which only the rating sort's rung bends and only
        # back to that same floor. What this search got to is the most it can
        # claim.
        return (
            "I've already shown you everything I found for that. Try "
            "something different and I'll keep looking."
        )
    if not relaxed:
        return "Nothing matched that. Try loosening what you're looking for."
    labels = _relaxed_labels(relaxed)
    return f"Nothing matched, even after loosening {labels}. Try a different request."


def _relaxed_note(relaxed: Sequence[str], exact_matches: int, total: int) -> str:
    """The line above a result set the ladder had to widen to fill.

    Composed here rather than in the gateway so _RELAXED_LABELS isn't
    duplicated across two services and left to drift.

    Without this the widening is invisible and reads as the assistant ignoring
    half the request. It describes what is on screen rather than the catalog —
    "only the first two match exactly" holds whether the narrow query found
    nothing or found things the user had already rated. search() orders picks
    exactness-first, so "the first" is accurate.

    Empty when nothing was relaxed, and also when widening changed nothing
    that reached the screen — a note about a wider search that contributed no
    pick would send the user looking for results that aren't there.

    The zero-exact wording says "nothing left" rather than "nothing matched":
    a query can be widened because the exact matches were all shown on an
    earlier turn, and telling that user nothing matched their request
    contradicts the results they are still scrolled past.
    """
    if not relaxed or exact_matches >= total:
        return ""
    labels = _relaxed_labels(relaxed)
    if exact_matches == 0:
        return f"Nothing left that matches exactly, so I widened past {labels}."
    lead = (
        "Only the first matches"
        if exact_matches == 1
        else f"Only the first {exact_matches} match"
    )
    return f"{lead} exactly what you asked for — I widened past {labels}."


def _capability_message(provider_names: Sequence[str]) -> str:
    """Capability-question answer: names the caller's own services (if
    resolved — provider_names can be empty while the per-country name
    cache warms up) plus example phrasing, never a feature list. No concrete
    numbers in the example: this text becomes conversation history and feeds
    the next turn's interpret() (_with_history, above).
    """
    where = f" on {_human_join(provider_names)}" if provider_names else ""
    return (
        f"I can find something to watch{where} — try something like "
        '"something funny and short" or "a slow-burn thriller with '
        "[actor].\" Tell me what you're in the mood for."
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
    # Logged where the history backstop above is not: that one sits at twice a
    # window the gateway enforces, so only a gateway bug reaches it, while this
    # one has no window behind it at all — see MAX_SHOWN.
    shown = req.shown[-MAX_SHOWN:]
    if len(req.shown) > MAX_SHOWN:
        logger.warning(
            "shown set truncated to %d, dropping the %d oldest",
            MAX_SHOWN,
            len(req.shown) - MAX_SHOWN,
        )
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def on_intent(intent: DiscoverIntent) -> None:
        # is_capability_question/clarifying_question never reach here in
        # practice (search() returns early for both — see catalog_tool.py)
        # but excluded on principle: neither is a search parameter,
        # meaningless in the gateway's template. `title` is not excluded with
        # them — it is what the turn is searching for, and the gateway's
        # interpreting line is templated off it.
        await queue.put(
            {
                "type": "intent",
                "intent": intent.model_dump(
                    exclude_none=True,
                    exclude={"is_capability_question", "clarifying_question"},
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
                shown=shown,
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

        # is_capability_question checked first, deliberately: the two fields
        # are independent (nothing stops the model setting both), and a
        # capability question is the more specific signal when it happens.
        if result.intent is not None and result.intent.is_capability_question:
            await queue.put(
                {
                    "type": "message",
                    "text": _capability_message(req.watch_provider_names),
                }
            )
        elif result.intent is not None and result.intent.clarifying_question:
            await queue.put(
                {"type": "message", "text": result.intent.clarifying_question}
            )
        elif result.picks:
            await queue.put(
                {
                    "type": "results",
                    "kind": result.kind,
                    "relaxed": result.relaxed,
                    "note": _relaxed_note(
                        result.relaxed, result.exact_matches, len(result.picks)
                    ),
                    "picks": result.picks,
                }
            )
        elif result.kind == "lookup" and result.intent is not None:
            # A name that resolved to nothing — never the relaxation copy,
            # which has no constraint to talk about here. Keyed on the path
            # search() actually took, the same signal the results event
            # carries, so the two can never disagree about what a lookup is.
            await queue.put(
                {
                    "type": "message",
                    "text": _title_not_found_message(result.intent.title),
                }
            )
        else:
            await queue.put(
                {
                    "type": "message",
                    "text": _nothing_found_message(
                        result.relaxed, result.all_judged, result.all_shown
                    ),
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
