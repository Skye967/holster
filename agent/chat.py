"""Chat pipeline — turns one message into a stream of NDJSON events.

Runs one turn through holster_catalog's stream_turn and maps what it yields to
the wire: the opener and the intent the moment each arrives, then the outcome.

Event shapes on the wire (one JSON object per line):

    {"type": "intent", "intent": {...SearchArgs fields but reply...}
                                 | {"title": str}}   -- a search | a lookup
    {"type": "results", "kind": "search" | "lookup", "relaxed": [...],
        "note": str,
        "picks": [{...Title fields, "genre_names":
        [...], "runtime_minutes": int | None, "cast": [...], "available_on":
        [...Provider fields] | None, "rent_on": [...Provider fields] | None,
        "watch_link": str | None, "blurb": str}]}
        -- available_on and rent_on are None only when the availability check
           itself failed — see catalog_tool.py's _safe_availability. Empty is
           a different answer on each: available_on [] means the title streams
           on none of the caller's own services, rent_on [] that TMDB names no
           rental store in the region. rent_on is where the title can be paid
           for per view and is not cut to the caller's services, see
           catalog_tool.py's _merge_enrichment. watch_link is the region's
           watch page, None on that same failure or when TMDB gave no usable
           one
        -- kind is which path answered the turn, read from the graph's
           state rather than inferred from intent fields; the gateway
           writes it into stored history (chat.go's summarizePicks)
        -- note is the one-line explanation of a widened search, empty
           whenever nothing was relaxed; `relaxed` stays alongside it as the
           machine-readable form
    {"type": "reply", "text": str}     -- the model's one-line opener,
                                          emitted before "intent" on a turn
                                          that runs a search or a lookup.
                                          Ephemeral: it is written before any
                                          result exists, so it is never stored
                                          as the turn's assistant text (see
                                          gateway/chat.go's runTurn)
    {"type": "message", "text": str}   -- a plain reply with no candidates
                                           (no subscriptions, capability
                                           question, clarifying question, a
                                           chat or off-topic reply, a named
                                           title that didn't resolve, or
                                           nothing found)
    {"type": "error", "reason": str}   -- one of a fixed set, see _error_reason
    {"type": "done"}

This is the agent-internal protocol, not the browser-facing one. The gateway
owns the public event vocabulary (reply/interpreting/results/token/done/error) and
templates it from these — see chat.go's outboundEvent for the vocabulary the
browser actually sees. Nothing here assumes anything about WebSockets; this
module has no transport opinion beyond "an async stream of dicts."

Cancellation is plain asyncio cancellation, nothing bespoke, and it stops the
turn rather than the graph step in flight. When the client disconnects, the
caller (main.py's StreamingResponse) cancels the task this generator is running
in: the response ends at once and no later step starts. A step in flight is not
cancelled, and every step of a turn runs in the graph, finish_search and the
composer included. StreamingResponse's anyio task group redelivers the cancel
until the task ends, and that lands on LangGraph's cleanup before it can cancel
the step, so the step runs to its end under its own timeouts, TMDB calls it has
yet to make included, and its result goes nowhere. contextlib.aclosing is what
closes the turn's stream before a close at a yield here returns, rather than
whenever the stream is finalized. See ARCHITECTURE.md's "One socket per session"
for why the agent needs no cancel protocol of its own.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from holster_catalog import (
    CatalogGraph,
    CatalogToolError,
    Intent,
    LookUpArgs,
    Opener,
    Outcome,
    Results,
    TitleRef,
    TitleVerdict,
    TMDBClient,
    TMDBUnavailable,
    Turn,
    TurnContext,
    TurnEvent,
    stream_turn,
)

logger = logging.getLogger("holster.chat")

# Defensive cap on history length even though the gateway is the one that
# decides the window (ARCHITECTURE.md: "the gateway assembles the context...
# the agent receives what it needs") — a bug upstream should cost a truncated
# prompt, not an unbounded one.
#
# Counted in messages, not exchanges: a Turn is one row. Deliberately well
# clear of gateway's windowHistory (maxHistoryExchanges*2 = 10), which
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

# Hand-written, and the one turn that reaches no model at all. Composing it
# would put the first message most new visitors ever see behind a call on a
# 15-request-a-minute tier, to rephrase advice that is always the same; it
# would also make a frame that gateway/chat.go meters neither by its
# guest turn cap nor by its per-IP limiter — both gated on the frame carrying
# providers — cost a model call.
NO_PROVIDERS_MESSAGE = (
    "You haven't picked any streaming services yet. Head to Connections to "
    "set them up and I'll be able to find something for you."
)


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
    history: list[Turn] = Field(default_factory=list)
    # The caller's whole title_verdicts set, loaded by the gateway with the
    # message. Passed straight through; catalog_tool.find_candidates()
    # decides what each value means.
    verdicts: list[TitleVerdict] = Field(default_factory=list)
    # Everything the gateway has already shown in this conversation, so "show
    # me 10 more" means ten different titles. Unwindowed, unlike `history`
    # above: a title re-offered is a visible bug where an old turn falling out
    # of the prompt is not. This agent holds nothing between turns.
    shown: list[TitleRef] = Field(default_factory=list)


def _error_reason(exc: Exception) -> str:
    """A fixed, closed set of reason codes — never the raw exception text,
    which could carry TMDB/model response detail. Mirrors main.go's
    authFailure(): the gateway (or here, the caller of stream_chat) owns the
    user-facing copy; this only classifies.

    TMDBError that is not TMDBUnavailable (a malformed query) means the
    *caller* passed something invalid — watch_region did not come from the
    model, so this is a bug, not a "try later" case — hence "internal", not
    a TMDB-specific code.

    Keep this set in sync by hand with gateway/chat.go's
    friendlyError(), which maps each of these three strings to user-facing
    copy — its own default case logs anything it doesn't recognize, so a
    drift here surfaces as a gateway log line, not silent degradation.
    """
    if isinstance(exc, TMDBUnavailable):
        return "tmdb_unavailable"
    if isinstance(exc, CatalogToolError):
        return "model_unavailable"
    return "internal"


def _wire(event: TurnEvent | Outcome) -> dict[str, Any]:
    """The NDJSON dict for one thing a turn yielded, built by hand because the
    wire is not the type: a Results' exact_matches stays here, and an intent
    loses its reply."""
    if isinstance(event, Opener):
        return {"type": "reply", "text": event.text}
    if isinstance(event, Intent):
        args = event.args
        # A lookup's is title alone, which gateway/chat.go's interpretingLine
        # templates off. A search's is SearchArgs by JSON name, which its
        # agentIntent mirrors; reply leaves it because it has its own event.
        if isinstance(args, LookUpArgs):
            return {"type": "intent", "intent": {"title": args.title}}
        intent = args.model_dump(exclude={"reply"}, exclude_none=True)
        return {"type": "intent", "intent": intent}
    if isinstance(event, Results):
        return {
            "type": "results",
            "kind": event.kind,
            "relaxed": event.relaxed,
            "note": event.note,
            "picks": event.picks,
        }
    return {"type": "message", "text": event.text}


async def stream_chat(
    req: ChatRequest,
    *,
    client: TMDBClient,
    agent: CatalogGraph,
    correlation_id: str | None,
) -> AsyncGenerator[dict[str, Any]]:
    # gateway's runTurn skips loading verdicts entirely for a caller
    # with no ticked services, on the strength of this early return — and its
    # guest throttle meters only frames that carry providers, on the strength
    # of this turn reaching no model.
    if not req.watch_providers:
        yield {"type": "message", "text": NO_PROVIDERS_MESSAGE}
        yield {"type": "done"}
        return

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
    context = TurnContext(
        client=client,
        watch_region=req.watch_region,
        watch_providers=req.watch_providers,
        watch_provider_names=req.watch_provider_names,
        verdicts=req.verdicts,
        shown=shown,
    )
    # The turn's run carries the request's correlation id, the one this
    # service's log lines for it carry, and every run inside it inherits that.
    # LangSmith receives these runs only when tracing is switched on; see
    # README.md.
    config = RunnableConfig(
        run_name="chat_turn", metadata={"correlation_id": correlation_id}
    )

    try:
        stream = stream_turn(
            agent,
            message=req.message,
            history=req.history[-MAX_HISTORY_TURNS:],
            context=context,
            config=config,
        )
        async with contextlib.aclosing(stream):
            async for event in stream:
                yield _wire(event)
        yield {"type": "done"}
    except Exception as exc:
        # Not `except BaseException` — a cancelled turn (asyncio.CancelledError)
        # and a closed stream (GeneratorExit) must propagate, not be reported
        # as an error.
        reason = _error_reason(exc)
        if reason == "internal":
            logger.exception("chat pipeline failed")
        else:
            logger.info("chat pipeline failed", extra={"reason": reason})
        yield {"type": "error", "reason": reason}
