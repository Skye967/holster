"""Chat pipeline — turns one message into a stream of NDJSON events.

Runs catalog_tool's agent and passes on what its tools write as they start
(the opener and the intent, streamed as LangChain "custom" events) the moment
each arrives, then answers from the agent's final state. That is what lets an
"intent" event reach the caller in ~1-2s while the full turn (a deep search is
several TMDB round trips, plus the rank call) is still running.

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
        -- kind is which tool answered the turn, read from the agent's
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

Cancellation is plain asyncio cancellation, nothing bespoke: the caller
(main.py's StreamingResponse) closes this generator when the client
disconnects, and closing it closes the agent's stream, which cancels the step
in flight — so an in-flight TMDB or model call unwinds the same way any
cancelled asyncio call does. contextlib.aclosing is what makes that happen
before the close returns rather than whenever the stream is finalized. See
ARCHITECTURE.md's "One socket per session" for why the agent needs no cancel
protocol of its own.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, Sequence
from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from catalog_tool import (
    CatalogAgent,
    CatalogToolError,
    Composer,
    OutcomeCause,
    TitleRef,
    TitleVerdict,
    TurnContext,
    TurnState,
    compose,
    finish_search,
)
from tmdb import TMDBClient, TMDBUnavailable

logger = logging.getLogger("holster.chat")

# Defensive cap on history length even though the gateway is the one that
# decides the window (ARCHITECTURE.md: "the gateway assembles the context...
# the agent receives what it needs") — a bug upstream should cost a truncated
# prompt, not an unbounded one.
#
# Counted in messages, not exchanges: HistoryTurn is one row. Deliberately well
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

# What the outcome composer gets before the turn answers from its template.
# The SDK's own envelope is no bound worth resting on: timeout=15.0 (main.py)
# is three times this cap on its own.
#
# It bounds what this call adds, not what the turn has already spent — an empty
# turn arrives here having run page 1 and the whole ladder — so a turn already
# near the gateway's 30s turnDeadline still passes it, and the dropped-stream
# fallback is what the user sees.
COMPOSE_TIMEOUT_SECONDS = 5.0

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
    # message. Passed straight through; catalog_tool.find_candidates()
    # decides what each value means.
    verdicts: list[TitleVerdict] = Field(default_factory=list)
    # Everything the gateway has already shown in this conversation, so "show
    # me 10 more" means ten different titles. Unwindowed, unlike `history`
    # above: a title re-offered is a visible bug where an old turn falling out
    # of the prompt is not. This agent holds nothing between turns.
    shown: list[TitleRef] = Field(default_factory=list)


def _turn_messages(message: str, history: list[HistoryTurn]) -> list[AnyMessage]:
    """The agent's input: each recent turn as its own message, oldest first,
    then the current message last. TwoCallTurn reads the last HumanMessage as
    the current one and hands both calls the turns before it."""
    turns: list[AnyMessage] = [
        HumanMessage(t.text) if t.role == "user" else AIMessage(t.text) for t in history
    ]
    return [*turns, HumanMessage(message)]


def _human_join(items: Sequence[str]) -> str:
    """ "Netflix, Hulu and Max" — mirrors gateway/chat.go's
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
    """The fallback when the outcome composer cannot write this one — see
    _outcome_text.

    A named title nothing carried — TMDB returned no rows, or none of the
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
    cause: Literal["nothing_matched", "all_shown", "all_judged"],
    relaxed: Sequence[str],
) -> str:
    """The fallback when the outcome composer cannot write this one — see
    _outcome_text.

    Why nothing came back, in the caller's words. Keyed on the cause
    finish_search() already decided rather than on the flags behind it, so the
    sentence the user reads and the duty the composer was given cannot describe
    the turn differently.

    On nothing_matched the copy names no constraint, deliberately: the ladder
    may already have dropped runtime or year to get this page, so "drop the
    runtime" can be advice to redo what was already done; and this text becomes
    assistant history feeding the next turn's interpret call, the same hazard
    _capability_message documents below."""
    if cause == "all_judged":
        return (
            "I found things, but you've rated all of them already. Try "
            "asking for something different."
        )
    if cause == "all_shown":
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
    nothing or found things the user had already rated. finish_search() orders
    picks exactness-first, so "the first" is accurate.

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
    """The fallback for an answer turn whose model-written text came back
    blank. Keeping it is what holds the guarantee gateway/chat.go's
    runTurn relies on for "message": that its text is never empty,
    and so can be stored as the turn's answer unguarded.

    Names the caller's own services (if
    resolved — provider_names can be empty while the per-country name
    cache warms up) plus example phrasing, never a feature list. No concrete
    numbers in the example: this text becomes conversation history and feeds
    the next turn's interpret call (_turn_messages, above).
    """
    where = f" on {_human_join(provider_names)}" if provider_names else ""
    return (
        f"I can find something to watch{where} — try something like "
        '"something funny and short" or "a slow-burn thriller with '
        "[actor].\" Tell me what you're in the mood for."
    )


async def _outcome_text(
    message: str,
    cause: OutcomeCause,
    *,
    model: Composer,
    config: RunnableConfig,
    title: str,
    relaxed: Sequence[str],
) -> str:
    """The model's own sentence for a turn that ended with nothing to show,
    bounded by COMPOSE_TIMEOUT_SECONDS and degrading to the template behind it.

    The detail the composer is given and the template it falls back to are
    chosen here, from the one cause, so the sentence a user reads cannot
    describe a different turn from the duty the model was set. Only
    nothing_matched has a detail the prompt defines beyond the title: the other
    two are about what the user has already seen or rated, where naming what
    the ladder loosened would merge two answers.

    Degrade, don't fail, the same split catalog_tool's _safe() makes: the turn
    already has its answer and only the wording is at stake.

    Warning, not exception: compose() flattens every model failure into
    CatalogToolError, so a free-tier refusal and a schema fault arrive here
    identically and this function has no signal to tell them apart. The
    traceback is kept because a schema or wiring fault fails every call and
    would otherwise show up only as prose that never got warmer. The SDK's own
    timeout is in that arm too, compose() having wrapped it; the deadline this
    function owns is the other one.

    A blank result falls back too: ComposedReply.text is required of the model,
    which still leaves "" satisfying the schema, and an empty "message" event
    would persist an empty assistant turn.
    """
    if cause == "title_not_found":
        fallback, detail = _title_not_found_message(title), title
    else:
        fallback = _nothing_found_message(cause, relaxed)
        detail = _relaxed_labels(relaxed) if cause == "nothing_matched" else ""
    try:
        async with asyncio.timeout(COMPOSE_TIMEOUT_SECONDS):
            composed = await compose(message, cause, detail, model=model, config=config)
    except TimeoutError:
        logger.info("outcome composer timed out, degraded", extra={"cause": cause})
        return fallback
    except CatalogToolError:
        logger.warning(
            "outcome composer failed, degraded", exc_info=True, extra={"cause": cause}
        )
        return fallback
    return composed.text.strip() or fallback


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


async def stream_chat(
    req: ChatRequest,
    *,
    client: TMDBClient,
    agent: CatalogAgent,
    compose_model: Composer,
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

    messages = _turn_messages(req.message, req.history[-MAX_HISTORY_TURNS:])
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
    # One config for every run the turn starts — the agent's and the outcome
    # composer's — so both carry the request's correlation id, the one this
    # service's log lines for it carry. LangSmith receives these runs only when
    # tracing is switched on; see README.md.
    config = RunnableConfig(
        run_name="chat_turn", metadata={"correlation_id": correlation_id}
    )

    try:
        # "custom" is what the tools write as they start — the opener and the
        # intent — passed through as they arrive. "values" is the whole state
        # after each step; the last one is the turn's result.
        state: TurnState | None = None
        # astream is an async generator, typed as the iterator it returns.
        stream = cast(
            AsyncGenerator[tuple[str, Any]],
            agent.astream(
                {"messages": [*messages]},
                config=config,
                context=context,
                stream_mode=["custom", "values"],
            ),
        )
        async with contextlib.aclosing(stream):
            async for mode, chunk in stream:
                if mode == "custom":
                    yield cast(dict[str, Any], chunk)
                else:
                    state = cast(TurnState, chunk)

        # Every empty-result branch below composes its sentence with a model
        # call, and every one of them has a call to spare: a lookup made one,
        # and a search whose pick list is empty made no rank call — an empty
        # pick list means an empty candidate list, since finish_search()'s
        # top-up otherwise fills the list to `floor`, which
        # min(RESULT_FLOOR, limit) and SearchArgs.limit's `ge=1` hold at one
        # or more, and TwoCallTurn skips the rank call on no candidates.
        if state is not None and "answer_kind" in state:
            # The model's own answer, written with the caller's services in
            # its system prompt (catalog_tool's _interpret_system_message) — no
            # second call, and no template unless it came back blank. The
            # template serves a blank reply of any kind: it says what the
            # assistant is for and asks what they are in the mood for.
            yield {
                "type": "message",
                "text": state["answer_text"]
                or _capability_message(req.watch_provider_names),
            }
        elif state is not None and "lookup_picks" in state:
            if state["lookup_picks"]:
                yield {
                    "type": "results",
                    "kind": "lookup",
                    "relaxed": [],
                    "note": "",
                    "picks": state["lookup_picks"],
                }
            else:
                yield {
                    "type": "message",
                    "text": await _outcome_text(
                        req.message,
                        "title_not_found",
                        model=compose_model,
                        config=config,
                        title=state["lookup_title"],
                        relaxed=[],
                    ),
                }
        elif state is not None and "found" in state:
            result = await finish_search(
                state["found"],
                state.get("structured_response"),
                client=client,
                watch_region=req.watch_region,
                watch_providers=req.watch_providers,
            )
            if result.picks:
                yield {
                    "type": "results",
                    "kind": "search",
                    "relaxed": result.relaxed,
                    "note": _relaxed_note(
                        result.relaxed, result.exact_matches, len(result.picks)
                    ),
                    "picks": result.picks,
                }
            elif result.cause is not None:
                yield {
                    "type": "message",
                    "text": await _outcome_text(
                        req.message,
                        result.cause,
                        model=compose_model,
                        config=config,
                        title="",
                        relaxed=result.relaxed,
                    ),
                }
            else:
                # Unreachable by construction: finish_search() sets a cause
                # whenever it returns no picks.
                yield {
                    "type": "message",
                    "text": _nothing_found_message("nothing_matched", []),
                }
        else:
            # Unreachable by construction — TwoCallTurn ends every turn with
            # exactly one tool run. Total anyway: a turn that emitted only
            # "done" would leave the gateway nothing to persist and the user a
            # blank answer.
            yield {
                "type": "message",
                "text": _nothing_found_message("nothing_matched", []),
            }
        yield {"type": "done"}
    except Exception as exc:
        # Not `except BaseException` — a cancelled turn (asyncio.CancelledError)
        # and a closed stream (GeneratorExit) must propagate, not be reported
        # as an error.
        #
        # Spans the whole body, not just the agent: the outcome branch awaits
        # a model call of its own, and an exception escaping it would
        # otherwise end the turn with no event at all, which the gateway would
        # see as a stream that stopped rather than an error frame.
        reason = _error_reason(exc)
        if reason == "internal":
            logger.exception("chat pipeline failed")
        else:
            logger.info("chat pipeline failed", extra={"reason": reason})
        yield {"type": "error", "reason": reason}
