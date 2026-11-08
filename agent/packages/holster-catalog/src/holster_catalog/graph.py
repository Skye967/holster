"""The catalog turn as a LangGraph graph: the steps catalog_tool.py defines, and
the edges that bound them.

    START -> interpret -+-> search -+-> rank -> finish -> END
                        |           +-> compose -> END    (no candidates)
                        +-> lookup -+-> END
                        |           +-> compose -> END    (title not found)
                        +-> answer -> END

interpret is the one node offered tools, and no edge leads back to it, so
nothing read within a turn after the user's message — TMDB overviews included —
can pick an action. That is the graph's shape rather than a check at run time:
there is no loop to stop, and rank binds a response schema and no tools.

Nothing executes a tool: the model fills one declared schema and the router
reads which. turn.py's outcome_of reads the turn's answer off the final state.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol, TypedDict, cast

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.runnables import Runnable
from langgraph.errors import NodeError, NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from holster_catalog.catalog_tool import (
    _RANK_SYSTEM_PROMPT,
    CATALOG_TOOLS,
    AnswerArgs,
    Candidates,
    CatalogResult,
    CatalogToolError,
    Composer,
    LookUpArgs,
    OutcomeCause,
    RankResult,
    SearchArgs,
    TitleRef,
    TitleVerdict,
    _capability_message,
    _empty_cause,
    _interpret_system_message,
    _nothing_found_message,
    _one_line,
    _rank_content,
    _relaxed_labels,
    _title_not_found_message,
    find_candidates,
    finish_search,
    lookup_title,
)
from holster_catalog.tmdb import TMDBClient
from holster_catalog.turn import Intent, Opener, Turn

logger = logging.getLogger("holster.graph")

# The node names that count as a model call. No path from START to END may pass
# through more than two of them.
#
# Two because the step a third call would take — try again, looser — is what
# find_candidates()'s ladder does without one, and a third would follow the
# rank call's reading of the overviews. main.py gives the graph's model a 15s
# timeout per attempt and one retry, so a single slow call can already spend
# gateway/chat.go's 30s turnDeadline; and gateway/chat.go's
# defaultGuestRateLimit is sized on a turn costing two calls of the free tier's
# 15 requests a minute. Allowing a third call takes a step the ladder cannot do,
# a call that reads no third-party text, and room under turnDeadline and
# defaultGuestRateLimit.
#
# compose takes the rank call's place in the two, and every turn that reaches it
# has that call to spare: it is entered from search and lookup alone. That rests
# on finish having one edge, which rests on a rank call never leaving an empty
# pick list: finish_search()'s top-up fills the list to `floor`, which
# min(RESULT_FLOOR, limit) and SearchArgs.limit's `ge=1` hold at one or more, and
# search routes to rank only with candidates.
MODEL_NODES = frozenset({"interpret", "rank", "compose"})

# One super-step for each node on the longest path — interpret, search, rank,
# finish — plus the input step LangGraph runs first. A run that needs more raises
# GraphRecursionError, which agent/chat.py's _error_reason reports as internal.
RECURSION_LIMIT = 5

# What the outcome composer gets before the turn answers from its template. The
# SDK's own envelope is no bound worth resting on: agent/main.py builds the
# composer's model with timeout=15.0, three times this cap on its own.
#
# It bounds what this call adds, not what the turn has already spent — an empty
# turn arrives here having run page 1 and the whole ladder — so a turn already
# near the gateway's 30s turnDeadline still passes it, and the dropped-stream
# fallback is what the user sees.
COMPOSE_TIMEOUT_SECONDS = 5.0


@dataclass
class TurnContext:
    """What the nodes read from the run rather than from the model: the
    caller's own TMDB client, country, ticked services and their names, verdicts
    and shown titles. None of these is a field of a declared tool, so none is
    something the model can fill."""

    client: TMDBClient
    watch_region: str
    # Never empty: agent/chat.py answers a caller with no ticked services before the
    # graph runs, and discover() raises on an empty list.
    watch_providers: list[int]
    # Read by the interpret and answer nodes, never rank — see
    # _interpret_system_message.
    watch_provider_names: list[str]
    verdicts: list[TitleVerdict]
    shown: list[TitleRef]


class TurnState(TypedDict, total=False):
    """The turn's input, plus what each node wrote. `message` and `history` are
    the input and no node writes them. One of search, lookup and answer runs a
    turn, and compose adds `text` to what search or lookup wrote, which is why
    turn.py's outcome_of reads `text` first."""

    message: str
    # Oldest first, the current message not included.
    history: list[Turn]
    # interpret: the one declared schema the model filled.
    intent: SearchArgs | LookUpArgs | AnswerArgs
    # search, and rank and finish when there were candidates.
    found: Candidates
    ranked: RankResult
    result: CatalogResult
    # lookup: the enriched rows.
    lookup_picks: list[dict[str, Any]]
    # answer or compose: the turn's reply, never empty.
    text: str


# A Protocol because add_node's types want the parameter named `runtime`, which
# a Callable alias cannot state.
class _ModelNode(Protocol):
    async def __call__(
        self, state: TurnState, runtime: Runtime[TurnContext]
    ) -> TurnState: ...


async def _model_call[T](
    model: Runnable[LanguageModelInput, T], messages: Sequence[BaseMessage]
) -> T:
    """Run the call, wrapping any failure — a network/API error, a tool call
    that names no declared tool or carries arguments its schema rejects, or a
    ranking that doesn't satisfy RankResult's schema — as CatalogToolError,
    which agent/chat.py's _error_reason maps to model_unavailable. There is no handing
    the error back to the model, which would be another call. ``except
    Exception``, not ``BaseException``, so a cancelled step still propagates."""
    try:
        return await model.ainvoke(messages)
    except Exception as exc:
        raise CatalogToolError(str(exc)) from exc


def _history_messages(history: Sequence[Turn]) -> list[AnyMessage]:
    """Each earlier turn as its own message, oldest first. Both the interpret and
    the rank call are sent these ahead of their own last message."""
    return [
        HumanMessage(t.text) if t.role == "user" else AIMessage(t.text) for t in history
    ]


def _narrate(runtime: Runtime[TurnContext], args: SearchArgs | LookUpArgs) -> None:
    """The opener, then the intent the gateway templates its interpreting line
    from, both before any TMDB call. The opener goes first because it is prose
    above the interpreting line and the gateway forwards in arrival order.
    The opener is cut to one line, and skipped when blank rather than sent
    empty — reply is defaulted, so a response can leave it out."""
    reply = _one_line(args.reply)
    if reply:
        runtime.stream_writer(Opener(reply))
    runtime.stream_writer(Intent(args))


def interpret_node(model: BaseChatModel) -> _ModelNode:
    """The interpret call around ``model``: the interpret prompt, the three
    declared tools and tool_choice="any"."""
    bound = model.bind_tools(CATALOG_TOOLS, tool_choice="any")
    interpreter = bound | PydanticToolsParser(tools=[*CATALOG_TOOLS])

    async def interpret(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
        names = runtime.context.watch_provider_names
        system = _interpret_system_message(names, today=date.today())
        messages = [
            system,
            *_history_messages(state["history"]),
            HumanMessage(state["message"]),
        ]
        picked = await _model_call(interpreter, messages)
        # Never the first of several: a message asking two things at once can
        # draw two tool calls, and taking one would answer half of it.
        if len(picked) != 1:
            raise CatalogToolError("interpret call did not pick exactly one tool")
        args = picked[0]
        # An answer narrates nothing: its reply is the answer itself, and an
        # interpreting line above it would announce a search that never runs.
        if not isinstance(args, AnswerArgs):
            _narrate(runtime, args)
        return {"intent": args}

    return interpret


async def search(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
    context = runtime.context
    found = await find_candidates(
        cast(SearchArgs, state["intent"]),
        client=context.client,
        watch_region=context.watch_region,
        watch_providers=context.watch_providers,
        verdicts=context.verdicts,
        shown=context.shown,
    )
    return {"found": found}


async def lookup(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
    context = runtime.context
    picks = await lookup_title(
        cast(LookUpArgs, state["intent"]).title,
        client=context.client,
        watch_region=context.watch_region,
        watch_providers=context.watch_providers,
    )
    return {"lookup_picks": picks}


async def answer(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
    args = cast(AnswerArgs, state["intent"])
    # _one_line turns whitespace-only text into "", and the capability template
    # serves a blank reply of any kind: it says what the assistant is for and
    # asks what they are in the mood for. A plain line break it flattens would
    # render as a space in the web's <p> anyway.
    names = runtime.context.watch_provider_names
    return {"text": _one_line(args.text) or _capability_message(names)}


def rank_node(model: BaseChatModel) -> _ModelNode:
    """The rank call around ``model``: the rank prompt, no tools, and RankResult
    as its only answer. ``method`` is named so the call uses the provider's own
    JSON schema support and parses message text, whatever the library's default
    becomes.

    A reply that cannot be read as a RankResult ends the turn, with no retry.
    The parser repairs cut-off JSON and nothing here reads the finish reason,
    so a truncated reply can pass with its last blurb cut or its last picks
    gone. That is the accepted cost: no call sets max_output_tokens, so the cap
    is the model's own, far past a rank reply. Setting one means checking
    finish_reason here via include_raw."""
    ranker = model.with_structured_output(RankResult, method="json_schema")

    async def rank(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
        request = HumanMessage(_rank_content(state["message"], state["found"]))
        messages = [
            SystemMessage(_RANK_SYSTEM_PROMPT),
            *_history_messages(state["history"]),
            request,
        ]
        return {"ranked": cast(RankResult, await _model_call(ranker, messages))}

    return rank


async def finish(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
    context = runtime.context
    result = await finish_search(
        state["found"],
        state["ranked"],
        client=context.client,
        watch_region=context.watch_region,
        watch_providers=context.watch_providers,
    )
    return {"result": result}


def _empty_outcome(state: TurnState) -> tuple[OutcomeCause, str, str]:
    """The cause, the composer's detail and the template, chosen together from
    the one cause, so the sentence a user reads cannot describe a different turn
    from the duty the model was set. Only nothing_matched has a detail the
    prompt defines beyond the title: the other two are about what the user has
    already seen or rated, where naming what the ladder loosened would merge two
    answers."""
    intent = state["intent"]
    if isinstance(intent, LookUpArgs):
        return "title_not_found", intent.title, _title_not_found_message(intent.title)
    found = state["found"]
    cause = _empty_cause(found)
    detail = _relaxed_labels(found.relaxed) if cause == "nothing_matched" else ""
    return cause, detail, _nothing_found_message(cause, found.relaxed)


def compose_node(composer: Composer) -> _ModelNode:
    """The model's own sentence for a turn that ended with nothing to show. A
    blank result answers from the template: ComposedReply.text is required of
    the model, which still leaves "" satisfying the schema, and an empty reply
    would persist an empty assistant turn."""

    async def compose_outcome(
        state: TurnState, runtime: Runtime[TurnContext]
    ) -> TurnState:
        cause, detail, fallback = _empty_outcome(state)
        composed = await composer(state["message"], cause, detail)
        return {"text": composed.text.strip() or fallback}

    return compose_outcome


async def compose_fallback(state: TurnState, error: NodeError) -> TurnState:
    """Degrade, don't fail, the same split catalog_tool's _safe() makes: the turn
    already has its answer and only the wording is at stake.

    Every composer failure lands here, a reply of the wrong shape included:
    google_composer's cast is erased at runtime, so compose_outcome's .text raises
    on it. Warning, not exception, for that same reason; the traceback is kept
    because a schema or wiring fault fails every call and would otherwise show up
    only as prose that never got warmer. A timeout inside the SDK is in that arm
    too; NodeTimeoutError is the deadline this graph owns."""
    cause, _, fallback = _empty_outcome(state)
    if isinstance(error.error, NodeTimeoutError):
        logger.info("outcome composer timed out, degraded", extra={"cause": cause})
    else:
        logger.warning(
            "outcome composer failed, degraded",
            exc_info=error.error,
            extra={"cause": cause},
        )
    return {"text": fallback}


def _route_intent(state: TurnState) -> str:
    intent = state["intent"]
    if isinstance(intent, SearchArgs):
        return "search"
    if isinstance(intent, LookUpArgs):
        return "lookup"
    return "answer"


def _route_search(state: TurnState) -> str:
    return "rank" if state["found"].candidates else "compose"


def _route_lookup(state: TurnState) -> str:
    return END if state["lookup_picks"] else "compose"


CatalogGraph = CompiledStateGraph[TurnState, TurnContext]


def build_turn(model: BaseChatModel, composer: Composer) -> CatalogGraph:
    """The turn's graph around ``model`` and ``composer``, each binding made once,
    here. Every run gets RECURSION_LIMIT from the compiled graph, so no caller can
    leave it out."""
    graph = StateGraph(TurnState, context_schema=TurnContext)
    graph.add_node("interpret", interpret_node(model))
    graph.add_node("search", search)
    graph.add_node("lookup", lookup)
    graph.add_node("answer", answer)
    graph.add_node("rank", rank_node(model))
    graph.add_node("finish", finish)
    graph.add_node(
        "compose",
        compose_node(composer),
        timeout=COMPOSE_TIMEOUT_SECONDS,
        error_handler=compose_fallback,
    )
    graph.add_edge(START, "interpret")
    # path_map is what puts a conditional edge's targets in get_graph().
    graph.add_conditional_edges(
        "interpret", _route_intent, path_map=["search", "lookup", "answer"]
    )
    graph.add_conditional_edges("search", _route_search, path_map=["rank", "compose"])
    graph.add_conditional_edges("lookup", _route_lookup, path_map=[END, "compose"])
    graph.add_edge("rank", "finish")
    graph.add_edge("finish", END)
    graph.add_edge("answer", END)
    graph.add_edge("compose", END)
    return graph.compile().with_config(recursion_limit=RECURSION_LIMIT)
