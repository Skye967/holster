"""The catalog turn as a LangGraph graph: the steps catalog_tool.py defines, and
the edges that bound them.

    START -> interpret -+-> search -+-> rank -> END
                        |           +-> END           (no candidates)
                        +-> lookup -> END
                        +-> answer -> END

interpret is the one node offered tools, and no edge leads back to it, so
nothing read within a turn after the user's message — TMDB overviews included —
can pick an action. That is the graph's shape rather than a check at run time:
there is no loop to stop, and rank binds a response schema and no tools.

Nothing executes a tool: the model fills one declared schema and the router
reads which. chat.py reads the final state, then runs finish_search() and writes
the turn's answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol, TypedDict, cast

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers import PydanticToolsParser
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from holster_catalog.catalog_tool import (
    _RANK_SYSTEM_PROMPT,
    CATALOG_TOOLS,
    AnswerArgs,
    AnswerKind,
    Candidates,
    CatalogToolError,
    LookUpArgs,
    RankResult,
    SearchArgs,
    TitleRef,
    TitleVerdict,
    _interpret_system_message,
    _one_line,
    _rank_content,
    find_candidates,
    lookup_title,
)
from holster_catalog.tmdb import TMDBClient

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
# compose is listed though this graph has no such node, so that a node of that
# name counts toward the two as soon as it exists. Today chat.py's stream_chat
# calls the composer once the graph has ended, and only on a turn that made no
# rank call, so it takes the rank call's place in the two. These edges do not
# hold that.
MODEL_NODES = frozenset({"interpret", "rank", "compose"})

# One super-step for each node on the longest path — interpret, search, rank —
# plus the input step LangGraph runs first. A run that needs more raises
# GraphRecursionError, which chat.py's _error_reason reports as internal.
RECURSION_LIMIT = 4


@dataclass
class TurnContext:
    """What the nodes read from the run rather than from the model: the
    caller's own TMDB client, country, ticked services and their names, verdicts
    and shown titles. None of these is a field of a declared tool, so none is
    something the model can fill."""

    client: TMDBClient
    watch_region: str
    # Never empty: chat.py answers a caller with no ticked services before the
    # graph runs, and discover() raises on an empty list.
    watch_providers: list[int]
    # Read by the interpret call alone, never the rank call — see
    # _interpret_system_message.
    watch_provider_names: list[str]
    verdicts: list[TitleVerdict]
    shown: list[TitleRef]


class TurnState(TypedDict, total=False):
    """The turn's input, plus what each node wrote, typed for chat.py to read.
    One of search, lookup and answer runs a turn, so at most one group below
    `intent` is set.

    A history turn in `messages` is text only, never a replayed tool call:
    chat.py's HistoryTurn has no tool fields. Giving it some would let an
    earlier turn's tool results, TMDB overviews included, reach the interpret
    call, which picks the action."""

    # The input chat.py always sends: the history, oldest first, then the
    # current message last. No node writes it.
    messages: list[AnyMessage]
    # interpret: the one declared schema the model filled.
    intent: SearchArgs | LookUpArgs | AnswerArgs
    # search, and rank when there were candidates.
    found: Candidates
    ranked: RankResult
    # lookup: the enriched rows, and the name asked about, which the composer
    # quotes back when nothing carries it.
    lookup_picks: list[dict[str, Any]]
    lookup_title: str
    # answer
    answer_kind: AnswerKind
    answer_text: str


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
    which chat.py's _error_reason maps to model_unavailable. There is no handing
    the error back to the model, which would be another call. ``except
    Exception``, not ``BaseException``, so a cancelled step still propagates."""
    try:
        return await model.ainvoke(messages)
    except Exception as exc:
        raise CatalogToolError(str(exc)) from exc


def _narrate(runtime: Runtime[TurnContext], reply: str, intent: dict[str, Any]) -> None:
    """The opener, then the intent the gateway templates its interpreting line
    from, both before any TMDB call. The opener goes first because it is prose
    above the interpreting line and the gateway forwards in arrival order.
    The opener is cut to one line, and skipped when blank rather than sent
    empty — reply is defaulted, so a response can leave it out."""
    reply = _one_line(reply)
    if reply:
        runtime.stream_writer({"type": "reply", "text": reply})
    runtime.stream_writer({"type": "intent", "intent": intent})


def interpret_node(model: BaseChatModel) -> _ModelNode:
    """The interpret call around ``model``: the interpret prompt, the three
    declared tools and tool_choice="any". The messages go through as chat.py
    sent them, history included, unlike the rank call's."""
    bound = model.bind_tools(CATALOG_TOOLS, tool_choice="any")
    interpreter = bound | PydanticToolsParser(tools=[*CATALOG_TOOLS])

    async def interpret(state: TurnState, runtime: Runtime[TurnContext]) -> TurnState:
        names = runtime.context.watch_provider_names
        system = _interpret_system_message(names, today=date.today())
        picked = await _model_call(interpreter, [system, *state["messages"]])
        # Never the first of several: a message asking two things at once can
        # draw two tool calls, and taking one would answer half of it.
        if len(picked) != 1:
            raise CatalogToolError("interpret call did not pick exactly one tool")
        args = picked[0]
        # An answer narrates nothing: its reply is the answer itself, and an
        # interpreting line above it would announce a search that never runs.
        if isinstance(args, SearchArgs):
            # The intent event is SearchArgs by JSON name, which
            # gateway/chat.go's agentIntent mirrors. reply leaves it because it
            # has its own event.
            _narrate(
                runtime,
                args.reply,
                args.model_dump(exclude_none=True, exclude={"reply"}),
            )
        elif isinstance(args, LookUpArgs):
            # The gateway templates a lookup's interpreting line off title
            # alone (gateway/chat.go's interpretingLine).
            _narrate(runtime, args.reply, {"title": args.title})
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
    title = cast(LookUpArgs, state["intent"]).title
    context = runtime.context
    picks = await lookup_title(
        title,
        client=context.client,
        watch_region=context.watch_region,
        watch_providers=context.watch_providers,
    )
    return {"lookup_picks": picks, "lookup_title": title}


async def answer(state: TurnState) -> TurnState:
    args = cast(AnswerArgs, state["intent"])
    # _one_line turns whitespace-only text into "", which chat.py answers with
    # its capability template instead; a plain line break it flattens would
    # render as a space in the web's <p> anyway.
    return {"answer_kind": args.kind, "answer_text": _one_line(args.text)}


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
        *history, current = state["messages"]
        request = HumanMessage(_rank_content(current.text, state["found"]))
        messages = [SystemMessage(_RANK_SYSTEM_PROMPT), *history, request]
        return {"ranked": cast(RankResult, await _model_call(ranker, messages))}

    return rank


def _route_intent(state: TurnState) -> str:
    intent = state["intent"]
    if isinstance(intent, SearchArgs):
        return "search"
    if isinstance(intent, LookUpArgs):
        return "lookup"
    return "answer"


def _route_search(state: TurnState) -> str:
    return "rank" if state["found"].candidates else END


CatalogGraph = CompiledStateGraph[TurnState, TurnContext]


def build_turn(model: BaseChatModel) -> CatalogGraph:
    """The turn's graph around ``model``, each binding made once, here. Every
    run gets RECURSION_LIMIT from the compiled graph, so no caller can leave it
    out."""
    graph = StateGraph(TurnState, context_schema=TurnContext)
    graph.add_node("interpret", interpret_node(model))
    graph.add_node("search", search)
    graph.add_node("lookup", lookup)
    graph.add_node("answer", answer)
    graph.add_node("rank", rank_node(model))
    graph.add_edge(START, "interpret")
    # path_map is what puts a conditional edge's targets in get_graph().
    graph.add_conditional_edges(
        "interpret", _route_intent, path_map=["search", "lookup", "answer"]
    )
    graph.add_conditional_edges("search", _route_search, path_map=["rank", END])
    graph.add_edge("lookup", END)
    graph.add_edge("answer", END)
    graph.add_edge("rank", END)
    return graph.compile().with_config(recursion_limit=RECURSION_LIMIT)
