"""The turn's graph, offline: the edges that hold the two-call bound, what each
model binding carried, and the nodes refusing a misbehaving model, each called
on its own with a hand-built Runtime. test_turn.py runs most of those refusals
through a whole turn.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import replace
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START
from langgraph.runtime import Runtime

from holster_catalog import tmdb
from holster_catalog.catalog_tool import (
    RESULT_FLOOR,
    Candidates,
    CatalogToolError,
    RankResult,
    SearchCause,
    _nothing_found_message,
)
from holster_catalog.graph import (
    MODEL_NODES,
    RECURSION_LIMIT,
    TurnContext,
    TurnState,
    _empty_outcome,
    build_turn,
    interpret_node,
    rank_node,
)
from holster_catalog.testing import (
    MOVIE_A,
    FakeTMDB,
    _tool_call,
    lookup_call,
    make_intent,
    ok_compose,
    scripted,
    search_call,
)


def _edges() -> list[tuple[str, str]]:
    return [
        (e.source, e.target)
        for e in build_turn(scripted(), ok_compose).get_graph().edges
    ]


def _paths() -> list[list[str]]:
    """Every path from START to END, refusing an edge back onto its own path."""
    following: dict[str, list[str]] = defaultdict(list)
    for source, target in _edges():
        following[source].append(target)

    def walk(path: list[str]) -> list[list[str]]:
        if path[-1] == END:
            return [path]
        paths = []
        for target in following[path[-1]]:
            assert target not in path, f"cycle: {[*path, target]}"
            paths += walk([*path, target])
        return paths

    return walk([START])


def test_no_path_passes_through_more_than_two_model_nodes() -> None:
    """Also what proves there is no cycle: _paths() refuses one."""
    paths = _paths()
    assert paths
    assert max(len(MODEL_NODES.intersection(path)) for path in paths) == 2


def test_finish_ends_the_turn_and_compose_follows_no_rank_call() -> None:
    """finish ends the turn and compose is entered only where no rank call was
    made, which keeps a third model node off every path."""
    edges = _edges()
    assert [t for s, t in edges if s == "finish"] == [END]
    assert sorted(s for s, t in edges if t == "compose") == ["lookup", "search"]


def test_nothing_leads_back_to_the_interpret_node() -> None:
    """It alone is offered tools, so nothing read after it can pick an action."""
    assert [s for s, target in _edges() if target == "interpret"] == [START]


def test_the_recursion_limit_is_the_longest_path_plus_one() -> None:
    """Counted in nodes: a run spends one step on each, and one on its input."""
    longest = max(len(path) - 2 for path in _paths())
    assert longest == RECURSION_LIMIT - 1


def test_every_run_of_the_graph_carries_the_recursion_limit() -> None:
    config = build_turn(scripted(), ok_compose).config
    assert config is not None and config["recursion_limit"] == RECURSION_LIMIT


def test_each_binding_carries_only_what_its_call_needs() -> None:
    """The interpret call must answer with a tool call, so it has no response
    format, and the rank call must answer with a ranking, so it has no tools."""
    model = scripted()

    build_turn(model, ok_compose)

    assert model.bound == [
        {
            "tools": ["search_catalog", "look_up_title", "answer"],
            "tool_choice": "any",
        },
        {"tools": [], "schema": RankResult, "method": "json_schema"},
    ]


def _runtime(events: list[Any]) -> Runtime[TurnContext]:
    context = TurnContext(
        client=FakeTMDB().client(),
        watch_region="US",
        watch_providers=[8],
        watch_provider_names=[],
        verdicts=[],
        shown=[],
    )
    return Runtime(context=context, stream_writer=events.append)


def _state() -> TurnState:
    return TurnState(
        message="a heist movie",
        history=[],
        found=Candidates(
            candidates=[tmdb._trim_title(MOVIE_A, "movie")],
            rung_of={101: 0},
            relaxed=[],
            limit=10,
            floor=RESULT_FLOOR,
            taste=[],
            newest_first=False,
            all_judged=False,
            all_shown=False,
        ),
    )


@pytest.mark.parametrize(
    "reply",
    [
        AIMessage(content="Sure, here are some heist films."),
        AIMessage(
            content="",
            tool_calls=search_call().tool_calls + lookup_call("Fargo").tool_calls,
        ),
        _tool_call("fetch_url", {}),
        search_call(limit="lots"),
    ],
    ids=["no_tool_call", "two_tool_calls", "undeclared_tool", "rejected_argument"],
)
def test_the_interpret_node_refuses_anything_but_one_valid_tool_call(
    reply: AIMessage,
) -> None:
    """Refused before narrating: an opener or an intent would announce a search
    that never runs."""
    events: list[Any] = []
    node = interpret_node(scripted(reply))

    with pytest.raises(CatalogToolError):
        asyncio.run(node(_state(), _runtime(events)))

    assert events == []


@pytest.mark.parametrize(
    "reply",
    [AIMessage(content="not json"), search_call()],
    ids=["unparseable", "asks_for_a_tool"],
)
def test_the_rank_node_refuses_anything_but_a_ranking(reply: AIMessage) -> None:
    node = rank_node(scripted(reply))

    with pytest.raises(CatalogToolError):
        asyncio.run(node(_state(), _runtime([])))


@pytest.mark.parametrize(
    ("cause", "all_judged", "all_shown"),
    [("all_judged", True, False), ("all_shown", False, True)],
)
def test_a_seen_or_rated_cause_names_nothing_the_ladder_loosened(
    cause: SearchCause, all_judged: bool, all_shown: bool
) -> None:
    found = replace(
        _state()["found"],
        relaxed=["runtime"],
        all_judged=all_judged,
        all_shown=all_shown,
    )

    assert _empty_outcome(TurnState(intent=make_intent(), found=found)) == (
        cause,
        "",
        _nothing_found_message(cause, ["runtime"]),
    )
