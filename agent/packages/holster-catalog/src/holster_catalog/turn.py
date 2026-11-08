"""A turn from outside the graph: what goes in, what comes out as it runs, and the
one outcome it ends on.

stream_turn() yields each event as the interpret node writes it, then the
outcome read from the graph's final state. That is what lets a caller hear the
intent in ~1-2s while the rest of the turn (a deep search is several TMDB round
trips, plus the rank call) is still running. A failed turn raises instead of
yielding an outcome: tmdb.TMDBUnavailable, CatalogToolError, or a bug.

The types carry no wire format. agent/chat.py maps them to the NDJSON the gateway
reads.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from holster_catalog.catalog_tool import (
    LookUpArgs,
    RelaxedConstraint,
    SearchArgs,
    _relaxed_note,
)

# Types only: graph.py imports this module for Turn and the events.
if TYPE_CHECKING:
    from holster_catalog.graph import CatalogGraph, TurnContext, TurnState


class Turn(BaseModel):
    """One earlier message of the conversation. Text only, never a replayed tool
    call: a field for one would let an earlier turn's tool results, TMDB
    overviews included, reach the interpret call, which picks the action."""

    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class Opener:
    """The model's one-line opener, before any result exists."""

    text: str


@dataclass(frozen=True)
class Intent:
    """What a search or a lookup is about to look for, before any TMDB call."""

    args: SearchArgs | LookUpArgs


TurnEvent = Opener | Intent


@dataclass(frozen=True)
class Results:
    """A turn that has titles to show. picks is never empty."""

    kind: Literal["search", "lookup"]
    picks: list[dict[str, Any]]
    relaxed: list[RelaxedConstraint]
    # The line above a widened list, empty when there is nothing to say.
    note: str
    exact_matches: int


@dataclass(frozen=True)
class Message:
    """A turn that answers in words. text is never empty."""

    text: str


Outcome = Results | Message


def outcome_of(state: TurnState) -> Outcome:
    """The outcome of a finished run's final state. answer and compose write
    text, finish writes result, and the graph's edges leave a lookup with picks
    as the one other way to END. text goes first because compose writes it
    alongside what search or lookup left: an empty lookup_picks, say. A state
    that is none of these raises KeyError: no run ends that way, and the caller
    reports it as a failed turn."""
    if "text" in state:
        return Message(state["text"])
    if "result" in state:
        result = state["result"]
        note = _relaxed_note(result.relaxed, result.exact_matches, len(result.picks))
        return Results(
            kind="search",
            picks=result.picks,
            relaxed=result.relaxed,
            note=note,
            exact_matches=result.exact_matches,
        )
    return Results(
        kind="lookup",
        picks=state["lookup_picks"],
        relaxed=[],
        note="",
        exact_matches=0,
    )


async def stream_turn(
    graph: CatalogGraph,
    *,
    message: str,
    history: Sequence[Turn],
    context: TurnContext,
    config: RunnableConfig,
) -> AsyncGenerator[TurnEvent | Outcome]:
    """Run one turn. A caller that may stop early closes this generator, which
    closes the graph's stream before the close returns."""
    state: TurnState = {}
    # "custom" is what the interpret node writes, passed through as it arrives.
    # "values" is the whole state after each step; the last one is the turn's
    # result. astream is an async generator, typed as the iterator it returns.
    stream = cast(
        AsyncGenerator[tuple[str, Any]],
        graph.astream(
            {"message": message, "history": [*history]},
            config=config,
            context=context,
            stream_mode=["custom", "values"],
        ),
    )
    async with contextlib.aclosing(stream):
        async for mode, chunk in stream:
            if mode == "custom":
                yield cast(TurnEvent, chunk)
            else:
                state = cast("TurnState", chunk)
    yield outcome_of(state)
