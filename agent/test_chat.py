"""chat.py tests — offline, against fake interpret/rank models and a fake
TMDB transport, same shape as test_catalog_tool.py. These exercise the event
sequence and cancellation behaviour stream_chat() adds on top of search();
search()'s own behaviour (relaxation ladder, id-outside-candidates, etc.) is
covered there and not re-tested here.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import httpx2
import pytest

import tmdb
from catalog_tool import DiscoverIntent, RankedPick, RankResult
from chat import ChatRequest, HistoryTurn, stream_chat
from testutil import MOVIE_A, FakeTMDB, make_intent, ok_interpret
from tmdb import Title, TMDBClient


async def _collect(
    req: ChatRequest, client: TMDBClient, interpret: Any, rank: Any
) -> list[dict[str, Any]]:
    return [
        e
        async for e in stream_chat(
            req, client=client, interpret_model=interpret, rank_model=rank
        )
    ]


def test_no_providers_short_circuits_with_a_message() -> None:
    fake_tmdb = FakeTMDB()
    req = ChatRequest(message="anything", watch_region="US", watch_providers=[])

    async def must_not_run(message: str) -> DiscoverIntent:
        raise AssertionError("interpret must not run with no subscriptions")

    events = asyncio.run(_collect(req, fake_tmdb.client(), must_not_run, must_not_run))

    assert [e["type"] for e in events] == ["message", "done"]
    assert "Connections" in events[0]["text"]
    assert fake_tmdb.requests == []


def test_successful_search_emits_intent_then_results_then_done() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})
    req = ChatRequest(message="a heist movie", watch_region="US", watch_providers=[8])

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(keywords=["heist"])

    async def fake_rank(message: str, candidates: list[Title]) -> RankResult:
        pick = RankedPick(tmdb_id=candidates[0]["tmdb_id"], blurb="great fit")
        return RankResult(picks=[pick])

    events = asyncio.run(_collect(req, fake_tmdb.client(), fake_interpret, fake_rank))

    assert [e["type"] for e in events] == ["intent", "results", "done"]
    assert events[0]["intent"]["keywords"] == ["heist"]
    [pick] = events[1]["picks"]
    assert pick["title"] == "Fake Heist"
    assert pick["blurb"] == "great fit"
    assert events[1]["relaxed"] == []


def test_intent_event_arrives_before_rank_is_called() -> None:
    """The whole reason for on_intent: the caller must be able to observe it
    before the (slower) rank step runs."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})
    req = ChatRequest(message="something", watch_region="US", watch_providers=[8])
    order: list[str] = []

    async def fake_interpret(message: str) -> DiscoverIntent:
        order.append("interpret")
        return make_intent()

    async def fake_rank(message: str, candidates: list[Title]) -> RankResult:
        order.append("rank")
        return RankResult(picks=[])

    events = asyncio.run(_collect(req, fake_tmdb.client(), fake_interpret, fake_rank))

    assert [e["type"] for e in events] == ["intent", "message", "done"]
    assert order == ["interpret", "rank"]


def test_no_candidates_after_ladder_emits_a_message_naming_what_was_relaxed() -> None:
    fake_tmdb = FakeTMDB()
    # every /discover/movie call returns nothing; keyword resolves so it counts
    fake_tmdb.ok("/search/keyword", {"results": [{"id": 42, "name": "cozy"}]})
    req = ChatRequest(message="cozy and short", watch_region="US", watch_providers=[8])

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(max_runtime_minutes=90, keywords=["cozy"])

    async def rank_must_not_run(message: str, candidates: list[Title]) -> RankResult:
        raise AssertionError("rank must not run on zero candidates")

    events = asyncio.run(
        _collect(req, fake_tmdb.client(), fake_interpret, rank_must_not_run)
    )

    assert [e["type"] for e in events] == ["intent", "message", "done"]
    assert "how long" in events[1]["text"]
    assert "the mood" in events[1]["text"]


def test_tmdb_unavailable_maps_to_a_closed_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tmdb, "_sleep", AsyncMock())
    fake_tmdb = FakeTMDB()
    fake_tmdb.queue("/discover/movie", *[httpx2.Response(503) for _ in range(3)])
    req = ChatRequest(message="something", watch_region="US", watch_providers=[8])

    events = asyncio.run(_collect(req, fake_tmdb.client(), ok_interpret, ok_interpret))

    # error is terminal on its own — no trailing "done" (a turn ends with
    # exactly one of the two, never both; see chat.py's module docstring).
    assert [e["type"] for e in events] == ["intent", "error"]
    assert events[1]["reason"] == "tmdb_unavailable"


def test_interpret_failure_maps_to_model_unavailable() -> None:
    fake_tmdb = FakeTMDB()
    req = ChatRequest(message="something", watch_region="US", watch_providers=[8])

    async def failing_interpret(message: str) -> DiscoverIntent:
        raise RuntimeError("anthropic timed out")

    events = asyncio.run(
        _collect(req, fake_tmdb.client(), failing_interpret, ok_interpret)
    )

    assert [e["type"] for e in events] == ["error"]
    assert events[0]["reason"] == "model_unavailable"


def test_bad_watch_region_maps_to_internal_not_tmdb_specific() -> None:
    """watch_region is caller-supplied, never model-derived (catalog_tool.py) —
    a malformed one is a gateway/caller bug, not 'try later'."""
    fake_tmdb = FakeTMDB()
    req = ChatRequest(message="something", watch_region="usa", watch_providers=[8])

    events = asyncio.run(_collect(req, fake_tmdb.client(), ok_interpret, ok_interpret))

    # on_intent already fired by the time discover() validates watch_region —
    # see catalog_tool.search(): interpret() -> on_intent() -> discover().
    assert [e["type"] for e in events] == ["intent", "error"]
    assert events[1]["reason"] == "internal"


def test_history_is_folded_into_the_message_text() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": []})
    req = ChatRequest(
        message="anything shorter?",
        watch_region="US",
        watch_providers=[8],
        history=[
            HistoryTurn(role="user", text="something like Heat"),
            HistoryTurn(role="assistant", text="Here are some heist films."),
        ],
    )
    seen: list[str] = []

    async def capturing_interpret(message: str) -> DiscoverIntent:
        seen.append(message)
        return make_intent()

    asyncio.run(_collect(req, fake_tmdb.client(), capturing_interpret, ok_interpret))

    [message] = seen
    assert "something like Heat" in message
    assert "Here are some heist films." in message
    assert message.endswith("Current message: anything shorter?")


def test_early_disconnect_cancels_the_background_search() -> None:
    """Simulates the caller (main.py's StreamingResponse) stopping iteration
    early, as it does on client disconnect — see chat.py's module docstring
    on cancellation.

    Deliberately does not assert *where* the cancellation lands (inside
    discover() or inside rank(), depending on scheduling) — only the
    contract that matters: the pipeline never reaches a result once the
    generator has been closed, and aclose() does not return until the
    background task has actually stopped (see stream_chat()'s finally,
    which awaits it rather than firing-and-forgetting task.cancel())."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})
    req = ChatRequest(message="something", watch_region="US", watch_providers=[8])
    completed = asyncio.Event()

    async def slow_rank(message: str, candidates: list[Title]) -> RankResult:
        await asyncio.sleep(10)
        completed.set()
        return RankResult(picks=[])

    async def run() -> None:
        gen = stream_chat(
            req,
            client=fake_tmdb.client(),
            interpret_model=ok_interpret,
            rank_model=slow_rank,
        )
        first = await gen.__anext__()
        assert first["type"] == "intent"
        await gen.aclose()

    asyncio.run(asyncio.wait_for(run(), timeout=2))
    assert not completed.is_set()
