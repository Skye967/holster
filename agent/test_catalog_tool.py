"""catalog_tool tests — offline, against fake interpret/rank models and a
fake TMDB transport (same FakeTMDB shape as test_tmdb.py). No LangChain
import here: interpret()/rank()/search() depend on plain callables, so a
fake is just an ``async def``, matching TMDBClient's ``transport=`` seam.

The live LLM path is in test_catalog_tool_live.py, which skips without
ANTHROPIC_API_KEY — same shape as test_tmdb_live.py.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import httpx2
import pytest

import tmdb
from catalog_tool import (
    CatalogToolError,
    DiscoverIntent,
    RankedPick,
    RankResult,
    interpret,
    rank,
    search,
)
from testutil import MOVIE_A, FakeTMDB, make_intent, ok_interpret
from tmdb import Title, TMDBUnavailable


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def _title(tmdb_id: int, media_type: tmdb.MediaType = "movie") -> Title:
    return Title(
        tmdb_id=tmdb_id,
        media_type=media_type,
        title=f"Title {tmdb_id}",
        year=2020,
        overview="An overview.",
        poster_url=None,
        vote_average=7.0,
        vote_count=500,
        genre_ids=[],
    )


async def _rank_must_not_run(message: str, candidates: list[Title]) -> RankResult:
    raise AssertionError("rank must not run once discover() has failed")


def test_interpret_calls_model_and_returns_its_intent() -> None:
    intent = make_intent(keywords=["heist"])

    async def fake(message: str) -> DiscoverIntent:
        assert message == "something like Heat"
        return intent

    assert run(interpret("something like Heat", model=fake)) is intent


def test_rank_drops_ids_outside_candidates() -> None:
    candidates = [_title(1), _title(2)]

    async def fake(message: str, cands: list[Title]) -> RankResult:
        return RankResult(
            picks=[
                RankedPick(tmdb_id=1, blurb="fits"),
                RankedPick(tmdb_id=999, blurb="hallucinated"),
            ]
        )

    picks = run(rank("mood", candidates, limit=5, model=fake))
    assert [p.tmdb_id for p in picks] == [1]


def test_rank_dedupes_and_truncates_to_limit() -> None:
    candidates = [_title(1), _title(2), _title(3)]

    async def fake(message: str, cands: list[Title]) -> RankResult:
        return RankResult(
            picks=[
                RankedPick(tmdb_id=1, blurb="a"),
                RankedPick(tmdb_id=1, blurb="dup"),
                RankedPick(tmdb_id=2, blurb="b"),
                RankedPick(tmdb_id=3, blurb="c"),
            ]
        )

    picks = run(rank("mood", candidates, limit=2, model=fake))
    assert [p.tmdb_id for p in picks] == [1, 2]


def test_rank_short_circuits_on_empty_candidates() -> None:
    called = False

    async def fake(message: str, cands: list[Title]) -> RankResult:
        nonlocal called
        called = True
        return RankResult(picks=[])

    assert run(rank("mood", [], limit=5, model=fake)) == []
    assert not called


def test_search_uses_callers_region_and_providers_never_the_models() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(keywords=["heist"])

    async def fake_rank(message: str, candidates: list[Title]) -> RankResult:
        return RankResult(picks=[])

    result = run(
        search(
            "something",
            client=fake_tmdb.client(),
            watch_region="GB",
            watch_providers=[8, 9],
            interpret_model=fake_interpret,
            rank_model=fake_rank,
        )
    )

    params = fake_tmdb.params_for("/discover/movie")
    assert params["watch_region"] == "GB"
    assert params["with_watch_providers"] == "8|9"
    # Correctness floors from tmdb.py — the model must not be able to lower
    # either, and this must still hold once a request has gone through the
    # full interpret() -> discover() -> rank() pipeline, not just discover()
    # in isolation.
    assert params["vote_count.gte"] == str(tmdb.MIN_VOTE_COUNT)
    assert params["with_watch_monetization_types"] == "flatrate"
    assert result.intent is not None
    assert result.intent.keywords == ["heist"]


def test_search_joins_blurb_with_the_real_title_not_model_metadata() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})

    async def fake_rank(message: str, candidates: list[Title]) -> RankResult:
        [only] = candidates
        return RankResult(
            picks=[RankedPick(tmdb_id=only["tmdb_id"], blurb="great fit")]
        )

    result = run(
        search(
            "mood",
            client=fake_tmdb.client(),
            watch_region="US",
            watch_providers=[8],
            interpret_model=ok_interpret,
            rank_model=fake_rank,
        )
    )

    [(title, blurb)] = result.picks
    assert title["title"] == "Fake Heist"
    assert blurb == "great fit"


def test_search_wraps_interpret_failure_as_catalog_tool_error() -> None:
    fake_tmdb = FakeTMDB()

    async def fake_interpret(message: str) -> DiscoverIntent:
        raise RuntimeError("anthropic timed out")

    async def fake_rank(message: str, candidates: list[Title]) -> RankResult:
        raise AssertionError("rank must not run once interpret() has failed")

    with pytest.raises(CatalogToolError):
        run(
            search(
                "something",
                client=fake_tmdb.client(),
                watch_region="US",
                watch_providers=[8],
                interpret_model=fake_interpret,
                rank_model=fake_rank,
            )
        )


def test_search_wraps_rank_failure_as_catalog_tool_error() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})

    async def fake_rank(message: str, candidates: list[Title]) -> RankResult:
        raise RuntimeError("model returned an unparseable response")

    with pytest.raises(CatalogToolError):
        run(
            search(
                "something",
                client=fake_tmdb.client(),
                watch_region="US",
                watch_providers=[8],
                interpret_model=ok_interpret,
                rank_model=fake_rank,
            )
        )


def test_search_lets_a_bad_watch_region_propagate_as_a_plain_value_error() -> None:
    """watch_region comes from the caller, never from the model — a malformed
    one is a caller bug to fix, not a case CatalogToolError should describe as
    'the model produced something invalid'."""
    fake_tmdb = FakeTMDB()

    with pytest.raises(ValueError) as excinfo:
        run(
            search(
                "something",
                client=fake_tmdb.client(),
                watch_region="usa",
                watch_providers=[8],
                interpret_model=ok_interpret,
                rank_model=_rank_must_not_run,
            )
        )
    assert not isinstance(excinfo.value, CatalogToolError)


def test_search_lets_tmdb_unavailable_propagate_unwrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tmdb, "_sleep", AsyncMock())
    fake_tmdb = FakeTMDB()
    fake_tmdb.queue("/discover/movie", *[httpx2.Response(429) for _ in range(3)])

    with pytest.raises(TMDBUnavailable):
        run(
            search(
                "something",
                client=fake_tmdb.client(),
                watch_region="US",
                watch_providers=[8],
                interpret_model=ok_interpret,
                rank_model=_rank_must_not_run,
            )
        )


def test_search_lets_tmdb_error_propagate_unwrapped() -> None:
    """A non-transient status (e.g. an expired TMDB token) is a service/config
    problem, not model-derived intent — it must not become CatalogToolError,
    which would misattribute the cause."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.queue("/discover/movie", httpx2.Response(422, json={"success": False}))

    with pytest.raises(tmdb.TMDBError) as excinfo:
        run(
            search(
                "something",
                client=fake_tmdb.client(),
                watch_region="US",
                watch_providers=[8],
                interpret_model=ok_interpret,
                rank_model=_rank_must_not_run,
            )
        )
    assert not isinstance(excinfo.value, tmdb.TMDBUnavailable)


def test_search_short_circuits_on_empty_watch_providers() -> None:
    fake_tmdb = FakeTMDB()
    called = False

    async def fake_interpret(message: str) -> DiscoverIntent:
        nonlocal called
        called = True
        return make_intent()

    async def fake_rank(message: str, candidates: list[Title]) -> RankResult:
        raise AssertionError("rank must not run")

    result = run(
        search(
            "something",
            client=fake_tmdb.client(),
            watch_region="US",
            watch_providers=[],
            interpret_model=fake_interpret,
            rank_model=fake_rank,
        )
    )

    assert result.intent is None
    assert result.picks == []
    assert result.relaxed == []
    assert not called
    assert fake_tmdb.requests == []


# --- relaxation ladder -----------------------------------------------------


async def _rank_first(message: str, candidates: list[Title]) -> RankResult:
    return RankResult(picks=[RankedPick(tmdb_id=candidates[0]["tmdb_id"], blurb="ok")])


def test_search_relaxes_runtime_when_the_first_query_is_empty() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": []})  # first pass: nothing
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})  # runtime dropped: hit

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(max_runtime_minutes=90)

    result = run(
        search(
            "something short",
            client=fake_tmdb.client(),
            watch_region="US",
            watch_providers=[8],
            interpret_model=fake_interpret,
            rank_model=_rank_first,
        )
    )

    calls = fake_tmdb.all_params_for("/discover/movie")
    assert len(calls) == 2
    assert calls[0]["with_runtime.lte"] == "90"
    assert "with_runtime.lte" not in calls[1]
    assert result.relaxed == ["runtime"]
    assert [title["title"] for title, _ in result.picks] == ["Fake Heist"]


def test_search_skips_a_rung_with_nothing_to_drop() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/search/keyword", {"results": [{"id": 42, "name": "cozy"}]})
    fake_tmdb.ok("/discover/movie", {"results": []})  # first pass
    fake_tmdb.ok("/discover/movie", {"results": []})  # runtime dropped
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})  # keywords dropped

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(max_runtime_minutes=90, keywords=["cozy"])

    result = run(
        search(
            "something cozy and short",
            client=fake_tmdb.client(),
            watch_region="US",
            watch_providers=[8],
            interpret_model=fake_interpret,
            rank_model=_rank_first,
        )
    )

    calls = fake_tmdb.all_params_for("/discover/movie")
    assert len(calls) == 3  # first, -runtime, -keywords; the year rung is skipped
    assert "with_runtime.lte" in calls[0] and "with_keywords" in calls[0]
    assert "with_runtime.lte" not in calls[1] and "with_keywords" in calls[1]
    assert "with_keywords" not in calls[2]
    assert result.relaxed == ["runtime", "keywords"]


def test_search_skips_keywords_rung_when_nothing_resolved() -> None:
    """A keyword the model invented (no TMDB match) never reaches the query,
    so dropping it can't change anything — the rung must not fire a wasted,
    identical retry or falsely report a constraint that was never applied."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/search/keyword", {"results": []})  # "cozy" never resolves
    fake_tmdb.ok("/discover/movie", {"results": []})  # the only call made

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(keywords=["cozy"])  # nothing else set to relax

    result = run(
        search(
            "something cozy",
            client=fake_tmdb.client(),
            watch_region="US",
            watch_providers=[8],
            interpret_model=fake_interpret,
            rank_model=_rank_must_not_run,
        )
    )

    calls = fake_tmdb.all_params_for("/discover/movie")
    assert len(calls) == 1
    assert result.relaxed == []
    assert result.picks == []


def test_search_never_relaxes_services_region_or_exclusions() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/search/keyword", {"results": [{"id": 42, "name": "cozy"}]})
    # every /discover/movie falls through to the fake's default empty result

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(
            max_runtime_minutes=90,
            release_year_gte=2015,
            keywords=["cozy"],
            without_genres=["horror"],
        )

    result = run(
        search(
            "something",
            client=fake_tmdb.client(),
            watch_region="GB",
            watch_providers=[8, 9],
            interpret_model=fake_interpret,
            rank_model=_rank_must_not_run,  # rank() short-circuits on no candidates
        )
    )

    calls = fake_tmdb.all_params_for("/discover/movie")
    assert len(calls) == 4  # first + three rungs, then the ladder is exhausted
    assert result.relaxed == ["runtime", "year", "keywords"]
    assert result.picks == []
    for params in calls:
        assert params["watch_region"] == "GB"
        assert params["with_watch_providers"] == "8|9"
        assert params["with_watch_monetization_types"] == "flatrate"
        assert params["vote_count.gte"] == str(tmdb.MIN_VOTE_COUNT)
        assert params["without_genres"] == "27"  # horror, from the frozen table


def test_search_does_not_relax_when_the_first_query_returns_results() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(max_runtime_minutes=90)

    result = run(
        search(
            "something",
            client=fake_tmdb.client(),
            watch_region="US",
            watch_providers=[8],
            interpret_model=fake_interpret,
            rank_model=_rank_first,
        )
    )

    calls = fake_tmdb.all_params_for("/discover/movie")
    assert len(calls) == 1
    assert calls[0]["with_runtime.lte"] == "90"  # kept — nothing was relaxed
    assert result.relaxed == []
