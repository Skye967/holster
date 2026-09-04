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
from pydantic import ValidationError

import tmdb
from catalog_tool import DiscoverIntent, RankedPick, RankResult, TitleVerdict
from chat import (
    ChatRequest,
    HistoryTurn,
    _nothing_found_message,
    stream_chat,
)
from testutil import (
    ALL_JUDGED,
    CAPABILITY_QUESTION,
    MOVIE_A,
    NO_CANDIDATES,
    FakeTMDB,
    make_intent,
    ok_interpret,
    rank_must_not_run,
)
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


def test_capability_question_short_circuits_with_a_message() -> None:
    fake_tmdb = FakeTMDB()
    req = ChatRequest(
        message="what can you do?",
        watch_region="US",
        watch_providers=[8],
        watch_provider_names=["Netflix", "Hulu"],
    )

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(is_capability_question=True)

    events = asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            fake_interpret,
            rank_must_not_run(CAPABILITY_QUESTION),
        )
    )

    # No "intent" event either — on_intent never fires for a capability
    # question (catalog_tool.py's search()), so there is nothing to
    # translate into an interpreting line.
    assert [e["type"] for e in events] == ["message", "done"]
    # "Netflix and Hulu", not "Netflix, Hulu" — matches services/gateway/
    # chat.go's humanJoin, used for the same provider list elsewhere in the
    # same conversation (the interpreting line).
    assert "Netflix and Hulu" in events[0]["text"]
    assert fake_tmdb.requests == []


def test_capability_question_degrades_gracefully_with_uncached_provider_names() -> (
    None
):
    """watch_providers non-empty but watch_provider_names empty is a real,
    reachable state (loadChatContext's per-country name cache hasn't warmed
    up yet — see chat.go) — must not be misread as "no services picked"."""
    fake_tmdb = FakeTMDB()
    req = ChatRequest(
        message="what can you do?",
        watch_region="US",
        watch_providers=[8],
        watch_provider_names=[],
    )

    async def fake_interpret(message: str) -> DiscoverIntent:
        return make_intent(is_capability_question=True)

    events = asyncio.run(
        _collect(
            req,
            fake_tmdb.client(),
            fake_interpret,
            rank_must_not_run(CAPABILITY_QUESTION),
        )
    )

    assert [e["type"] for e in events] == ["message", "done"]
    assert "Connections" not in events[0]["text"]
    assert "haven't picked" not in events[0]["text"]


def test_successful_search_emits_intent_then_results_then_done() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})  # MOVIE_A is id 101
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
    # Enrichment (TASKS.md T16) reaches the wire event: genre_names is a pure
    # local lookup from MOVIE_A's genre_ids ([80] -> Crime), no TMDB call
    # needed; the other three fields come back empty because this test's
    # FakeTMDB has no /movie/101* endpoints queued, so title_details()/
    # watch_providers() succeed with a benign empty response — not the
    # None-on-failure degrade path, which test_catalog_tool.py covers
    # separately. This just confirms the fields reach the wire.
    assert pick["genre_names"] == ["Crime"]
    assert pick["runtime_minutes"] is None
    assert pick["cast"] == []
    assert pick["available_on"] == []


def test_intent_event_arrives_before_rank_is_called() -> None:
    """The whole reason for on_intent: the caller must be able to observe it
    before the (slower) rank step runs."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})  # MOVIE_A is id 101
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

    events = asyncio.run(
        _collect(
            req, fake_tmdb.client(), fake_interpret, rank_must_not_run(NO_CANDIDATES)
        )
    )

    assert [e["type"] for e in events] == ["intent", "message", "done"]
    # "how long and the mood", not "how long, the mood" — _human_join, same
    # as the capability answer's provider-name grammar.
    assert "how long and the mood" in events[1]["text"]


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
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})  # MOVIE_A is id 101
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


def test_chat_request_parses_the_gateways_verdict_json_verbatim() -> None:
    """The one place the Go->Python verdict contract is checked. The gateway
    marshals its own Verdict struct (services/gateway/verdicts.go), which is
    also the GET /api/verdicts response body and is mirrored a third time in
    web/src/lib/verdicts.ts -- so a rename made to suit the browser would
    otherwise reach production as a 422 on every chat turn, with the Go and
    Python suites both still green. This asserts the exact JSON tag names.
    """
    body = {
        "message": "something good",
        "watch_region": "US",
        "watch_providers": [8],
        # Verbatim from the Go struct's json tags.
        "verdicts": [
            {"tmdb_id": 101, "media_type": "movie", "verdict": "seen"},
            {"tmdb_id": 202, "media_type": "tv", "verdict": "liked"},
        ],
    }

    req = ChatRequest.model_validate(body)

    assert [(v.tmdb_id, v.media_type, v.verdict) for v in req.verdicts] == [
        (101, "movie", "seen"),
        (202, "tv", "liked"),
    ]


def test_chat_request_defaults_verdicts_when_the_gateway_omits_them() -> None:
    """agentChatRequest tags Verdicts omitempty, so a caller with no verdicts
    sends no key at all. Validated from a body because that is the shape
    omitempty produces — the key absent rather than null.
    The key must be absent, not null; omitempty on the Go side is what
    guarantees that, and services/gateway/verdicts_test.go pins it there."""
    req = ChatRequest.model_validate(
        {"message": "hi", "watch_region": "US", "watch_providers": [8]}
    )
    assert req.verdicts == []


def test_all_judged_message_replaces_the_loosening_advice() -> None:
    """all_judged wins over relaxed: if the ladder ran and the survivors were
    all already rated, "even after loosening how long" names the wrong cause."""
    assert "rated all of them" in _nothing_found_message([], all_judged=True)
    assert "rated all of them" in _nothing_found_message(["runtime"], all_judged=True)
    assert "loosening" in _nothing_found_message([], all_judged=False)
    assert "how long" in _nothing_found_message(["runtime"], all_judged=False)


def test_chat_request_rejects_an_unknown_media_type() -> None:
    """TitleVerdict types media_type as a Literal while verdict is a plain str,
    and the docstring justifies the split at length - this is what holds the
    Literal half in place. A bad media_type must not reach the agent: it would
    never match an excluded key (so the title silently stops being excluded),
    and on a liked row _validate_media_type's ValueError isn't a TMDBError, so
    nothing on the taste path catches it - it fails the whole turn with a
    traceback, not quietly."""
    body = {
        "message": "hi",
        "watch_region": "US",
        "watch_providers": [8],
        "verdicts": [{"tmdb_id": 101, "media_type": "film", "verdict": "seen"}],
    }

    with pytest.raises(ValidationError):
        ChatRequest.model_validate(body)


def test_verdicts_reach_search_and_shape_the_reply() -> None:
    """The seam. stream_chat passing req.verdicts to search(), and passing
    result.all_judged on to the copy, are each one line, and deleting either
    leaves every other test in this repo green - the feature would silently
    stop excluding anything while three suites stayed passing. This drives the
    whole path: a verdict on the only candidate must produce the already-rated
    message rather than results."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})  # MOVIE_A is id 101
    req = ChatRequest(
        message="a heist movie",
        watch_region="US",
        watch_providers=[8],
        verdicts=[TitleVerdict(tmdb_id=101, media_type="movie", verdict="seen")],
    )

    events = asyncio.run(
        _collect(req, fake_tmdb.client(), ok_interpret, rank_must_not_run(ALL_JUDGED))
    )

    assert [e["type"] for e in events] == ["intent", "message", "done"]
    assert "rated all of them" in events[1]["text"]
