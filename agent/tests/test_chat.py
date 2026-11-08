"""chat.py tests — offline, against the real catalog graph driven by a scripted
chat model, and a fake TMDB transport. What a turn does is test_turn.py's in the
library; these exercise the service's own: the NDJSON the gateway reads, the
request's bounds and parsing, the reason codes, and what a disconnect does.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx2
import pytest
from langchain_core.messages import AIMessage
from langchain_core.tracers.context import collect_runs
from langsmith.utils import tracing_is_enabled
from pydantic import ValidationError
from starlette.types import Message

import chat
import main
from chat import MAX_HISTORY_TURNS, ChatRequest, stream_chat
from holster_catalog import (
    SearchArgs,
    TitleRef,
    TitleVerdict,
    TMDBClient,
    Turn,
    build_turn,
    google_composer,
    graph,
    tmdb,
)
from holster_catalog.testing import (
    COMPOSED_TEXT,
    MOVIE_A,
    FailingChatModel,
    FakeTMDB,
    RecordingComposer,
    ScriptedChatModel,
    answer_call,
    composed_reply,
    lookup_call,
    ok_compose,
    rank_reply,
    raw_named,
    scripted,
    search_call,
)

# gateway's constant, hand-copied -- the two services share no code.
# Nothing enforces the copy: update it by hand when chat.go changes.
GATEWAY_MAX_HISTORY_EXCHANGES = 5


async def _collect(
    req: ChatRequest,
    client: TMDBClient,
    model: ScriptedChatModel,
    compose: Any = None,
    correlation_id: str | None = None,
) -> list[dict[str, Any]]:
    return [
        e
        async for e in stream_chat(
            req,
            client=client,
            agent=build_turn(model, compose or ok_compose),
            correlation_id=correlation_id,
        )
    ]


def test_no_providers_short_circuits_with_a_message() -> None:
    fake_tmdb = FakeTMDB()
    req = ChatRequest(message="anything", watch_region="US", watch_providers=[])
    model = scripted()

    composer = RecordingComposer()
    events = asyncio.run(_collect(req, fake_tmdb.client(), model, composer()))

    assert [e["type"] for e in events] == ["message", "done"]
    assert "Connections" in events[0]["text"]
    # No model call at all, which is the property gateway/chat.go's
    # guest throttle rests on: it meters a frame by the turn cap and the
    # per-IP limiter only when that frame carries providers.
    assert model.calls == []
    assert composer.calls == []
    assert fake_tmdb.requests == []


def test_intent_event_carries_every_search_field() -> None:
    """gateway/chat.go's agentIntent decodes all but sort_by and limit by name,
    and a field the event drops vanishes silently from the interpreting line."""
    fields = {
        "media_type": "tv",
        "keywords": ["heist"],
        "without_keywords": ["bleak"],
        "genres": ["drama"],
        "without_genres": ["comedy"],
        "cast": ["Al Pacino"],
        "crew": ["Michael Mann"],
        "max_runtime_minutes": 90,
        "release_year_gte": 1990,
        "release_year_lte": 1999,
        "sort_by": "vote_average.desc",
        "monetization": "rent",
        "limit": 5,
    }
    assert fields.keys() == SearchArgs.model_fields.keys() - {"reply"}
    req = ChatRequest(message="a heist show", watch_region="US", watch_providers=[8])

    events = asyncio.run(
        _collect(req, FakeTMDB().client(), scripted(search_call(**fields)))
    )

    assert events[0] == {"type": "intent", "intent": fields}


def test_tmdb_unavailable_maps_to_a_closed_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tmdb, "_sleep", AsyncMock())
    fake_tmdb = FakeTMDB()
    fake_tmdb.queue("/discover/movie", *[httpx2.Response(503) for _ in range(3)])
    req = ChatRequest(message="something", watch_region="US", watch_providers=[8])
    model = scripted(search_call())

    events = asyncio.run(_collect(req, fake_tmdb.client(), model))

    # error is terminal on its own — no trailing "done" (a turn ends with
    # exactly one of the two, never both; see README.md's Chat pipeline).
    assert [e["type"] for e in events] == ["intent", "error"]
    assert events[1]["reason"] == "tmdb_unavailable"
    # The search node's exception ends the turn; no rank call follows it.
    assert len(model.calls) == 1


def test_a_model_failure_on_call_one_maps_to_model_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A provider error (a rate limit, a timeout) is the model being
    unavailable, not a bug: no traceback at error level, and no second call."""
    fake_tmdb = FakeTMDB()
    req = ChatRequest(message="something", watch_region="US", watch_providers=[8])
    model = FailingChatModel(messages=iter([]))

    with caplog.at_level(logging.INFO):
        events = asyncio.run(_collect(req, fake_tmdb.client(), model))

    assert [e["type"] for e in events] == ["error"]
    assert events[0]["reason"] == "model_unavailable"
    assert len(model.calls) == 1
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_a_model_failure_on_the_rank_call_maps_to_model_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A rank reply that does not parse as RankResult — or a provider error on
    that call — is the model being unavailable too, not a bug."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})
    req = ChatRequest(message="something", watch_region="US", watch_providers=[8])
    model = scripted(search_call(), AIMessage(content="not json"))

    with caplog.at_level(logging.INFO):
        events = asyncio.run(_collect(req, fake_tmdb.client(), model))

    assert [e["type"] for e in events] == ["intent", "error"]
    assert events[1]["reason"] == "model_unavailable"
    assert len(model.calls) == 2
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_bad_watch_region_maps_to_internal_not_tmdb_specific() -> None:
    """watch_region is caller-supplied, never model-derived (catalog_tool.py) —
    a malformed one is a gateway/caller bug, not 'try later'."""
    fake_tmdb = FakeTMDB()
    req = ChatRequest(message="something", watch_region="usa", watch_providers=[8])

    events = asyncio.run(_collect(req, fake_tmdb.client(), scripted(search_call())))

    # The intent is already out by the time discover() validates watch_region
    # — the interpret node writes it before the search node runs.
    assert [e["type"] for e in events] == ["intent", "error"]
    assert events[1]["reason"] == "internal"


def test_only_the_newest_turns_reach_the_agent() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": []})
    turns = [
        Turn(role="user" if i % 2 == 0 else "assistant", text=f"turn {i}")
        for i in range(MAX_HISTORY_TURNS + 2)
    ]
    req = ChatRequest(
        message="anything", watch_region="US", watch_providers=[8], history=turns
    )
    model = scripted(search_call())

    asyncio.run(_collect(req, fake_tmdb.client(), model))

    sent = model.calls[0]
    assert [m.text for m in sent[1:-1]] == [t.text for t in turns[-MAX_HISTORY_TURNS:]]


def test_a_disconnect_ends_the_response_and_starts_no_later_step() -> None:
    """main.py's StreamingResponse, with the client gone while the search node's
    discover() is in flight — see chat.py's module docstring on cancellation.
    The discover() is released after the response ends so that the rank call
    could start, if anything were still driving the graph."""
    req = ChatRequest(message="something", watch_region="US", watch_providers=[8])

    async def run() -> None:
        started, release = asyncio.Event(), asyncio.Event()
        sent: list[Message] = []

        async def hanging(request: httpx2.Request) -> httpx2.Response:
            started.set()
            await release.wait()
            return httpx2.Response(200, json={"results": [MOVIE_A]})

        async def disconnect() -> dict[str, str]:
            await started.wait()
            return {"type": "http.disconnect"}

        async def send(message: Message) -> None:
            sent.append(message)

        model = scripted(search_call(), rank_reply(101))
        response = await main.chat(
            req,
            tmdb_client=TMDBClient(
                "test-token", transport=httpx2.MockTransport(hanging)
            ),
            agent=build_turn(model, ok_compose),
        )
        # What uvicorn reports; below 2.4 Starlette cancels on a disconnect.
        scope = {"type": "http", "asgi": {"spec_version": "2.3"}}
        await response(scope, disconnect, send)

        release.set()
        while others := asyncio.all_tasks() - {asyncio.current_task()}:
            await asyncio.wait(others)
        assert len(model.calls) == 1
        assert [json.loads(m["body"])["type"] for m in sent[1:]] == ["intent"]

    asyncio.run(asyncio.wait_for(run(), timeout=2))


def test_a_close_at_a_yield_closes_the_stream() -> None:
    """A close at a yield in stream_chat, which is what contextlib.aclosing is
    for: the graph's stream is closed by the time the close returns."""
    req = ChatRequest(message="something", watch_region="US", watch_providers=[8])
    turn = build_turn(scripted(search_call(), rank_reply(101)), ok_compose)
    streams: list[AsyncGenerator[Any]] = []

    def astream(*args: Any, **kwargs: Any) -> AsyncGenerator[Any]:
        streams.append(cast(AsyncGenerator[Any], turn.astream(*args, **kwargs)))
        return streams[-1]

    async def run() -> None:
        gen = stream_chat(
            req,
            client=FakeTMDB().client(),
            agent=cast(Any, SimpleNamespace(astream=astream)),
            correlation_id=None,
        )
        assert (await anext(gen))["type"] == "intent"
        await gen.aclose()
        assert inspect.getasyncgenstate(streams[0]) == inspect.AGEN_CLOSED

    asyncio.run(run())


def test_chat_request_parses_the_gateways_verdict_json_verbatim() -> None:
    """The one place the Go->Python verdict contract is checked. The gateway
    marshals its own Verdict struct (gateway/verdicts.go), which is
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
    guarantees that, and gateway/verdicts_test.go pins it there."""
    req = ChatRequest.model_validate(
        {"message": "hi", "watch_region": "US", "watch_providers": [8]}
    )
    assert req.verdicts == []


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


def test_an_oversized_shown_set_is_cut_to_the_newest_and_says_so(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Nothing upstream bounds `shown` any more, so if this ever fires the
    gateway is suppressing titles the search will not exclude. The oldest go,
    which is the wrong end — hence the warning: it is the only symptom."""
    monkeypatch.setattr(chat, "MAX_SHOWN", 2)
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})  # MOVIE_A is id 101
    req = ChatRequest(
        message="a heist movie",
        watch_region="US",
        watch_providers=[8],
        # 101 is the oldest, so the cut makes it eligible again.
        shown=[TitleRef(tmdb_id=i, media_type="movie") for i in (101, 900, 901)],
    )
    model = scripted(search_call(), rank_reply(101))

    with caplog.at_level("WARNING"):
        events = asyncio.run(_collect(req, fake_tmdb.client(), model))

    assert [e["type"] for e in events] == ["intent", "results", "done"]
    assert [p["tmdb_id"] for p in events[1]["picks"]] == [101]
    assert "shown set truncated" in caplog.text


@pytest.mark.parametrize(
    ("turn", "cause"),
    [
        (
            {
                "verdicts": [
                    TitleVerdict(tmdb_id=101, media_type="movie", verdict="seen")
                ]
            },
            "all_judged",
        ),
        ({"shown": [TitleRef(tmdb_id=101, media_type="movie")]}, "all_shown"),
    ],
)
def test_verdicts_and_shown_reach_the_search(turn: dict[str, Any], cause: str) -> None:
    """stream_chat copies each into the run context, and the library's tests build
    that context themselves."""
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})  # MOVIE_A is id 101
    req = ChatRequest(
        message="a heist movie", watch_region="US", watch_providers=[8], **turn
    )
    composer = RecordingComposer()

    asyncio.run(_collect(req, fake_tmdb.client(), scripted(search_call()), composer()))

    assert composer.causes == [cause]


def test_the_request_reaches_the_turn() -> None:
    fake_tmdb = FakeTMDB()
    req = ChatRequest(
        message="a heist movie",
        watch_region="US",
        watch_providers=[337, 350],
        watch_provider_names=["Disney Plus"],
    )
    model = scripted(search_call())

    asyncio.run(_collect(req, fake_tmdb.client(), model))

    assert fake_tmdb.params_for("/discover/movie")["with_watch_providers"] == "337|350"
    assert "Disney Plus" in model.calls[0][0].text
    assert model.calls[0][-1].text == "a heist movie"


def test_the_history_backstop_stays_clear_of_the_gateways_window() -> None:
    """Pins MAX_HISTORY_TURNS against the gateway's maxHistoryExchanges.

    A backstop sitting at the operating point stops being a backstop and
    becomes a second, tighter window — the gateway would send five exchanges
    and the agent would quietly keep fewer, with no error anywhere."""
    gateway_window = GATEWAY_MAX_HISTORY_EXCHANGES * 2
    assert gateway_window <= MAX_HISTORY_TURNS, (
        f"MAX_HISTORY_TURNS {MAX_HISTORY_TURNS} is under the gateway's "
        f"{gateway_window}-message window; history would be cut here with no "
        f"error anywhere"
    )


def test_every_run_of_a_turn_carries_its_correlation_id() -> None:
    """The composer's call inherits the id from the graph step it runs in. A real
    google_composer, because a plain async Composer starts no run to see."""
    fake_tmdb = FakeTMDB()
    req = ChatRequest(message="a heist movie", watch_region="US", watch_providers=[8])
    model = scripted(search_call())
    composer = google_composer(scripted(composed_reply(COMPOSED_TEXT)))

    with collect_runs() as runs:
        events = asyncio.run(
            _collect(req, fake_tmdb.client(), model, composer, correlation_id="c-1")
        )

    assert events[-2] == {"type": "message", "text": COMPOSED_TEXT}
    traced = runs.traced_runs
    assert [r.name for r in traced].count("chat_turn") == 1
    assert any(
        "ChatPromptTemplate" in [child.name for child in r.child_runs] for r in traced
    )
    assert all(r.metadata["correlation_id"] == "c-1" for r in traced)
    assert not tracing_is_enabled()


# --- The wire ----------------------------------------------------------------
#
# Each list is the events the gateway's agentEvent decodes, written as literals so
# an edit to the mapping cannot also edit what it is held to.

_PICK = {
    "tmdb_id": 101,
    "media_type": "movie",
    "title": "Fake Heist",
    "year": 2020,
    "overview": "A heist movie.",
    "poster_url": None,
    "vote_average": 7.5,
    "vote_count": 500,
    "genre_ids": [80],
    "genre_names": ["Crime"],
    "runtime_minutes": None,
    "cast": [],
    "available_on": [],
    "rent_on": [],
    "watch_link": None,
    "blurb": "fits",
}
_INTENT = {
    "media_type": "movie",
    "keywords": [],
    "without_keywords": [],
    "genres": [],
    "without_genres": [],
    "cast": [],
    "crew": [],
    "sort_by": "vote_count.desc",
    "monetization": "subscription",
    "limit": 10,
}


def _wire_turn(model: ScriptedChatModel, fake_tmdb: FakeTMDB) -> list[dict[str, Any]]:
    req = ChatRequest(message="a heist movie", watch_region="US", watch_providers=[8])
    return asyncio.run(_collect(req, fake_tmdb.client(), model))


def test_a_search_turn_on_the_wire() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})
    model = scripted(search_call(keywords=["heist"], reply="On it."), rank_reply(101))

    assert _wire_turn(model, fake_tmdb) == [
        {"type": "reply", "text": "On it."},
        {"type": "intent", "intent": {**_INTENT, "keywords": ["heist"]}},
        {
            "type": "results",
            "kind": "search",
            "relaxed": [],
            "note": "",
            "picks": [_PICK],
        },
        {"type": "done"},
    ]


def test_a_widened_search_on_the_wire() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": []})
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})
    model = scripted(search_call(max_runtime_minutes=90), rank_reply(101))

    assert _wire_turn(model, fake_tmdb) == [
        {"type": "intent", "intent": {**_INTENT, "max_runtime_minutes": 90}},
        {
            "type": "results",
            "kind": "search",
            "relaxed": ["runtime"],
            "note": "Nothing left that matches exactly, so I widened past how long.",
            "picks": [_PICK],
        },
        {"type": "done"},
    ]


def test_a_lookup_on_the_wire() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/search/multi", {"results": [raw_named(101, "Heat", 500)]})
    model = scripted(lookup_call("Heat", reply="Looking."))
    pick = {
        **_PICK,
        "title": "Heat",
        "overview": "An overview.",
        "genre_ids": [],
        "genre_names": [],
        "blurb": "",
    }

    assert _wire_turn(model, fake_tmdb) == [
        {"type": "reply", "text": "Looking."},
        {"type": "intent", "intent": {"title": "Heat"}},
        {
            "type": "results",
            "kind": "lookup",
            "relaxed": [],
            "note": "",
            "picks": [pick],
        },
        {"type": "done"},
    ]


def test_an_answer_on_the_wire() -> None:
    assert _wire_turn(scripted(answer_call("Hi there.", "chat")), FakeTMDB()) == [
        {"type": "message", "text": "Hi there."},
        {"type": "done"},
    ]


def test_a_turn_with_nothing_found_on_the_wire() -> None:
    assert _wire_turn(scripted(search_call()), FakeTMDB()) == [
        {"type": "intent", "intent": _INTENT},
        {"type": "message", "text": COMPOSED_TEXT},
        {"type": "done"},
    ]


def test_a_title_not_found_on_the_wire() -> None:
    assert _wire_turn(scripted(lookup_call("Zzz")), FakeTMDB()) == [
        {"type": "intent", "intent": {"title": "Zzz"}},
        {"type": "message", "text": COMPOSED_TEXT},
        {"type": "done"},
    ]


def test_a_refused_model_reply_on_the_wire() -> None:
    assert _wire_turn(scripted(AIMessage(content="no tool")), FakeTMDB()) == [
        {"type": "error", "reason": "model_unavailable"}
    ]


def test_a_tmdb_failure_in_the_finish_step_is_an_error_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """finish_search runs in the graph, so its failure reaches the same except
    every other step's does."""

    async def unavailable(*args: Any, **kwargs: Any) -> None:
        raise tmdb.TMDBUnavailable("down")

    monkeypatch.setattr(graph, "finish_search", unavailable)
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})

    events = _wire_turn(scripted(search_call(), rank_reply(101)), fake_tmdb)

    assert events[-1] == {"type": "error", "reason": "tmdb_unavailable"}
