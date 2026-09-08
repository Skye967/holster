import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from catalog_tool import DiscoverIntent, RankedPick, RankResult
from main import (
    CORRELATION_ID_HEADER,
    CorrelationIdMiddleware,
    app,
    asgi_app,
    get_interpret_model,
    get_rank_model,
    get_tmdb_client,
    lifespan,
)
from testutil import MOVIE_A, FakeTMDB
from tmdb import Title

client = TestClient(asgi_app)


def test_health_ok() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_mints_correlation_id() -> None:
    resp = client.get("/health")
    assert resp.headers[CORRELATION_ID_HEADER]


def test_health_echoes_incoming_correlation_id() -> None:
    resp = client.get("/health", headers={CORRELATION_ID_HEADER: "corr-123"})
    assert resp.headers[CORRELATION_ID_HEADER] == "corr-123"


def test_correlation_id_survives_server_error() -> None:
    err_app = FastAPI()

    @err_app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("boom")

    err_client = TestClient(
        CorrelationIdMiddleware(err_app), raise_server_exceptions=False
    )
    resp = err_client.get("/boom", headers={CORRELATION_ID_HEADER: "trace-me"})
    assert resp.status_code == 500
    assert resp.headers[CORRELATION_ID_HEADER] == "trace-me"


# --- /chat -------------------------------------------------------------
#
# Real TMDBClient/model construction happens in lifespan(), which needs live
# keys (see the live-suite pattern in test_catalog_tool_live.py). These tests
# never trigger it: they override the Depends() seams directly, the same way
# TMDBClient's transport= and catalog_tool's Interpreter/Ranker do for
# everything else in this codebase.


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def test_chat_streams_ndjson_events() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok("/discover/movie", {"results": [MOVIE_A]})

    async def fake_interpret(message: str) -> DiscoverIntent:
        return DiscoverIntent(media_type="movie", keywords=["heist"])

    async def fake_rank(message: str, candidates: list[Title]) -> RankResult:
        pick = RankedPick(tmdb_id=candidates[0]["tmdb_id"], blurb="fits")
        return RankResult(picks=[pick])

    app.dependency_overrides[get_tmdb_client] = fake_tmdb.client
    app.dependency_overrides[get_interpret_model] = lambda: fake_interpret
    app.dependency_overrides[get_rank_model] = lambda: fake_rank

    resp = client.post(
        "/chat",
        json={"message": "a heist movie", "watch_region": "US", "watch_providers": [8]},
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/x-ndjson"
    lines = [json.loads(line) for line in resp.text.splitlines() if line]
    assert [e["type"] for e in lines] == ["intent", "results", "done"]


def test_chat_requires_a_message() -> None:
    # FastAPI resolves Depends() alongside body validation rather than only
    # after it succeeds, so the route's dependencies still need a value even
    # on a request this test expects to fail on the body — see main.py's
    # get_tmdb_client/get_interpret_model/get_rank_model.
    app.dependency_overrides[get_tmdb_client] = FakeTMDB().client
    app.dependency_overrides[get_interpret_model] = lambda: _must_not_be_called
    app.dependency_overrides[get_rank_model] = lambda: _must_not_be_called

    resp = client.post("/chat", json={"watch_region": "US", "watch_providers": [8]})

    assert resp.status_code == 422


# --- /providers ----------------------------------------------------------
#
# Feeds the gateway's streaming_providers cache refresh (TASKS.md T15). No
# LangChain/model dependency here, just the TMDB client seam.


def test_providers_merges_movie_and_tv() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok(
        "/watch/providers/movie",
        {
            "results": [
                {
                    "provider_id": 8,
                    "provider_name": "Netflix",
                    "logo_path": "/n.jpg",
                    "display_priorities": {"US": 1},
                }
            ]
        },
    )
    fake_tmdb.ok("/watch/providers/tv", {"results": []})
    app.dependency_overrides[get_tmdb_client] = fake_tmdb.client

    resp = client.get("/providers", params={"region": "US"})

    assert resp.status_code == 200
    assert resp.json() == [
        {
            "provider_id": 8,
            "provider_name": "Netflix",
            "logo_url": "https://image.tmdb.org/t/p/w92/n.jpg",
            "display_priority": 1,
        }
    ]


def test_providers_requires_region() -> None:
    app.dependency_overrides[get_tmdb_client] = FakeTMDB().client

    resp = client.get("/providers")

    assert resp.status_code == 422


async def _must_not_be_called(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("should not run — body validation must reject first")


# --- /titles ---------------------------------------------------------------
#
# The watchlist's read path (TASKS.md T18.5). No LLM dependency — same seam
# shape as /providers, just POST with a body of already-known ids.


def test_titles_enriches_known_ids() -> None:
    fake_tmdb = FakeTMDB()
    fake_tmdb.ok(
        "/movie/101",
        {"id": 101, "title": "Fake Heist", "release_date": "2020-01-01", "credits": {}},
    )
    fake_tmdb.ok("/movie/101/watch/providers", {"results": {}})
    app.dependency_overrides[get_tmdb_client] = fake_tmdb.client

    resp = client.post(
        "/titles",
        json={
            "watch_region": "US",
            "watch_providers": [8],
            "items": [{"tmdb_id": 101, "media_type": "movie"}],
        },
    )

    assert resp.status_code == 200
    [pick] = resp.json()
    assert pick["tmdb_id"] == 101
    assert pick["title"] == "Fake Heist"
    assert pick["blurb"] == ""
    assert pick["unavailable"] is False


def test_titles_requires_items() -> None:
    app.dependency_overrides[get_tmdb_client] = FakeTMDB().client

    resp = client.post("/titles", json={"watch_region": "US"})

    assert resp.status_code == 422


async def _run_lifespan(test_app: FastAPI) -> None:
    async with lifespan(test_app):
        pass


def test_lifespan_requires_google_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("TMDB_API_KEY", "t")
    with pytest.raises(SystemExit, match="GOOGLE_API_KEY"):
        asyncio.run(_run_lifespan(FastAPI(lifespan=lifespan)))


def test_lifespan_requires_tmdb_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "a")
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="TMDB_API_KEY"):
        asyncio.run(_run_lifespan(FastAPI(lifespan=lifespan)))
