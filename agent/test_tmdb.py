"""TMDB client tests — offline, against httpx2.MockTransport.

The live path (real token, real HTTP) is in test_tmdb_live.py, which skips
without TMDB_API_KEY — same shape as the gateway's TEST_DATABASE_URL tests.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import httpx2
import pytest

import tmdb
from tmdb import MIN_VOTE_COUNT, TMDBClient, TMDBError, TMDBUnavailable


def run(coro: Any) -> Any:
    return asyncio.run(coro)


class FakeTMDB:
    """Records every outgoing request and replays queued responses by path.

    A path with a queue pops one response per call (so a 429 then a 200 tests a
    retry); a path with none gets an empty result set.
    """

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self._queues: dict[str, list[httpx2.Response]] = {}

    def queue(self, path: str, *responses: httpx2.Response) -> None:
        self._queues.setdefault("/3" + path, []).extend(responses)

    def ok(self, path: str, payload: dict[str, Any]) -> None:
        self.queue(path, httpx2.Response(200, json=payload))

    def _handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        queue = self._queues.get(request.url.path)
        if queue:
            return queue.pop(0)
        return httpx2.Response(200, json={"results": []})

    def client(self) -> TMDBClient:
        return TMDBClient("test-token", transport=httpx2.MockTransport(self._handler))

    def params_for(self, path: str) -> dict[str, str]:
        for request in self.requests:
            if request.url.path == "/3" + path:
                return dict(request.url.params)
        raise AssertionError(
            f"no request to {path}: {[r.url.path for r in self.requests]}"
        )

    def count(self, path: str) -> int:
        return sum(1 for r in self.requests if r.url.path == "/3" + path)


PERSON = {"results": [{"id": 3063, "name": "Tilda Swinton"}]}
KEYWORD = {"results": [{"id": 9748, "name": "heist"}]}


def test_discover_resolves_names_to_ids() -> None:
    fake = FakeTMDB()
    fake.ok("/search/person", PERSON)
    fake.ok("/search/keyword", KEYWORD)

    run(
        fake.client().discover(
            media_type="movie",
            watch_region="US",
            watch_providers=[8, 9],
            cast=["Tilda Swinton"],
            keywords=["heist"],
            genres=["Comedy"],
        )
    )

    params = fake.params_for("/discover/movie")
    assert params["with_cast"] == "3063"
    assert params["with_keywords"] == "9748"
    assert params["with_genres"] == "35"  # Comedy, from the frozen table
    assert params["with_watch_providers"] == "8|9"
    assert params["with_watch_monetization_types"] == "flatrate"
    assert params["watch_region"] == "US"


def test_vote_count_floor_always_applied() -> None:
    fake = FakeTMDB()
    run(fake.client().discover(media_type="movie", watch_region="GB"))
    assert fake.params_for("/discover/movie")["vote_count.gte"] == str(MIN_VOTE_COUNT)


def test_watch_region_is_required() -> None:
    with pytest.raises(TypeError):
        run(FakeTMDB().client().discover(media_type="movie"))  # type: ignore[call-arg]


def test_bad_watch_region_rejected() -> None:
    with pytest.raises(ValueError):
        run(FakeTMDB().client().discover(media_type="movie", watch_region="usa"))


def test_empty_watch_providers_rejected() -> None:
    with pytest.raises(ValueError):
        run(
            FakeTMDB()
            .client()
            .discover(media_type="movie", watch_region="US", watch_providers=[])
        )


def test_bad_media_type_rejected() -> None:
    with pytest.raises(ValueError):
        run(FakeTMDB().client().discover(media_type="film", watch_region="US"))  # type: ignore[arg-type]


def test_unresolved_name_is_dropped_not_fatal() -> None:
    fake = FakeTMDB()
    fake.ok("/search/person", {"results": []})

    titles = run(
        fake.client().discover(
            media_type="movie", watch_region="US", cast=["Nobody McNoone"]
        )
    )

    assert titles == []
    assert "with_cast" not in fake.params_for("/discover/movie")


def test_resolution_is_cached_across_calls() -> None:
    fake = FakeTMDB()
    fake.ok("/search/person", PERSON)

    client = fake.client()
    run(client.discover(media_type="movie", watch_region="US", cast=["Tilda Swinton"]))
    run(client.discover(media_type="tv", watch_region="US", crew=["tilda swinton"]))

    assert fake.count("/search/person") == 1


def test_year_range_becomes_date_bounds() -> None:
    fake = FakeTMDB()
    run(
        fake.client().discover(
            media_type="movie",
            watch_region="US",
            release_year_gte=1990,
            release_year_lte=1999,
        )
    )
    params = fake.params_for("/discover/movie")
    assert params["primary_release_date.gte"] == "1990-01-01"
    assert params["primary_release_date.lte"] == "1999-12-31"


def test_discover_trims_results() -> None:
    fake = FakeTMDB()
    fake.queue(
        "/discover/movie",
        httpx2.Response(
            200,
            json={
                "results": [
                    {
                        "id": 27205,
                        "title": "Inception",
                        "release_date": "2010-07-15",
                        "overview": "A thief who steals corporate secrets.",
                        "poster_path": "/poster.jpg",
                        "vote_average": 8.4,
                        "vote_count": 34000,
                        "genre_ids": [28, 878, 12],
                    },
                    {
                        "id": 999,
                        "title": "No Poster",
                        "release_date": "",
                        "poster_path": None,
                        "vote_average": 5.0,
                        "vote_count": 250,
                    },
                ]
            },
        ),
    )

    titles = run(fake.client().discover(media_type="movie", watch_region="US"))

    assert titles[0] == {
        "tmdb_id": 27205,
        "media_type": "movie",
        "title": "Inception",
        "year": 2010,
        "overview": "A thief who steals corporate secrets.",
        "poster_url": "https://image.tmdb.org/t/p/w342/poster.jpg",
        "vote_average": 8.4,
        "vote_count": 34000,
        "genre_ids": [28, 878, 12],
    }
    assert titles[1]["year"] is None
    assert titles[1]["poster_url"] is None


def test_watch_providers_splits_monetization() -> None:
    fake = FakeTMDB()
    fake.ok(
        "/movie/438631/watch/providers",
        {
            "id": 438631,
            "results": {
                "US": {
                    "link": "https://www.themoviedb.org/movie/438631/watch",
                    "flatrate": [
                        {
                            "provider_id": 8,
                            "provider_name": "Netflix",
                            "logo_path": "/n.jpg",
                        }
                    ],
                    "rent": [
                        {
                            "provider_id": 3,
                            "provider_name": "Apple TV",
                            "logo_path": None,
                        }
                    ],
                },
                "GB": {"link": "https://example.test/gb", "flatrate": []},
            },
        },
    )

    avail = run(
        fake.client().watch_providers(
            media_type="movie", tmdb_id=438631, watch_region="US"
        )
    )

    assert avail["link"] == "https://www.themoviedb.org/movie/438631/watch"
    assert avail["flatrate"] == [
        {
            "provider_id": 8,
            "provider_name": "Netflix",
            "logo_url": "https://image.tmdb.org/t/p/w92/n.jpg",
        }
    ]
    assert avail["rent"][0]["logo_url"] is None
    assert avail["buy"] == []


def test_watch_providers_unknown_region_is_empty() -> None:
    fake = FakeTMDB()
    fake.ok("/movie/1/watch/providers", {"id": 1, "results": {}})
    avail = run(
        fake.client().watch_providers(media_type="movie", tmdb_id=1, watch_region="US")
    )
    assert avail == {"link": None, "flatrate": [], "rent": [], "buy": []}


def test_watch_providers_404_is_empty() -> None:
    fake = FakeTMDB()
    fake.queue(
        "/movie/1/watch/providers", httpx2.Response(404, json={"success": False})
    )
    avail = run(
        fake.client().watch_providers(media_type="movie", tmdb_id=1, watch_region="US")
    )
    assert avail["flatrate"] == []


def test_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tmdb, "_sleep", AsyncMock())
    fake = FakeTMDB()
    fake.queue(
        "/discover/movie",
        httpx2.Response(429),
        httpx2.Response(503),
        httpx2.Response(200, json={"results": []}),
    )
    assert run(fake.client().discover(media_type="movie", watch_region="US")) == []
    assert fake.count("/discover/movie") == 3


def test_retries_exhausted_raises_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tmdb, "_sleep", AsyncMock())
    fake = FakeTMDB()
    fake.queue("/discover/movie", *[httpx2.Response(429) for _ in range(3)])
    with pytest.raises(TMDBUnavailable):
        run(fake.client().discover(media_type="movie", watch_region="US"))


def test_client_error_is_a_bug_not_transient() -> None:
    fake = FakeTMDB()
    fake.queue("/discover/movie", httpx2.Response(422, json={"success": False}))
    with pytest.raises(TMDBError) as excinfo:
        run(fake.client().discover(media_type="movie", watch_region="US"))
    assert not isinstance(excinfo.value, TMDBUnavailable)


def test_non_json_body_is_unavailable_not_a_crash() -> None:
    fake = FakeTMDB()
    fake.queue("/discover/movie", httpx2.Response(200, content=b"<html>down</html>"))
    with pytest.raises(TMDBUnavailable):
        run(fake.client().discover(media_type="movie", watch_region="US"))


def test_search_titles_merges_movie_and_tv() -> None:
    fake = FakeTMDB()
    fake.ok(
        "/search/movie", {"results": [{"id": 1, "title": "Dune", "vote_count": 100}]}
    )
    fake.ok(
        "/search/tv",
        {"results": [{"id": 2, "name": "Dune: Prophecy", "vote_count": 500}]},
    )

    titles = run(fake.client().search_titles("dune"))

    assert [t["tmdb_id"] for t in titles] == [2, 1]  # ordered by vote_count
    assert titles[0]["media_type"] == "tv"
    assert titles[1]["media_type"] == "movie"
