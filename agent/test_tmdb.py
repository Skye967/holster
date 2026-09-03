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
from testutil import FakeTMDB
from tmdb import MIN_VOTE_COUNT, TMDBError, TMDBUnavailable


def run(coro: Any) -> Any:
    return asyncio.run(coro)


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


def test_is_keyword_resolved_reflects_the_cache() -> None:
    fake = FakeTMDB()
    fake.ok("/search/keyword", KEYWORD)

    client = fake.client()
    assert client.is_keyword_resolved("heist") is False  # not looked up yet

    run(client.discover(media_type="movie", watch_region="US", keywords=["heist"]))

    assert client.is_keyword_resolved("heist") is True
    assert client.is_keyword_resolved("HEIST") is True  # normalized like the cache key
    assert client.is_keyword_resolved("nonexistent") is False


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


def test_title_details_movie_returns_runtime_and_cast() -> None:
    fake = FakeTMDB()
    fake.ok(
        "/movie/101",
        {
            "runtime": 128,
            "credits": {
                "cast": [
                    {"name": "Third Billed", "order": 2},
                    {"name": "Star", "order": 0},
                    {"name": "Second Billed", "order": 1},
                    {"name": "Uncredited Extra", "order": 3},
                ]
            },
        },
    )

    details = run(fake.client().title_details(media_type="movie", tmdb_id=101))

    assert fake.params_for("/movie/101")["append_to_response"] == "credits"
    assert details == {
        "runtime_minutes": 128,
        "cast": ["Star", "Second Billed", "Third Billed"],
    }


def test_title_details_tv_uses_first_episode_run_time() -> None:
    fake = FakeTMDB()
    fake.ok("/tv/9", {"episode_run_time": [45, 60], "credits": {"cast": []}})

    details = run(fake.client().title_details(media_type="tv", tmdb_id=9))

    assert details["runtime_minutes"] == 45


def test_title_details_tv_empty_episode_run_time_is_none() -> None:
    fake = FakeTMDB()
    fake.ok("/tv/9", {"episode_run_time": [], "credits": {"cast": []}})

    details = run(fake.client().title_details(media_type="tv", tmdb_id=9))

    assert details["runtime_minutes"] is None


def test_title_details_404_returns_none_and_is_cached() -> None:
    fake = FakeTMDB()
    fake.queue("/movie/1", httpx2.Response(404, json={"success": False}))

    client = fake.client()
    first = run(client.title_details(media_type="movie", tmdb_id=1))
    second = run(client.title_details(media_type="movie", tmdb_id=1))

    assert first is None
    assert second is None
    assert fake.count("/movie/1") == 1


def test_title_details_is_cached_across_calls() -> None:
    fake = FakeTMDB()
    fake.ok("/movie/101", {"runtime": 100, "credits": {"cast": []}})

    client = fake.client()
    run(client.title_details(media_type="movie", tmdb_id=101))
    run(client.title_details(media_type="movie", tmdb_id=101))

    assert fake.count("/movie/101") == 1


def test_title_details_bad_media_type_rejected() -> None:
    with pytest.raises(ValueError):
        run(FakeTMDB().client().title_details(media_type="film", tmdb_id=1))  # type: ignore[arg-type]


def test_title_with_details_one_call_returns_title_and_details() -> None:
    fake = FakeTMDB()
    fake.ok(
        "/movie/101",
        {
            "id": 101,
            "title": "Fake Heist",
            "release_date": "2020-01-01",
            "overview": "A heist movie.",
            "poster_path": "/poster.jpg",
            "vote_average": 7.5,
            "vote_count": 500,
            "genres": [{"id": 80, "name": "Crime"}, {"id": 28, "name": "Action"}],
            "runtime": 128,
            "credits": {"cast": [{"name": "Star", "order": 0}]},
        },
    )

    result = run(fake.client().title_with_details(media_type="movie", tmdb_id=101))

    assert result is not None
    title, details = result
    assert title == {
        "tmdb_id": 101,
        "media_type": "movie",
        "title": "Fake Heist",
        "year": 2020,
        "overview": "A heist movie.",
        "poster_url": "https://image.tmdb.org/t/p/w342/poster.jpg",
        "vote_average": 7.5,
        "vote_count": 500,
        "genre_ids": [80, 28],
    }
    assert details == {"runtime_minutes": 128, "cast": ["Star"]}
    assert fake.count("/movie/101") == 1


def test_title_with_details_writes_title_details_cache() -> None:
    fake = FakeTMDB()
    fake.ok(
        "/movie/101",
        {"id": 101, "release_date": "2020-01-01", "runtime": 128, "credits": {}},
    )

    client = fake.client()
    run(client.title_with_details(media_type="movie", tmdb_id=101))
    details = run(client.title_details(media_type="movie", tmdb_id=101))

    assert details == {"runtime_minutes": 128, "cast": []}
    # title_with_details already fetched and cached this id's details; the
    # later title_details() call must be a cache hit, not a second request.
    assert fake.count("/movie/101") == 1


def test_title_with_details_404_returns_none() -> None:
    fake = FakeTMDB()
    fake.queue("/movie/1", httpx2.Response(404, json={"success": False}))

    result = run(fake.client().title_with_details(media_type="movie", tmdb_id=1))

    assert result is None


def test_title_with_details_never_caches_the_title_half() -> None:
    # Base title fields (vote_average etc.) are live TMDB data, unlike
    # runtime/cast — a second call must hit the network again, not reuse a
    # stale rating.
    fake = FakeTMDB()
    fake.ok(
        "/movie/101",
        {"id": 101, "release_date": "2020-01-01", "vote_average": 7.0, "credits": {}},
    )
    fake.ok(
        "/movie/101",
        {"id": 101, "release_date": "2020-01-01", "vote_average": 9.0, "credits": {}},
    )

    client = fake.client()
    first = run(client.title_with_details(media_type="movie", tmdb_id=101))
    second = run(client.title_with_details(media_type="movie", tmdb_id=101))

    assert first is not None and first[0]["vote_average"] == 7.0
    assert second is not None and second[0]["vote_average"] == 9.0
    assert fake.count("/movie/101") == 2


def test_genre_names_resolves_movie_and_tv_ids() -> None:
    assert tmdb.genre_names("movie", [80, 878]) == ["Crime", "Science Fiction"]
    assert tmdb.genre_names("tv", [10759]) == ["Action & Adventure"]


def test_genre_names_drops_unknown_ids() -> None:
    assert tmdb.genre_names("movie", [80, 999999]) == ["Crime"]


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


def test_watch_provider_list_merges_movie_and_tv() -> None:
    fake = FakeTMDB()
    fake.ok(
        "/watch/providers/movie",
        {
            "results": [
                {
                    "provider_id": 8,
                    "provider_name": "Netflix",
                    "logo_path": "/n.jpg",
                    "display_priorities": {"US": 2},
                },
                {
                    "provider_id": 337,
                    "provider_name": "Disney Plus",
                    "logo_path": "/d.jpg",
                    "display_priorities": {"US": 5},
                },
            ]
        },
    )
    fake.ok(
        "/watch/providers/tv",
        {
            "results": [
                # Same provider_id as movie's Netflix, worse priority here —
                # the merge must keep the better (lower) one.
                {
                    "provider_id": 8,
                    "provider_name": "Netflix",
                    "logo_path": "/n.jpg",
                    "display_priorities": {"US": 4},
                },
                {
                    "provider_id": 15,
                    "provider_name": "Hulu",
                    "logo_path": None,
                    "display_priorities": {"US": 3},
                },
                # No US entry at all -- not available in this region, dropped.
                {
                    "provider_id": 99,
                    "provider_name": "Region-locked",
                    "logo_path": "/r.jpg",
                    "display_priorities": {"GB": 1},
                },
            ]
        },
    )

    providers = run(fake.client().watch_provider_list(watch_region="US"))

    assert [p["provider_id"] for p in providers] == [8, 15, 337]
    assert providers[0] == {
        "provider_id": 8,
        "provider_name": "Netflix",
        "logo_url": "https://image.tmdb.org/t/p/w92/n.jpg",
        "display_priority": 2,
    }
    assert providers[1]["logo_url"] is None


def test_watch_provider_list_bad_region_rejected() -> None:
    with pytest.raises(ValueError):
        run(FakeTMDB().client().watch_provider_list(watch_region="usa"))


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
