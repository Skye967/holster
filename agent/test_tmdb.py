"""TMDB client tests — offline, against httpx2.MockTransport.

There is no live path here: every request goes through MockTransport, so the
suite needs no TMDB_API_KEY and makes no network call.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import httpx2
import pytest

import tmdb
from testutil import FakeTMDB
from tmdb import (
    MIN_VOTE_COUNT,
    MIN_VOTE_COUNT_TOP_RATED,
    TMDBError,
    TMDBUnavailable,
)


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
            "id": 101,
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
    fake.ok("/tv/9", {"id": 9, "episode_run_time": [45, 60], "credits": {"cast": []}})

    details = run(fake.client().title_details(media_type="tv", tmdb_id=9))

    assert details["runtime_minutes"] == 45


def test_title_details_tv_empty_episode_run_time_is_none() -> None:
    fake = FakeTMDB()
    fake.ok("/tv/9", {"id": 9, "episode_run_time": [], "credits": {"cast": []}})

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
    fake.ok("/movie/101", {"id": 101, "runtime": 100, "credits": {"cast": []}})

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


def test_is_genre_resolved_is_per_media_type() -> None:
    """The two vocabularies genuinely differ, which is what callers need to
    know before treating a genre as a filter that was applied."""
    assert tmdb.is_genre_resolved("movie", "horror")
    assert not tmdb.is_genre_resolved("tv", "horror")
    assert tmdb.is_genre_resolved("tv", "sci-fi & fantasy")
    assert not tmdb.is_genre_resolved("movie", "sci-fi & fantasy")


def test_is_genre_resolved_normalizes_like_the_query_does() -> None:
    assert tmdb.is_genre_resolved("movie", "  Science Fiction ")
    assert not tmdb.is_genre_resolved("movie", "sciencefiction")


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


def test_non_object_json_body_is_unavailable_not_a_crash() -> None:
    """Parseable JSON that is not an object. Every caller does data.get(...),
    so without the guard this surfaces as an AttributeError - a bug, when it
    is a TMDB fault. Sibling of the non-JSON case above."""
    fake = FakeTMDB()
    fake.queue("/discover/movie", httpx2.Response(200, json=[1, 2, 3]))
    with pytest.raises(TMDBUnavailable):
        run(fake.client().discover(media_type="movie", watch_region="US"))


def test_search_titles_keeps_tmdbs_order_and_drops_people() -> None:
    """The order is the signal: re-sorting it (by vote count, say) lifts a
    popular near-match above an exact one. /search/multi also returns people,
    which are not titles."""
    fake = FakeTMDB()
    fake.ok(
        "/search/multi",
        {
            "results": [
                {"media_type": "person", "id": 9, "name": "Someone"},
                {
                    "media_type": "tv",
                    "id": 2,
                    "name": "Dune: Prophecy",
                    "vote_count": 500,
                },
                {
                    "media_type": "movie",
                    "id": 1,
                    "title": "Dune",
                    "vote_count": 15000,
                },
            ]
        },
    )

    titles = run(fake.client().search_titles("dune"))

    # TMDB's order, not vote order — the movie has 30x the votes and stays second.
    assert [t["tmdb_id"] for t in titles] == [2, 1]
    assert [t["media_type"] for t in titles] == ["tv", "movie"]


def test_search_titles_unusable_results_is_not_nothing_matched() -> None:
    """A shape change must not read as "no such title". That answer is a claim
    about the user's spelling; only TMDB knows it is really about TMDB."""
    fake = FakeTMDB()
    fake.ok("/search/multi", {"results": {"unexpected": "object"}})

    with pytest.raises(TMDBError):
        run(fake.client().search_titles("dune"))


def test_search_titles_makes_one_request_not_one_per_media_type() -> None:
    fake = FakeTMDB()
    fake.ok("/search/multi", {"results": []})

    run(fake.client().search_titles("dune"))

    assert fake.count("/search/multi") == 1
    assert fake.count("/search/movie") == 0
    assert fake.count("/search/tv") == 0


# --- candidate quality: sort, floor, and how multi-value filters join --------


def test_the_default_sort_is_most_watched_not_most_trending() -> None:
    """popularity.desc is not offered at all. It is a rolling trend metric, and
    it put War of the Worlds (2025), rated 4.0, above The Dark Knight."""
    fake = FakeTMDB()
    run(fake.client().discover(media_type="movie", watch_region="US"))
    assert fake.params_for("/discover/movie")["sort_by"] == "vote_count.desc"


def test_ordering_by_rating_raises_the_vote_floor() -> None:
    """200 votes is enough to keep an obscure title out of a filtered list and
    nowhere near enough to keep it off the top of a list ordered by rating."""
    fake = FakeTMDB()
    run(
        fake.client().discover(
            media_type="tv", watch_region="US", sort_by="vote_average.desc"
        )
    )
    params = fake.params_for("/discover/tv")
    assert params["vote_count.gte"] == str(MIN_VOTE_COUNT_TOP_RATED)
    assert MIN_VOTE_COUNT_TOP_RATED > MIN_VOTE_COUNT


def test_the_raised_floor_is_scoped_to_the_rating_sort() -> None:
    """Raising it globally would answer "highest rated" better and every narrow
    request worse — Korean horror is 15 titles at 200 and 3 at 1000."""
    fake = FakeTMDB()
    run(
        fake.client().discover(
            media_type="movie", watch_region="US", sort_by="vote_count.desc"
        )
    )
    assert fake.params_for("/discover/movie")["vote_count.gte"] == str(MIN_VOTE_COUNT)


def test_mood_keywords_are_ored_so_a_mood_does_not_zero_the_query() -> None:
    """TMDB's tag data is far too sparse for an AND: "thriller" is 1459 titles,
    "thriller AND slow-burn" is 0, and 0 loses the mood to the relaxation
    ladder — which is how a mood came to be silently dropped on almost every
    request."""
    fake = FakeTMDB()
    fake.ok("/search/keyword", {"results": [{"id": 11, "name": "slow burn"}]})
    fake.ok("/search/keyword", {"results": [{"id": 22, "name": "tense"}]})

    run(
        fake.client().discover(
            media_type="movie",
            watch_region="US",
            keywords=["slow burn", "tense"],
        )
    )

    assert fake.params_for("/discover/movie")["with_keywords"] == "11|22"


def test_naming_two_people_still_means_both_of_them() -> None:
    """The OR above is scoped to keywords. "With Hanks and Ryan" is both, and
    joining cast with "|" would quietly turn every such request into either."""
    fake = FakeTMDB()
    fake.ok("/search/person", {"results": [{"id": 31, "name": "Tom Hanks"}]})
    fake.ok("/search/person", {"results": [{"id": 32, "name": "Meg Ryan"}]})

    run(
        fake.client().discover(
            media_type="movie",
            watch_region="US",
            cast=["Tom Hanks", "Meg Ryan"],
        )
    )

    assert fake.params_for("/discover/movie")["with_cast"] == "31,32"
