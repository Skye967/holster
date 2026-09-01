"""Live TMDB checks — real token, real HTTP.

Skips unless TMDB_API_KEY is set, the same way the gateway's database tests skip
without TEST_DATABASE_URL. No separate test key: TMDB is read-only, so there is
nothing a stray run can damage.

    TMDB_API_KEY=... uv run pytest test_tmdb_live.py -q
"""

from __future__ import annotations

import asyncio
import os

import pytest

from tmdb import TMDBClient

pytestmark = pytest.mark.skipif(
    not os.getenv("TMDB_API_KEY"),
    reason="TMDB_API_KEY not set; skipping live TMDB tests",
)

TOKEN = os.getenv("TMDB_API_KEY", "")


def test_name_resolution_and_discover() -> None:
    async def go() -> None:
        async with TMDBClient(TOKEN) as client:
            titles = await client.discover(
                media_type="movie",
                watch_region="US",
                cast=["Tilda Swinton"],
                sort_by="vote_average.desc",
            )
            assert titles, "expected some Tilda Swinton films"
            assert all(t["vote_count"] >= 200 for t in titles)

    asyncio.run(go())


def test_title_lookup_returns_streaming_services() -> None:
    async def go() -> None:
        async with TMDBClient(TOKEN) as client:
            matches = await client.search_titles("Dune", media_type="movie")
            assert matches
            avail = await client.watch_providers(
                media_type="movie",
                tmdb_id=matches[0]["tmdb_id"],
                watch_region="US",
            )
            # A title this prominent is on at least one service somewhere.
            assert avail["flatrate"] or avail["rent"] or avail["buy"]

    asyncio.run(go())
