"""Live catalog_tool checks — real Anthropic calls, no TMDB.

Skips unless ANTHROPIC_API_KEY is set, the same way test_tmdb_live.py skips
without TMDB_API_KEY. TMDB's own live behaviour is already covered there;
this file's only job is proving interpret()/rank() actually work against the
real model, so it does not also require a TMDB key.

An LLM call is nondeterministic and costs money per run, unlike a TMDB
lookup, so each test here makes exactly one real round trip and asserts
structure only — never blurb wording or which titles got picked, which would
flake against model updates.

    ANTHROPIC_API_KEY=... uv run pytest test_catalog_tool_live.py -q
"""

from __future__ import annotations

import asyncio
import os

import pytest
from langchain_anthropic import ChatAnthropic

from catalog_tool import (
    DEFAULT_MODEL,
    RankedPick,
    anthropic_interpreter,
    anthropic_ranker,
)
from tmdb import Title

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping live catalog_tool tests",
)

MODEL_NAME = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)


def _model() -> ChatAnthropic:
    return ChatAnthropic(model_name=MODEL_NAME, timeout=10.0, max_retries=2, stop=None)


def test_interpret_returns_a_valid_intent() -> None:
    async def go() -> None:
        call = anthropic_interpreter(_model())
        intent = await call("something funny under 90 minutes, nothing bleak")
        assert intent.media_type in ("movie", "tv")
        assert intent.max_runtime_minutes is not None
        assert intent.max_runtime_minutes <= 90
        assert 1 <= intent.limit <= 20

    asyncio.run(go())


def test_rank_picks_only_from_the_real_candidates() -> None:
    candidates: list[Title] = [
        Title(
            tmdb_id=1,
            media_type="movie",
            title="Heat",
            year=1995,
            overview="A crew of professional bank robbers start to feel the "
            "heat from police.",
            poster_url=None,
            vote_average=8.2,
            vote_count=1000,
            genre_ids=[80],
        ),
        Title(
            tmdb_id=2,
            media_type="movie",
            title="Notting Hill",
            year=1999,
            overview="A romantic comedy set in London.",
            poster_url=None,
            vote_average=7.0,
            vote_count=800,
            genre_ids=[35],
        ),
    ]
    valid_ids = {c["tmdb_id"] for c in candidates}

    async def go() -> None:
        call = anthropic_ranker(_model())
        result = await call("a heist movie", candidates)
        assert result.picks
        for pick in result.picks:
            assert isinstance(pick, RankedPick)
            assert pick.tmdb_id in valid_ids
            assert pick.blurb

    asyncio.run(go())
