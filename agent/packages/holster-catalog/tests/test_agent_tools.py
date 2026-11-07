"""The declared agent tools and the scripted fake chat model, offline: each
tool's schema, and the replies the fake returns through bind_tools.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from holster_catalog.catalog_tool import (
    CATALOG_TOOLS,
    LookUpArgs,
    RankResult,
    SearchArgs,
)
from holster_catalog.testing import ScriptedChatModel


def test_no_tool_schema_shows_the_callers_own_context() -> None:
    """Which services and country results come from, and what the user has
    judged or seen, are the caller's facts. The schema actually sent to the
    model must not offer them as arguments."""
    hidden = {"watch_region", "watch_providers", "verdicts", "shown", "runtime"}
    for tool in CATALOG_TOOLS:
        schema = tool.model_json_schema()
        assert not hidden & set(schema["properties"]), schema["title"]


def test_the_search_tool_keeps_the_names_the_gateway_reads() -> None:
    """gateway/chat.go's agentIntent reads the intent event by these JSON
    names; title comes from look_up_title instead."""
    assert {
        "media_type",
        "genres",
        "without_genres",
        "keywords",
        "without_keywords",
        "cast",
        "crew",
        "max_runtime_minutes",
        "release_year_gte",
        "release_year_lte",
        "monetization",
    } <= set(SearchArgs.model_json_schema()["properties"])


def test_a_looked_up_title_is_one_line_and_never_blank() -> None:
    """A blank title would reach TMDB as an empty query."""
    assert LookUpArgs(title="Heat\n").title == "Heat"
    with pytest.raises(ValidationError):
        LookUpArgs(title=" \u2028 ")


def test_the_scripted_model_returns_a_tool_call_then_rank_json() -> None:
    call_one = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_catalog", "args": {"media_type": "movie"}, "id": "1"}
        ],
    )
    call_two = AIMessage(content='{"picks": [{"tmdb_id": 7, "blurb": "Fits."}]}')
    model = ScriptedChatModel(messages=iter([call_one, call_two]))
    bound = model.bind_tools(CATALOG_TOOLS)

    async def go() -> tuple[AIMessage, AIMessage]:
        return await bound.ainvoke("hi"), await bound.ainvoke("hi")

    first, second = asyncio.run(go())

    assert first.tool_calls[0]["name"] == "search_catalog"
    assert first.tool_calls[0]["args"] == {"media_type": "movie"}
    assert not second.tool_calls
    assert isinstance(second.content, str)
    assert RankResult.model_validate_json(second.content).picks[0].tmdb_id == 7
