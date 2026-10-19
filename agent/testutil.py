"""Shared test doubles for the offline test suites.

Not named test_*.py so pytest never collects it as a test module — it's a plain
module the test files import, the same way each imports tmdb.py/catalog_tool.py.

FakeTMDB is the union of what those suites need; each uses a subset.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import httpx2
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field

from catalog_tool import (
    CatalogToolError,
    ComposedReply,
    Composer,
    OutcomeCause,
    RankedPick,
    RankResult,
    SearchArgs,
)
from tmdb import TMDBClient

# gateway's constant, hand-copied -- the two services share no code.
# Nothing enforces the copy: update it by hand when chat.go changes.
GATEWAY_MAX_HISTORY_EXCHANGES = 5


def raw_movie(tmdb_id: int) -> dict[str, Any]:
    """discover()/search_titles()'s list shape — genre_ids flat, unlike
    raw_movie_full's genre objects. For fixtures that need several distinct
    discover() rows; MOVIE_A below is spelled out separately on purpose."""
    return {
        "id": tmdb_id,
        "title": f"Title {tmdb_id}",
        "release_date": "2020-01-01",
        "overview": "An overview.",
        "vote_average": 7.5,
        "vote_count": 500,
        "genre_ids": [],
    }


def raw_named(
    tmdb_id: int, title: str, vote_count: int, media_type: str = "movie"
) -> dict[str, Any]:
    """A /search/multi result row. Unlike raw_movie's generated "Title 101"
    names, a title lookup asserts on the name, on the vote floor and on which
    media type won, so all three are callers' business here. media_type is on
    the row itself, which is what distinguishes /search/multi's shape from the
    per-type endpoints'."""
    return {
        **raw_movie(tmdb_id),
        "media_type": media_type,
        "title": title,
        "name": title,
        "first_air_date": "2020-01-01",
        "vote_count": vote_count,
    }


def raw_movie_full(tmdb_id: int, title: str) -> dict[str, Any]:
    """The single-title endpoint's shape, which _trim_title_from_full reads:
    genres arrive as objects here, not the list endpoint's flat genre_ids."""
    return {
        "id": tmdb_id,
        "title": title,
        "release_date": "2016-01-01",
        "overview": "",
        "vote_average": 7.0,
        "vote_count": 100,
        "genres": [{"id": 80, "name": "Crime"}],
    }


# Spelled out rather than derived from raw_movie(): tests assert against these
# values, and a field added to raw_movie must not change them underneath.
MOVIE_A = {
    "id": 101,
    "title": "Fake Heist",
    "release_date": "2020-01-01",
    "overview": "A heist movie.",
    "vote_average": 7.5,
    "vote_count": 500,
    "genre_ids": [80],
}


class FakeTMDB:
    """Records every outgoing request and replays queued responses by path.

    A path with a queue pops one response per call (so a 429 then a 200 tests
    a retry); a path with none gets an empty result set. pages() answers by
    page number instead, for callers that care which page carried what. The two
    are mutually exclusive per path — registering both raises rather than
    letting one silently win.
    """

    def __init__(self) -> None:
        self.requests: list[httpx2.Request] = []
        self._queues: dict[str, list[httpx2.Response]] = {}
        self._pages: dict[str, list[dict[str, Any]]] = {}

    def queue(self, path: str, *responses: httpx2.Response) -> None:
        if "/3" + path in self._pages:
            raise AssertionError(f"{path} already answers by page; see pages()")
        self._queues.setdefault("/3" + path, []).extend(responses)

    def ok(self, path: str, payload: dict[str, Any]) -> None:
        self.queue(path, httpx2.Response(200, json=payload))

    def pages(self, path: str, *payloads: dict[str, Any]) -> None:
        """Answer this path from the request's own `page` param rather than
        from arrival order — catalog_tool reads pages concurrently, so a queue
        binds a payload to whichever request the event loop happened to run
        first. Past the last payload the path answers empty, which is what TMDB
        does past total_pages.

        Every rung's discover() is page 1, so a rung is served the first payload
        again and absorb() dedupes it away — a test that needs a rung to bring
        new rows wants queue()."""
        if "/3" + path in self._queues:
            raise AssertionError(f"{path} already has a queue; see queue()")
        if "/3" + path in self._pages:
            raise AssertionError(f"{path} is already answering by page")
        self._pages["/3" + path] = list(payloads)

    def _handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if request.url.path in self._pages:
            payloads = self._pages[request.url.path]
            page = int(request.url.params.get("page", 1))
            if page <= len(payloads):
                return httpx2.Response(200, json=payloads[page - 1])
            return httpx2.Response(200, json={"results": []})
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

    def all_params_for(self, path: str) -> list[dict[str, str]]:
        return [dict(r.url.params) for r in self.requests if r.url.path == "/3" + path]

    def count(self, path: str) -> int:
        return sum(1 for r in self.requests if r.url.path == "/3" + path)


class ScriptedChatModel(GenericFakeChatModel):
    """A chat model that answers each call with the next scripted AIMessage:
    a tool call for call one, or RankResult JSON as text with no tool calls
    for call two, which is the reply ProviderStrategy parses. Each call's
    messages are recorded in `calls`, so a test counts a turn's model calls
    and reads what each one was sent.

    bind_tools returns the model itself, so the script ignores which tools or
    response format a call binds. The base class raises NotImplementedError
    there, and create_agent binds on every call. Its streaming path drops
    tool_calls, so a scripted tool call survives ainvoke but not a token
    stream."""

    calls: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(messages)
        return self._generate(messages, stop=stop, **kwargs)


class FailingChatModel(ScriptedChatModel):
    """A model whose every call fails, after being recorded — a provider
    error such as a rate limit."""

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(messages)
        raise RuntimeError("model call timed out")


def scripted(*replies: AIMessage) -> ScriptedChatModel:
    return ScriptedChatModel(messages=iter(replies))


def _tool_call(name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": f"call-{name}"}]
    )


def search_call(**args: Any) -> AIMessage:
    """Call one picking search_catalog. media_type defaults to "movie", the
    one argument the schema requires."""
    return _tool_call("search_catalog", {"media_type": "movie", **args})


def lookup_call(title: str, reply: str = "") -> AIMessage:
    return _tool_call("look_up_title", {"title": title, "reply": reply})


def answer_call(text: str, kind: str) -> AIMessage:
    return _tool_call("answer", {"text": text, "kind": kind})


def rank_reply(*tmdb_ids: int, blurb: str = "fits") -> AIMessage:
    """Call two: RankResult as JSON text, picking these ids in this order."""
    picks = [RankedPick(tmdb_id=i, blurb=blurb) for i in tmdb_ids]
    return AIMessage(content=RankResult(picks=picks).model_dump_json())


def make_intent(**overrides: Any) -> SearchArgs:
    return SearchArgs(media_type="movie", **overrides)


COMPOSED_TEXT = "A sentence the model wrote."


class RecordingComposer:
    """Records every outcome call instead of raising when it is unwanted.

    Recorded rather than raising, for the same reason FakeTMDB records
    requests: compose() wraps anything a Composer
    raises as CatalogToolError, which _outcome_text degrades from — so a double
    that raised would answer from the template and assert nothing. A call this
    turn should not have made is asserted on `calls` after it finishes.
    """

    def __init__(self, text: str = COMPOSED_TEXT) -> None:
        self._text = text
        self.calls: list[tuple[OutcomeCause, str]] = []

    @property
    def causes(self) -> list[OutcomeCause]:
        return [cause for cause, _ in self.calls]

    def __call__(self) -> Composer:
        async def call(message: str, cause: OutcomeCause, detail: str) -> ComposedReply:
            self.calls.append((cause, detail))
            return ComposedReply(text=self._text)

        return call


async def ok_compose(message: str, cause: OutcomeCause, detail: str) -> ComposedReply:
    return ComposedReply(text=COMPOSED_TEXT)


async def failing_compose(
    message: str, cause: OutcomeCause, detail: str
) -> ComposedReply:
    """Drives chat.py back onto the templates that are now fallbacks."""
    raise CatalogToolError("composer unavailable")
