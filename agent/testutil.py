"""Shared test doubles for the offline test suites.

Not named test_*.py so pytest never tries to collect it as a test module —
it's a plain module imported by test_tmdb.py, test_catalog_tool.py, and
test_chat.py, the same way each of those imports tmdb.py/catalog_tool.py.

FakeTMDB started as test_tmdb.py's own fake, then got copied into
test_catalog_tool.py and test_chat.py as each needed it, drifting slightly
apart in which methods each copy carried. This is the union of all three —
every test file uses a subset of it now instead of maintaining its own copy.
"""

from __future__ import annotations

from typing import Any

import httpx2

from catalog_tool import DiscoverIntent
from tmdb import TMDBClient

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
    a retry); a path with none gets an empty result set.
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

    def all_params_for(self, path: str) -> list[dict[str, str]]:
        return [dict(r.url.params) for r in self.requests if r.url.path == "/3" + path]

    def count(self, path: str) -> int:
        return sum(1 for r in self.requests if r.url.path == "/3" + path)


def make_intent(**overrides: Any) -> DiscoverIntent:
    return DiscoverIntent(media_type="movie", **overrides)


async def ok_interpret(message: str) -> DiscoverIntent:
    return make_intent()
