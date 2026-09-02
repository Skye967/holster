"""Holster agent — FastAPI service.

Turns a message plus the caller's streaming subscriptions into TMDB catalog
queries and an answer. This module is the service skeleton: health, structured
logging, and correlation-ID propagation. Catalog tools and the chat endpoint
land in later tasks.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, cast

from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain_anthropic import ChatAnthropic
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from catalog_tool import (
    DEFAULT_MODEL,
    Interpreter,
    Ranker,
    anthropic_interpreter,
    anthropic_ranker,
)
from chat import ChatRequest, stream_chat
from tmdb import TMDBClient

# Header the gateway mints at the edge and forwards inward — see
# services/gateway and ARCHITECTURE.md. Lower-case per the ASGI/HTTP2 norm;
# HTTP header names are case-insensitive so clients and the gateway still match.
CORRELATION_ID_HEADER = "x-correlation-id"

# Set per request by CorrelationIdMiddleware, read by JsonFormatter so every line
# emitted while handling a request carries the same id.
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


class JsonFormatter(logging.Formatter):
    """One JSON object per line on stdout — the shape the gateway already emits."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if cid := _correlation_id.get():
            payload["correlation_id"] = cid
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _log_level() -> str:
    """LOG_LEVEL validated against the stdlib's own set, which already accepts the
    gateway's spellings (debug/info/warn/error). Unknown values stop startup."""
    level = os.getenv("LOG_LEVEL", "info").strip().upper() or "INFO"
    if level not in logging.getLevelNamesMapping():
        sys.exit(f"LOG_LEVEL: unknown value {level!r}")
    return level


def logging_config() -> dict[str, Any]:
    """Passed to uvicorn as its own logging config, so every logger — ours and
    uvicorn's — writes one JSON object per line to stdout from the first line on."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"json": {"()": JsonFormatter}},
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "json",
            }
        },
        "root": {"handlers": ["stdout"], "level": _log_level()},
        "loggers": {
            # Drop uvicorn's own handlers and let its records propagate to root.
            "uvicorn": {"handlers": [], "propagate": True},
            "uvicorn.error": {"handlers": [], "propagate": True},
            "uvicorn.access": {"handlers": [], "propagate": True},
        },
    }


class CorrelationIdMiddleware:
    """Adopt the incoming X-Correlation-ID or mint one, echo it on the response,
    and bind it for the request so log lines can pick it up.

    Wrapped around the app at the ASGI boundary (see ``asgi_app``) rather than via
    add_middleware, so it also covers the response and access-log line that
    Starlette's error handling produces. Pure ASGI rather than BaseHTTPMiddleware,
    which mishandles streaming responses and background tasks.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope["headers"]).get(CORRELATION_ID_HEADER.encode("latin-1"))
        adopted = incoming.decode("latin-1").strip() if incoming else ""
        cid = adopted or str(uuid.uuid4())
        token = _correlation_id.set(cid)

        async def send_with_header(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    *message.get("headers", []),
                    (CORRELATION_ID_HEADER.encode("latin-1"), cid.encode("latin-1")),
                ]
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_header)
        finally:
            _correlation_id.reset(token)


def _require_env(key: str) -> str:
    """Fail startup on a missing app-level key, same as the gateway's mustEnv —
    a bad deploy should not surface as the first chat request's error."""
    value = os.environ.get(key, "").strip()
    if not value:
        sys.exit(f"{key} not set")
    return value


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # ANTHROPIC_API_KEY is read by ChatAnthropic itself, not passed explicitly —
    # checked here anyway so a missing key stops startup rather than the first
    # /chat request. See TASKS.md T13's note on "app-level keys."
    _require_env("ANTHROPIC_API_KEY")
    app.state.tmdb_client = TMDBClient(_require_env("TMDB_API_KEY"))
    model = ChatAnthropic(
        model_name=DEFAULT_MODEL, timeout=15.0, max_retries=2, stop=None
    )
    app.state.interpret_model = anthropic_interpreter(model)
    app.state.rank_model = anthropic_ranker(model)
    try:
        yield
    finally:
        await app.state.tmdb_client.aclose()


app = FastAPI(title="holster-agent", lifespan=lifespan)
asgi_app = CorrelationIdMiddleware(app)


# Plain functions, not a class: request.app.state is the one instance-per-process
# TMDBClient/model pair built in lifespan() above. Depends() (rather than reading
# request.app.state directly in the route) is what lets tests swap in fakes via
# app.dependency_overrides, the same seam TMDBClient's transport= and
# catalog_tool's Interpreter/Ranker callables already use.
def get_tmdb_client(request: Request) -> TMDBClient:
    return cast(TMDBClient, request.app.state.tmdb_client)


def get_interpret_model(request: Request) -> Interpreter:
    return cast(Interpreter, request.app.state.interpret_model)


def get_rank_model(request: Request) -> Ranker:
    return cast(Ranker, request.app.state.rank_model)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(
    req: ChatRequest,
    tmdb_client: TMDBClient = Depends(get_tmdb_client),
    interpret_model: Interpreter = Depends(get_interpret_model),
    rank_model: Ranker = Depends(get_rank_model),
) -> StreamingResponse:
    """Streams chat.stream_chat()'s events as newline-delimited JSON. A plain
    chunked HTTP response, not a second WebSocket — the browser's socket is the
    gateway's alone (../DECISIONS.md); this is the "plain HTTP call" that
    decision describes, just with a body streamed as it becomes available
    rather than buffered, which is what lets the gateway forward an
    "interpreting" line before the full pipeline finishes.
    """

    async def events() -> AsyncIterator[bytes]:
        async for event in stream_chat(
            req,
            client=tmdb_client,
            interpret_model=interpret_model,
            rank_model=rank_model,
        ):
            yield json.dumps(event, default=str).encode() + b"\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(asgi_app, host="0.0.0.0", port=8000, log_config=logging_config())
