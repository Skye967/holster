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
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI
from starlette.types import ASGIApp, Message, Receive, Scope, Send

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


app = FastAPI(title="holster-agent")
asgi_app = CorrelationIdMiddleware(app)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(asgi_app, host="0.0.0.0", port=8000, log_config=logging_config())
