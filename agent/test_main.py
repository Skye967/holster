from fastapi import FastAPI
from fastapi.testclient import TestClient

from main import CORRELATION_ID_HEADER, CorrelationIdMiddleware, asgi_app

client = TestClient(asgi_app)


def test_health_ok() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_mints_correlation_id() -> None:
    resp = client.get("/health")
    assert resp.headers[CORRELATION_ID_HEADER]


def test_health_echoes_incoming_correlation_id() -> None:
    resp = client.get("/health", headers={CORRELATION_ID_HEADER: "corr-123"})
    assert resp.headers[CORRELATION_ID_HEADER] == "corr-123"


def test_correlation_id_survives_server_error() -> None:
    err_app = FastAPI()

    @err_app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("boom")

    err_client = TestClient(
        CorrelationIdMiddleware(err_app), raise_server_exceptions=False
    )
    resp = err_client.get("/boom", headers={CORRELATION_ID_HEADER: "trace-me"})
    assert resp.status_code == 500
    assert resp.headers[CORRELATION_ID_HEADER] == "trace-me"
