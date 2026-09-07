import structlog
import structlog.contextvars
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.api import middleware as mw
from app.api.middleware import access_log_middleware, request_context_middleware
from app.observability import bind_request_context, clear_request_context


def test_bind_and_clear_request_context_helpers() -> None:
    clear_request_context()
    assert structlog.contextvars.get_contextvars() == {}

    bind_request_context(request_id="rid-1", trace_id="tid-1")
    context = structlog.contextvars.get_contextvars()
    assert context["request_id"] == "rid-1"
    assert context["trace_id"] == "tid-1"

    clear_request_context()
    assert structlog.contextvars.get_contextvars() == {}


def test_request_context_binds_structlog_contextvars() -> None:
    captured: dict = {}

    test_app = FastAPI()

    @test_app.get("/echo")
    async def echo(request: Request) -> dict:
        captured.update(structlog.contextvars.get_contextvars())
        return {"ok": True}

    test_app.middleware("http")(request_context_middleware)
    client = TestClient(test_app)

    response = client.get(
        "/echo",
        headers={"X-Request-ID": "rid-abc", "X-Trace-ID": "tid-xyz"},
    )
    assert response.status_code == 200
    assert captured["request_id"] == "rid-abc"
    assert captured["trace_id"] == "tid-xyz"


def test_request_context_does_not_leak_between_requests() -> None:
    captured: dict = {}

    test_app = FastAPI()

    @test_app.get("/echo")
    async def echo(request: Request) -> dict:
        captured.update(structlog.contextvars.get_contextvars())
        return {"ok": True}

    test_app.middleware("http")(request_context_middleware)
    client = TestClient(test_app)

    client.get("/echo", headers={"X-Request-ID": "rid-1", "X-Trace-ID": "tid-1"})
    assert captured["request_id"] == "rid-1"

    client.get("/echo", headers={"X-Request-ID": "rid-2", "X-Trace-ID": "tid-2"})
    assert captured["request_id"] == "rid-2"
    assert captured["trace_id"] == "tid-2"


def test_access_log_emits_structured_fields_with_context(monkeypatch) -> None:
    test_app = FastAPI()

    @test_app.get("/ping")
    async def ping() -> dict:
        return {"pong": True}

    test_app.middleware("http")(access_log_middleware)
    test_app.middleware("http")(request_context_middleware)

    # capture_logs replaces the processor chain by default, so pass
    # merge_contextvars explicitly to keep context merging active.
    with structlog.testing.capture_logs(
        processors=[structlog.contextvars.merge_contextvars]
    ) as logs:
        monkeypatch.setattr(mw, "logger", structlog.get_logger("tests.access"))
        client = TestClient(test_app)
        response = client.get(
            "/ping",
            headers={"X-Request-ID": "rid-abc"},
        )

    assert response.status_code == 200
    assert len(logs) == 1
    entry = logs[0]
    assert entry["event"] == "request_completed"
    assert entry["method"] == "GET"
    assert entry["path"] == "/ping"
    assert entry["status_code"] == 200
    # request_id/trace_id come from structlog contextvars, not inline args.
    assert entry["request_id"] == "rid-abc"
    assert entry["trace_id"] == "rid-abc"
