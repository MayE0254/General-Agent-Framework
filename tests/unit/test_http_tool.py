"""Unit tests for the real outbound HTTP tool (HttpRequestTool).

Network access is simulated with an httpx.MockTransport injected through a
patched AsyncClient, so no external host is contacted.
"""

import asyncio
import json

import httpx

from app.tools import (
    ToolContext,
    ToolErrorCategory,
    ToolExecutionStatus,
    ToolRegistry,
)
from app.tools.system import HttpRequestTool


def _context() -> ToolContext:
    return ToolContext(
        request_id="req-http-1",
        agent_id="executor",
    )


def _tool(http_config: dict | None = None) -> HttpRequestTool:
    registry = ToolRegistry()
    registry.register(HttpRequestTool, config=http_config)
    tool = registry.create("http_request")
    assert isinstance(tool, HttpRequestTool)
    return tool


def _patch_client(monkeypatch: object, handler) -> httpx.MockTransport:
    """Redirect AsyncClient construction to a transport backed by ``handler``."""
    transport = httpx.MockTransport(handler)

    class _PatchedClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        "app.tools.system.builtin_tools.httpx.AsyncClient", _PatchedClient
    )
    return transport


def _run(tool: HttpRequestTool, arguments: dict):
    return asyncio.run(tool.run(_context(), arguments))


def test_http_get_success(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/ping"
        assert request.headers.get("User-Agent") == "multiagent-agent/0.1"
        return httpx.Response(200, json={"pong": True})

    _patch_client(monkeypatch, handler)
    result = _run(_tool(), {"url": "https://example.com/ping"})

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.errors == []
    assert result.content["status_code"] == 200
    assert json.loads(result.content["body"]) == {"pong": True}
    assert result.content["truncated"] is False
    assert result.metrics["response_bytes"] == len(result.content["body"])


def test_http_post_json_body(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/submit"
        assert request.headers.get("content-type") == "application/json"
        payload = json.loads(request.content)
        return httpx.Response(201, json={"submitted": payload})

    _patch_client(monkeypatch, handler)
    result = _run(
        _tool(),
        {
            "url": "https://example.com/submit",
            "method": "POST",
            "json_body": {"task": "run", "priority": 3},
        },
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.content["status_code"] == 201
    assert json.loads(result.content["body"]) == {
        "submitted": {"task": "run", "priority": 3}
    }


def test_http_method_not_allowed(monkeypatch) -> None:
    _patch_client(
        monkeypatch,
        lambda request: httpx.Response(200),
    )
    result = _run(
        _tool({"allowed_methods": ["GET"]}),
        {"url": "https://example.com/x", "method": "POST"},
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.errors == ["method_not_allowed"]


def test_http_host_always_blocked(monkeypatch) -> None:
    # Cloud metadata endpoints are refused before any connection attempt.
    _patch_client(
        monkeypatch,
        lambda request: httpx.Response(200),
    )
    result = _run(
        _tool(),
        {"url": "http://169.254.169.254/latest/meta-data/"},
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.errors == ["host_blocked"]


def test_http_configured_host_blocked(monkeypatch) -> None:
    _patch_client(
        monkeypatch,
        lambda request: httpx.Response(200),
    )
    result = _run(
        _tool({"blocked_hosts": ["internal.example.com"]}),
        {"url": "https://internal.example.com/secret"},
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.errors == ["host_blocked"]


def test_http_timeout(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout")

    _patch_client(monkeypatch, handler)
    result = _run(_tool({"timeout_seconds": 1.0}), {"url": "https://example.com/slow"})

    assert result.status == ToolExecutionStatus.TIMEOUT
    assert result.errors == ["http_timeout"]
    assert result.error_category == ToolErrorCategory.TIMEOUT
    assert result.retryable is True
    assert "latency_ms" in result.metrics


def test_http_transport_error(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    _patch_client(monkeypatch, handler)
    result = _run(_tool(), {"url": "https://example.com/refused"})

    assert result.status == ToolExecutionStatus.FAILED
    assert result.errors == ["http_error"]


def test_http_response_truncated(monkeypatch) -> None:
    body = "x" * 200

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    _patch_client(monkeypatch, handler)
    result = _run(
        _tool({"max_response_bytes": 32}),
        {"url": "https://example.com/big"},
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.content["truncated"] is True
    assert len(result.content["body"]) == 32


def test_http_missing_url(monkeypatch) -> None:
    _patch_client(
        monkeypatch,
        lambda request: httpx.Response(200),
    )
    result = _run(_tool(), {})

    assert result.status == ToolExecutionStatus.FAILED
    assert result.errors == ["missing_url"]


def test_http_config_binding() -> None:
    registry = ToolRegistry()
    registry.register(
        HttpRequestTool,
        config={"timeout_seconds": 7.5, "allowed_methods": ["GET", "POST"]},
    )

    assert registry.get_config("http_request")["timeout_seconds"] == 7.5
    tool = registry.create("http_request")
    assert tool.config["allowed_methods"] == ["GET", "POST"]
