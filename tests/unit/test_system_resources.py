import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import app.api.routes.system as system_module


def test_system_resources_endpoint_returns_database_runtime_status(
    api_client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        system_module,
        "probe_redis_connection",
        lambda settings: {
            "reachable": True,
            "latency_ms": 8.5,
            "error_type": "",
            "error_message": "",
        },
    )
    response = api_client.get("/api/v1/system/resources")

    assert response.status_code == 200
    assert "charset=utf-8" in response.headers["content-type"]
    body = response.json()
    assert "database" in body
    assert "status" in body["database"]
    assert "config" in body["database"]
    assert "masked_url" in body["database"]["config"]
    assert "sqlalchemy_available" in body["database"]["config"]
    assert "redis" in body
    assert "config" in body["redis"]
    assert "worker_pool" in body["redis"]["config"]
    assert "task_default_queue" in body["redis"]["config"]
    assert body["redis"]["config"]["reachable"] is True
    assert body["redis"]["config"]["latency_ms"] == 8.5
    assert body["redis"]["config"]["probe_error"] == ""
    assert "***" in body["redis"]["config"]["url"]
    assert "redis-test-password" not in body["redis"]["config"]["url"]


def test_system_dashboard_endpoint_returns_html(
    api_client: TestClient,
) -> None:
    response = api_client.get("/api/v1/system/dashboard")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "MultiAgent Runtime Console" in response.text
    assert 'const apiPrefix = "/api/v1"' in response.text
    assert "/runtime/reviews/events" in response.text
    assert "提交异步工作流" in response.text
    assert "最近异步提交" in response.text
    assert "/system/workers" in response.text
    assert "tool-calls-failed-only" in response.text
    assert "tool-calls-retryable-only" in response.text
    assert "function renderToolCalls" in response.text
    assert "function renderReviewRelation" in response.text
    assert 'id="review-relation"' in response.text
    assert "data-source-review-id" in response.text


def test_system_workers_endpoint_returns_worker_snapshot(
    api_client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        system_module,
        "_normalize_worker_snapshot",
        lambda: {
            "status": "ready",
            "message": "Detected 1 online worker(s).",
            "workers": [
                {
                    "worker_name": "celery@test",
                    "online": True,
                    "pool": "celery.concurrency.solo:TaskPool",
                    "max_concurrency": 1,
                    "queues": ["multiagent"],
                    "ping": {"ok": "pong"},
                }
            ],
            "troubleshooting": ["check queue"],
        },
    )

    response = api_client.get("/api/v1/system/workers")

    assert response.status_code == 200
    assert "charset=utf-8" in response.headers["content-type"]
    body = response.json()
    assert body["status"] == "ready"
    assert body["workers"][0]["worker_name"] == "celery@test"
    assert body["workers"][0]["queues"] == ["multiagent"]
