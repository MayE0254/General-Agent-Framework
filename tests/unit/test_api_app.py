import uuid

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


def test_health_endpoint_returns_service_status(api_client: TestClient) -> None:
    response = api_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "multiagent-service"
    assert body["resources"]["database"] in {"ready", "degraded"}
    assert response.headers["X-Request-ID"] == body["request_id"]
    assert response.headers["X-Trace-ID"] == body["trace_id"]
    assert "X-Process-Time-MS" in response.headers


def test_health_endpoint_preserves_request_headers(api_client: TestClient) -> None:
    response = api_client.get(
        "/health",
        headers={
            "X-Request-ID": "req-header-123",
            "X-Trace-ID": "trace-header-456",
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-header-123"
    assert response.headers["X-Trace-ID"] == "trace-header-456"


def test_agents_endpoint_lists_default_agents(api_client: TestClient) -> None:
    response = api_client.get("/api/v1/agents")

    assert response.status_code == 200
    body = response.json()
    agent_ids = {item["agent_id"] for item in body}
    # The 5 system agents always mount; optional business plugins under
    # app/business/* only mount when their package exists (a bare
    # infrastructure checkout ships without any of them).
    assert agent_ids >= {
        "executor_agent",
        "knowledge_agent",
        "planner_agent",
        "reviewer_agent",
        "router_agent",
    }


def test_workflow_run_endpoint_executes_default_chain(
    api_client: TestClient,
) -> None:
    response = api_client.post(
        "/api/v1/workflows/run",
        json={
            "request_id": f"api-req-{uuid.uuid4().hex[:12]}",
            "initial_agent_id": "planner_agent",
            "input_text": "build a reusable enterprise agent platform",
            "structured_input": {"requested_agent": "executor_agent"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["execution_path"] == [
        "planner_agent",
        "router_agent",
        "executor_agent",
        "reviewer_agent",
    ]
    assert body["final_result"]["agent_id"] == "reviewer_agent"


def test_memory_crud_endpoints(api_client: TestClient) -> None:
    scope_key = f"user-{uuid.uuid4().hex[:8]}"

    create_response = api_client.post(
        "/api/v1/runtime/memories",
        json={
            "memory_type": "preference",
            "content": "prefers bullet lists in summaries",
            "scope": "user",
            "scope_key": scope_key,
            "importance": 4,
            "tags": ["style"],
        },
    )
    assert create_response.status_code == 201
    memory_id = create_response.json()["id"]

    list_response = api_client.get(
        "/api/v1/runtime/memories",
        params={"scope": "user", "scope_key": scope_key},
    )
    assert list_response.status_code == 200
    assert list_response.json()["pagination"]["total"] == 1
    assert list_response.json()["items"][0]["content"] == (
        "prefers bullet lists in summaries"
    )

    update_response = api_client.patch(
        f"/api/v1/runtime/memories/{memory_id}",
        json={"importance": 5},
    )
    assert update_response.status_code == 200
    assert update_response.json()["importance"] == 5

    delete_response = api_client.delete(
        f"/api/v1/runtime/memories/{memory_id}"
    )
    assert delete_response.status_code == 200
    assert delete_response.json()["deleted"] is True

    missing_response = api_client.delete(
        f"/api/v1/runtime/memories/{memory_id}"
    )
    assert missing_response.status_code == 404


def test_session_messages_endpoint(api_client: TestClient) -> None:
    session_id = f"sess-{uuid.uuid4().hex[:8]}"
    run_response = api_client.post(
        "/api/v1/workflows/run",
        json={
            "request_id": f"api-req-{uuid.uuid4().hex[:12]}",
            "initial_agent_id": "planner_agent",
            "session_id": session_id,
            "user_id": "api-test-user",
            "input_text": "continue the project work",
            "structured_input": {"requested_agent": "executor_agent"},
        },
    )
    assert run_response.status_code == 200

    messages_response = api_client.get(
        f"/api/v1/runtime/sessions/{session_id}/messages"
    )
    assert messages_response.status_code == 200
    body = messages_response.json()
    assert body["session_id"] == session_id
    assert body["pagination"]["total"] == 2
    assert [item["role"] for item in body["items"]] == ["user", "assistant"]
    assert body["items"][0]["content"] == "continue the project work"
