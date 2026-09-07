"""Unit tests for the async workflow submission/status API."""

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.api.dependencies import get_async_workflow_submission_service
from app.core import NotFoundError
import app.api.routes.workflows as workflows_module


class _FakeSendResult:
    def __init__(self, task_id: str) -> None:
        self.id = task_id


class _FakeSubmissionRecord:
    def __init__(
        self,
        *,
        task_id: str,
        request_id: str,
        status: str = "pending",
        message: str = "workflow submitted to async queue",
        result_payload: dict | None = None,
        traceback: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        self.task_id = task_id
        self.request_id = request_id
        self.session_id = None
        self.user_id = None
        self.initial_agent_id = "planner_agent"
        self.task_name = "workflow.run"
        self.queue_name = "multiagent"
        self.status = status
        self.message = message
        self.result_payload = result_payload
        self.traceback = traceback
        self.metadata_payload = {}
        self.created_at = now
        self.updated_at = now


class _FakeSubmissionService:
    def __init__(self, records: list[_FakeSubmissionRecord] | None = None) -> None:
        self._records = {record.task_id: record for record in records or []}

    def get_by_request_id(self, request_id: str):
        for record in self._records.values():
            if record.request_id == request_id:
                return record
        return None

    def create_submission(self, request, *, task_id: str, task_name: str, queue_name: str):
        record = _FakeSubmissionRecord(
            task_id=task_id,
            request_id=request.request_id,
            status="pending",
            message="workflow submitted to async queue",
        )
        record.initial_agent_id = request.initial_agent_id
        record.session_id = request.session_id
        record.user_id = request.user_id
        record.task_name = task_name
        record.queue_name = queue_name
        record.metadata_payload = request.model_dump(mode="json")
        self._records[task_id] = record
        return record

    def refresh_submission_status(self, task_id: str, *, raise_if_missing: bool = True):
        record = self._records.get(task_id)
        if record is None and raise_if_missing:
            raise NotFoundError("missing", details={"task_id": task_id})
        return record

    def get_submission(self, task_id: str):
        record = self._records.get(task_id)
        if record is None:
            raise NotFoundError("missing", details={"task_id": task_id})
        return record

    def list_submissions(self, **kwargs):
        items = list(self._records.values())
        return {
            "items": items,
            "total": len(items),
            "limit": kwargs.get("limit", len(items)),
            "offset": kwargs.get("offset", 0),
            "filters": {
                "status": kwargs.get("status"),
                "request_id": kwargs.get("request_id"),
                "session_id": kwargs.get("session_id"),
                "user_id": kwargs.get("user_id"),
            },
        }

    def close(self) -> None:
        return None


def _payload() -> dict:
    return {
        "request_id": "req-async-api-1",
        "input_text": "run a task",
        "structured_input": {},
    }


def _override_submission_service(
    api_client: TestClient, service: _FakeSubmissionService
) -> None:
    api_client.app.dependency_overrides[get_async_workflow_submission_service] = (
        lambda: service
    )


def test_async_submit_503_when_redis_unreachable(
    api_client: TestClient, monkeypatch
) -> None:
    _override_submission_service(api_client, _FakeSubmissionService())
    monkeypatch.setattr(
        workflows_module,
        "probe_redis_connection",
        lambda settings: {
            "reachable": False,
            "latency_ms": 1000.0,
            "error_type": "TimeoutError",
            "error_message": "Timeout connecting to server",
        },
    )
    response = api_client.post("/api/v1/workflows/run/async", json=_payload())

    assert response.status_code == 503
    assert "Redis" in response.json()["detail"]


def test_async_submit_returns_202_and_task_id(
    api_client: TestClient, monkeypatch
) -> None:
    _override_submission_service(api_client, _FakeSubmissionService())
    monkeypatch.setattr(
        workflows_module,
        "probe_redis_connection",
        lambda settings: {
            "reachable": True,
            "latency_ms": 12.0,
            "error_type": "",
            "error_message": "",
        },
    )
    monkeypatch.setattr(workflows_module, "uuid4", lambda: "task-abc-123")
    monkeypatch.setattr(
        workflows_module.celery_app,
        "send_task",
        lambda name, args=None, task_id=None: _FakeSendResult(
            task_id or "task-abc-123"
        ),
    )
    response = api_client.post("/api/v1/workflows/run/async", json=_payload())

    assert response.status_code == 202
    body = response.json()
    assert body["task_id"] == "task-abc-123"
    assert body["request_id"] == "req-async-api-1"
    assert body["status"] == "pending"
    assert body["workflow_status"] == "pending"
    assert body["terminal_reason"] is None


def test_async_task_status_503_when_redis_unreachable(
    api_client: TestClient, monkeypatch
) -> None:
    _override_submission_service(api_client, _FakeSubmissionService())
    monkeypatch.setattr(
        workflows_module, "is_redis_reachable", lambda settings: False
    )
    response = api_client.get("/api/v1/workflows/tasks/task-abc-123")

    assert response.status_code == 503


def test_async_task_status_returns_result(
    api_client: TestClient, monkeypatch
) -> None:
    _override_submission_service(
        api_client,
        _FakeSubmissionService(
            [
                _FakeSubmissionRecord(
                    task_id="task-abc-123",
                    request_id="req-async-api-1",
                    status="success",
                    message="workflow completed successfully",
                    result_payload={
                        "request_id": "req-async-api-1",
                        "status": "completed",
                        "execution_path": ["planner_agent", "executor_agent"],
                        "errors": [],
                        "final_result": None,
                    },
                )
            ]
        ),
    )
    monkeypatch.setattr(
        workflows_module, "is_redis_reachable", lambda settings: True
    )
    response = api_client.get("/api/v1/workflows/tasks/task-abc-123")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["request_id"] == "req-async-api-1"
    assert body["workflow_status"] == "completed"
    assert body["result"]["request_id"] == "req-async-api-1"


def test_async_task_status_failure_returns_traceback(
    api_client: TestClient, monkeypatch
) -> None:
    _override_submission_service(
        api_client,
        _FakeSubmissionService(
            [
                _FakeSubmissionRecord(
                    task_id="task-abc-123",
                    request_id="req-async-api-1",
                    status="failure",
                    message="workflow execution failed",
                    traceback="boom",
                )
            ]
        ),
    )
    monkeypatch.setattr(
        workflows_module, "is_redis_reachable", lambda settings: True
    )
    response = api_client.get("/api/v1/workflows/tasks/task-abc-123")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failure"
    assert body["result"] is None
    assert body["traceback"] == "boom"


def test_async_submit_returns_409_for_duplicate_request_id(
    api_client: TestClient, monkeypatch
) -> None:
    _override_submission_service(
        api_client,
        _FakeSubmissionService(
            [
                _FakeSubmissionRecord(
                    task_id="task-existing",
                    request_id="req-async-api-1",
                )
            ]
        ),
    )
    monkeypatch.setattr(
        workflows_module,
        "probe_redis_connection",
        lambda settings: {
            "reachable": True,
            "latency_ms": 12.0,
            "error_type": "",
            "error_message": "",
        },
    )

    response = api_client.post("/api/v1/workflows/run/async", json=_payload())

    assert response.status_code == 409
    assert "request_id already exists" in response.json()["detail"]


def test_list_async_submissions_returns_items(
    api_client: TestClient, monkeypatch
) -> None:
    _override_submission_service(
        api_client,
        _FakeSubmissionService(
            [
                _FakeSubmissionRecord(
                    task_id="task-abc-123",
                    request_id="req-async-api-1",
                    status="pending",
                )
            ]
        ),
    )
    monkeypatch.setattr(
        workflows_module, "is_redis_reachable", lambda settings: True
    )

    response = api_client.get("/api/v1/workflows/submissions?limit=10")

    assert response.status_code == 200
    body = response.json()
    assert body["items"][0]["task_id"] == "task-abc-123"
    assert body["items"][0]["request_id"] == "req-async-api-1"
    assert body["pagination"]["total"] == 1
