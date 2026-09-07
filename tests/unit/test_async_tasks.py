"""Unit tests for the Celery app wiring and the workflow.run task."""

import pytest

import app.runtime_stack as runtime_stack_module
from app.core import Settings
from app.orchestrator import OrchestratorState, WorkflowStatus
from app.tasks.celery_app import build_celery_app
from app.tasks import workflow_tasks
from app.tasks.workflow_tasks import build_orchestrator_state, run_workflow_task


def test_celery_app_urls_from_settings() -> None:
    settings = Settings()
    settings.redis.host = "10.0.0.5"
    settings.redis.port = 7000
    settings.redis.username = "multiagent"
    settings.redis.password = "secret"
    settings.celery.broker_db = 1
    settings.celery.result_db = 2
    settings.celery.task_default_queue = "custom"
    settings.celery.worker_pool = "threads"

    celery_app = build_celery_app(settings)

    assert (
        celery_app.conf.broker_url
        == "redis://multiagent:secret@10.0.0.5:7000/1"
    )
    assert (
        celery_app.conf.result_backend
        == "redis://multiagent:secret@10.0.0.5:7000/2"
    )
    assert celery_app.conf.task_default_queue == "custom"
    assert celery_app.conf.worker_pool == "threads"
    assert "app.tasks.workflow_tasks" in celery_app.conf.include
    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.broker_connection_retry_on_startup is True


def test_workflow_run_task_registered() -> None:
    assert run_workflow_task.name == "workflow.run"


def test_build_orchestrator_state_from_json_dict() -> None:
    request = {
        "request_id": "req-async-1",
        "initial_agent_id": "planner_agent",
        "session_id": "session-1",
        "user_id": "user-1",
        "workflow_id": "wf-1",
        "trace_id": "trace-1",
        "input_text": "hello",
        "structured_input": {"requires_human": False},
        "shared_state": {"k": "v"},
        "memory": {"recall": []},
        "knowledge_refs": ["r1"],
        "max_steps": 5,
        "llm_profile": {"provider": "deepseek"},
    }
    state = build_orchestrator_state(request)

    assert state.request_id == "req-async-1"
    assert state.initial_agent_id == "planner_agent"
    assert state.session_id == "session-1"
    assert state.structured_input == {"requires_human": False}
    assert state.max_steps == 5
    assert state.llm_profile is not None and state.llm_profile.provider == "deepseek"


class _FakeOrchestrator:
    async def run(self, state: OrchestratorState) -> OrchestratorState:
        return OrchestratorState(
            request_id=state.request_id,
            initial_agent_id=state.initial_agent_id,
            status=WorkflowStatus.COMPLETED,
            execution_path=["planner_agent", "executor_agent"],
        )


class _FakeStack:
    def __init__(self) -> None:
        self.orchestrator = _FakeOrchestrator()


def test_run_workflow_task_returns_terminal_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workflow_tasks, "_get_runtime", lambda: _FakeStack())
    result = run_workflow_task(
        {
            "request_id": "req-async-2",
            "initial_agent_id": "planner_agent",
        }
    )

    assert result["request_id"] == "req-async-2"
    assert result["status"] == "completed"
    assert result["terminal_reason"] is None
    assert result["review_id"] is None
    assert result["execution_path"] == ["planner_agent", "executor_agent"]
    assert result["errors"] == []
    assert result["final_result"] is None


def test_run_workflow_task_serializes_final_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.agents.base.schemas import AgentExecutionStatus, AgentResult

    class _FakeOrchestratorWithResult:
        async def run(self, state: OrchestratorState) -> OrchestratorState:
            return OrchestratorState(
                request_id=state.request_id,
                initial_agent_id=state.initial_agent_id,
                status=WorkflowStatus.COMPLETED,
                final_result=AgentResult(
                    agent_id=state.initial_agent_id,
                    status=AgentExecutionStatus.SUCCESS,
                    output={"result": "done"},
                ),
            )

    class _FakeStackWithResult:
        def __init__(self) -> None:
            self.orchestrator = _FakeOrchestratorWithResult()

    monkeypatch.setattr(
        workflow_tasks, "_get_runtime", lambda: _FakeStackWithResult()
    )
    result = run_workflow_task(
        {
            "request_id": "req-async-3",
            "initial_agent_id": "planner_agent",
        }
    )

    serialized = result["final_result"]
    assert serialized["agent_id"] == "planner_agent"
    assert serialized["status"] == "success"
    assert serialized["output"] == {"result": "done"}


def test_runtime_stack_builds_once_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The process-level runtime is cached, not rebuilt per task."""
    calls = []

    def _fake_build():
        calls.append(1)
        return _FakeStack()

    monkeypatch.setattr(runtime_stack_module, "build_runtime_stack", _fake_build)
    # Reset the module cache so the test is deterministic.
    workflow_tasks._runtime_stack = None
    try:
        first = workflow_tasks._get_runtime()
        second = workflow_tasks._get_runtime()
    finally:
        workflow_tasks._runtime_stack = None

    assert first is second
    assert len(calls) == 1
