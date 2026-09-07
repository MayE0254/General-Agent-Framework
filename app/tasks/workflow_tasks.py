"""Celery tasks bridging the async layer to the orchestrator.

The worker process builds the same runtime stack as the API process
(``build_runtime_stack``), so workflows execute with identical wiring:
registries, LLM service, audit persistence, memory and human-review hooks.
"""

from __future__ import annotations

from typing import Any
from traceback import format_exc

from app.orchestrator import OrchestratorState
from app.repositories import AsyncWorkflowSubmissionRepository
from app.tasks.celery_app import celery_app


def build_orchestrator_state(request: dict[str, Any]) -> OrchestratorState:
    """Rebuild an ``OrchestratorState`` from a JSON-serialized request dict.

    Mirrors ``POST /workflows/run`` so async submissions execute exactly the
    same workflow as the synchronous endpoint.
    """
    return OrchestratorState(
        request_id=request["request_id"],
        initial_agent_id=request["initial_agent_id"],
        session_id=request.get("session_id"),
        user_id=request.get("user_id"),
        workflow_id=request.get("workflow_id"),
        trace_id=request.get("trace_id"),
        input_text=request.get("input_text"),
        structured_input=request.get("structured_input") or {},
        shared_state=request.get("shared_state") or {},
        memory=request.get("memory") or {},
        knowledge_refs=request.get("knowledge_refs") or [],
        max_steps=request.get("max_steps") or 10,
        llm_profile=request.get("llm_profile"),
    )


_runtime_stack: Any | None = None


def _get_runtime():
    """Process-level runtime stack, built once per worker process."""
    global _runtime_stack
    if _runtime_stack is None:
        from app.runtime_stack import build_runtime_stack

        _runtime_stack = build_runtime_stack()
    return _runtime_stack


def _update_async_submission(
    task_id: str,
    *,
    status: str,
    message: str,
    result_payload: dict[str, Any] | None = None,
    traceback_text: str | None = None,
) -> None:
    if not task_id or task_id == "None":
        return
    stack = _get_runtime()
    resources = getattr(stack, "resources", None)
    session_factory = getattr(resources, "database_session_factory", None)
    if session_factory is None:
        return

    session = session_factory()
    repository = AsyncWorkflowSubmissionRepository(session)
    try:
        record = repository.get_by_task_id(task_id)
        if record is None:
            return
        record.status = status
        record.message = message
        if result_payload is not None:
            record.result_payload = result_payload
            record.traceback = None
        if traceback_text is not None:
            record.traceback = traceback_text
        repository.commit()
    finally:
        repository.close()


@celery_app.task(
    name="workflow.run",
    bind=True,
    max_retries=1,
    default_retry_delay=5,
    acks_late=True,
)
def run_workflow_task(self, request: dict[str, Any]) -> dict[str, Any]:
    """Execute a workflow asynchronously and return the terminal state."""
    import asyncio

    stack = _get_runtime()
    state = build_orchestrator_state(request)
    task_id = str(self.request.id)
    _update_async_submission(
        task_id,
        status="started",
        message="workflow execution started by worker",
    )
    try:
        result_state = asyncio.run(stack.orchestrator.run(state))
        final_result = result_state.final_result
        payload = {
            "request_id": result_state.request_id,
            "status": result_state.status.value,
            "terminal_reason": (
                result_state.terminal_reason.value
                if result_state.terminal_reason is not None
                else None
            ),
            "review_id": result_state.memory.get("human_review", {}).get("review_id"),
            "execution_path": result_state.execution_path,
            "errors": result_state.errors,
            "final_result": (
                final_result.model_dump(mode="json")
                if final_result is not None
                else None
            ),
        }
        _update_async_submission(
            task_id,
            status="success",
            message="workflow completed successfully",
            result_payload=payload,
        )
        return payload
    except Exception:
        _update_async_submission(
            task_id,
            status="failure",
            message="workflow execution failed",
            traceback_text=format_exc(),
        )
        raise
