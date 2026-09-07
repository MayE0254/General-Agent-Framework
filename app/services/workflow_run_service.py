from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from app.models import WorkflowRunRecord
from app.repositories import WorkflowRunRepository

if TYPE_CHECKING:
    from app.orchestrator.state import OrchestratorState


def _human_review_refs(state: "OrchestratorState") -> tuple[str | None, str | None]:
    """Return (review_id, source_review_id) from the run's memory."""
    human_review_memory = state.memory.get("human_review", {})
    review_id = (
        human_review_memory.get("review_id")
        if state.status.value == "needs_human_review"
        else None
    )
    return review_id, human_review_memory.get("source_review_id")


class WorkflowRunService:
    """Transforms orchestrator state into persistent workflow records."""

    def __init__(self, repository: WorkflowRunRepository) -> None:
        self._repository = repository

    def build_record_from_state(self, state: OrchestratorState) -> WorkflowRunRecord:
        final_result = state.final_result
        final_agent_id = final_result.agent_id if final_result is not None else None
        summary = final_result.summary if final_result is not None else ""
        review_id, source_review_id = _human_review_refs(state)

        return WorkflowRunRecord(
            id=str(uuid4()),
            request_id=state.request_id,
            workflow_id=state.workflow_id,
            initial_agent_id=state.initial_agent_id,
            current_agent_id=state.current_agent_id,
            status=state.status.value,
            execution_path=state.execution_path,
            final_agent_id=final_agent_id,
            summary=summary,
            errors=state.errors,
            metadata_payload={
                "session_id": state.session_id,
                "user_id": state.user_id,
                "trace_id": state.trace_id,
                "knowledge_refs": state.knowledge_refs,
                "terminal_reason": (
                    state.terminal_reason.value
                    if state.terminal_reason is not None
                    else None
                ),
                "review_id": review_id,
                "source_review_id": source_review_id,
                "max_steps": state.max_steps,
            },
        )

    def save_workflow_run(self, state: OrchestratorState) -> WorkflowRunRecord:
        record = self.build_record_from_state(state)
        self._repository.add(record)
        self._repository.commit()
        return record
