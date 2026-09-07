from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from app.models import (
    AgentRunLog,
    LLMCallLog,
    SessionRecord,
    TaskRunRecord,
    ToolCallLog,
)
from app.repositories import RuntimeAuditRepository

if TYPE_CHECKING:
    from app.llm import LLMCallAudit
    from app.orchestrator.state import OrchestratorState
    from app.tools.schemas import ToolCallRequest, ToolContext, ToolResult


def _human_review_refs(state: "OrchestratorState") -> tuple[str | None, str | None]:
    """Return (review_id, source_review_id) from the run's memory.

    ``review_id`` is only set when this run itself ended in
    ``needs_human_review``; ``source_review_id`` marks continuation runs
    that were approved through a review.
    """
    human_review_memory = state.memory.get("human_review", {})
    review_id = (
        human_review_memory.get("review_id")
        if state.status.value == "needs_human_review"
        else None
    )
    return review_id, human_review_memory.get("source_review_id")


class RuntimeRecordService:
    """Builds and persists reusable runtime audit records."""

    def __init__(self, repository: RuntimeAuditRepository) -> None:
        self._repository = repository

    def build_session_record(self, state: OrchestratorState) -> SessionRecord | None:
        if not state.session_id:
            return None

        return SessionRecord(
            id=str(uuid4()),
            session_id=state.session_id,
            user_id=state.user_id,
            status="active",
            last_request_id=state.request_id,
            metadata_payload={
                "trace_id": state.trace_id,
                "workflow_id": state.workflow_id,
            },
        )

    def build_task_run_record(self, state: OrchestratorState) -> TaskRunRecord:
        final_result = state.final_result
        review_id, source_review_id = _human_review_refs(state)
        return TaskRunRecord(
            id=str(uuid4()),
            request_id=state.request_id,
            session_id=state.session_id,
            workflow_id=state.workflow_id,
            task_type="workflow",
            status=state.status.value,
            initial_agent_id=state.initial_agent_id,
            final_agent_id=final_result.agent_id if final_result else None,
            summary=final_result.summary if final_result else "",
            errors=state.errors,
            metadata_payload={
                "trace_id": state.trace_id,
                "knowledge_refs": state.knowledge_refs,
                "execution_path": state.execution_path,
                "current_agent_id": state.current_agent_id,
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

    def build_agent_run_logs(self, state: OrchestratorState) -> list[AgentRunLog]:
        return [
            AgentRunLog(
                id=str(uuid4()),
                request_id=state.request_id,
                session_id=state.session_id,
                workflow_id=state.workflow_id,
                trace_id=state.trace_id,
                step_index=record.step_index,
                agent_id=record.agent_id,
                status=record.status.value,
                next_agent_id=record.next_agent_id,
                summary=record.summary,
                started_at_runtime=record.started_at,
                completed_at_runtime=record.completed_at,
                output=record.output,
                errors=record.errors,
            )
            for record in state.run_history
        ]

    def build_tool_call_log(
        self,
        *,
        tool_context: ToolContext,
        tool_request: ToolCallRequest,
        tool_result: ToolResult,
        workflow_id: str | None = None,
    ) -> ToolCallLog:
        metrics = dict(tool_result.metrics)
        metrics.setdefault(
            "error_category",
            tool_result.error_category.value
            if tool_result.error_category is not None
            else None,
        )
        metrics.setdefault("retryable", tool_result.retryable)
        metrics.setdefault("error_detail", tool_result.error_detail)
        return ToolCallLog(
            id=str(uuid4()),
            request_id=tool_context.request_id,
            session_id=tool_context.session_id,
            workflow_id=workflow_id,
            trace_id=tool_context.trace_id,
            agent_id=tool_context.agent_id,
            tool_name=tool_request.tool_name,
            status=tool_result.status.value,
            message=tool_result.message,
            arguments=tool_request.arguments,
            content=tool_result.content,
            errors=tool_result.errors,
            metrics=metrics,
        )

    def save_runtime_records(
        self,
        state: OrchestratorState,
    ) -> dict[str, object]:
        saved: dict[str, object] = {}

        session_record = self.build_session_record(state)
        if session_record is not None:
            saved["session"] = self._repository.add_session(session_record)

        saved["task_run"] = self._repository.add_task_run(
            self.build_task_run_record(state)
        )

        agent_logs = self.build_agent_run_logs(state)
        saved["agent_runs"] = self._repository.add_agent_run_logs(agent_logs)
        self._repository.commit()

        return saved

    def save_tool_call_log(self, tool_log: ToolCallLog) -> ToolCallLog:
        saved = self._repository.add_tool_call_logs([tool_log])[0]
        self._repository.commit()
        return saved

    def save_tool_call_result(
        self,
        *,
        tool_context: ToolContext,
        tool_request: ToolCallRequest,
        tool_result: ToolResult,
        workflow_id: str | None = None,
    ) -> ToolCallLog:
        tool_log = self.build_tool_call_log(
            tool_context=tool_context,
            tool_request=tool_request,
            tool_result=tool_result,
            workflow_id=workflow_id,
        )
        return self.save_tool_call_log(tool_log)

    def build_llm_call_log(self, audit: "LLMCallAudit") -> LLMCallLog:
        return LLMCallLog(
            id=str(uuid4()),
            request_id=audit.request_id,
            session_id=audit.session_id,
            workflow_id=audit.workflow_id,
            trace_id=audit.trace_id,
            agent_id=audit.agent_id,
            model=audit.model,
            status=audit.status,
            prompt_tokens=audit.prompt_tokens,
            completion_tokens=audit.completion_tokens,
            total_tokens=audit.total_tokens,
            latency_ms=audit.latency_ms,
            cost_usd=audit.cost_usd,
            messages=audit.messages or None,
            error=audit.error,
        )

    def save_llm_call_log(self, audit: "LLMCallAudit") -> LLMCallLog:
        saved = self._repository.add_llm_call_logs(
            [self.build_llm_call_log(audit)]
        )[0]
        self._repository.commit()
        return saved

    def close(self) -> None:
        self._repository.close()
