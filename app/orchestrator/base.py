from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from app.agents.base.schemas import AgentContext, AgentExecutionStatus, AgentResult
from app.agents.registry import AgentRegistry
from app.orchestrator.state import (
    AgentRunRecord,
    OrchestratorState,
    WorkflowStatus,
)
from app.orchestrator.shared_state import record_agent_output
from app.services.memory_service import MemoryService
from app.services.runtime_record_service import RuntimeRecordService
from app.services.workflow_run_service import WorkflowRunService

if TYPE_CHECKING:
    from app.llm import LLMService

logger = logging.getLogger(__name__)


class BaseOrchestrator(ABC):
    """Shared base for orchestrator implementations."""

    def __init__(
        self,
        registry: AgentRegistry,
        runtime_record_service: RuntimeRecordService | None = None,
        workflow_run_service: WorkflowRunService | None = None,
        memory_service: MemoryService | None = None,
        human_review_service: Any | None = None,
        llm_service: LLMService | None = None,
        tracer: Any | None = None,
    ) -> None:
        self._registry = registry
        self._runtime_record_service = runtime_record_service
        self._workflow_run_service = workflow_run_service
        self._memory_service = memory_service
        self._human_review_service = human_review_service
        self._llm_service = llm_service
        self._tracer = tracer

    def _build_agent_context(self, state: OrchestratorState) -> AgentContext:
        """Build an AgentContext, injecting optional shared services."""
        context = state.to_agent_context()
        if self._llm_service is not None:
            context.llm_service = self._llm_service
        return context

    def _load_session_context(self, state: OrchestratorState) -> None:
        """Load previous turns and recalled long-term memories into state.memory.

        Called before agents execute so planner prompts can carry multi-turn
        context across requests sharing the same session_id.
        """
        if self._memory_service is None or not state.session_id:
            return
        context = self._memory_service.load_session_context(
            session_id=state.session_id,
            user_id=state.user_id,
            project_key=state.project_key,
        )
        state.memory.update(context)

    def _sync_knowledge_refs(self, state: OrchestratorState) -> None:
        """Merge knowledge-agent refs back onto the orchestrator state.

        Audit records snapshot ``state.knowledge_refs``; without this sync
        the refs stored for a run would only ever be the caller-provided
        initial value instead of what was actually retrieved.
        """
        knowledge_output = state.shared_state.get("agent_outputs", {}).get(
            "knowledge_agent", {}
        )
        retrieved_refs = knowledge_output.get("knowledge_refs") or []
        merged = list(dict.fromkeys([*state.knowledge_refs, *retrieved_refs]))
        if merged != state.knowledge_refs:
            state.knowledge_refs = merged

    async def _persist_session_context(self, state: OrchestratorState) -> None:
        """Persist the current turn and (optionally) extract durable memories."""
        if self._memory_service is None or not state.session_id:
            return
        user_input = state.input_text or ""
        assistant_output = ""
        if state.final_result is not None:
            assistant_output = (
                state.final_result.summary or ""
            )
        if not assistant_output:
            assistant_output = state.errors and "; ".join(state.errors) or ""

        self._memory_service.append_session_messages(
            session_id=state.session_id,
            request_id=state.request_id,
            turns=[
                {"role": "user", "content": user_input, "step_index": 1},
                {"role": "assistant", "content": assistant_output, "step_index": 2},
            ],
        )
        stored = await self._memory_service.extract_and_store_memories(
            session_id=state.session_id,
            user_id=state.user_id,
            request_id=state.request_id,
            user_input=user_input,
            assistant_output=assistant_output,
            project_key=state.project_key,
        )
        state.memory.setdefault("extraction", {})["stored_count"] = stored

    async def _create_human_review_if_needed(self, state: OrchestratorState) -> None:
        """Raise a human-in-the-loop review task for review-pending workflows."""
        if (
            self._human_review_service is None
            or state.status != WorkflowStatus.NEEDS_HUMAN_REVIEW
        ):
            return
        try:
            task = await self._human_review_service.create_from_state(state)
            state.memory.setdefault("human_review", {})["review_id"] = task.id
            logger.info(
                "Created human review task %s for request %s",
                task.id,
                state.request_id,
            )
        except Exception:
            # A review-task failure must not break the audited workflow result.
            logger.exception(
                "Failed to create human review task for request %s",
                state.request_id,
            )

    def _start_workflow_trace(self, state: OrchestratorState) -> Any:
        """Start a trace-level span for a workflow run, if a tracer is set."""
        if self._tracer is None:
            return None
        return self._tracer.start_workflow_span(state)

    def _finish_workflow_trace(self, span: Any, state: OrchestratorState) -> None:
        if self._tracer is None or span is None:
            return
        self._tracer.finish_workflow_span(span, state)

    async def _run_agent_with_tracing(
        self,
        agent: Any,
        context: AgentContext,
        *,
        step_index: int,
    ) -> AgentResult:
        """Run a single agent step wrapped in an observability span."""
        span = None
        if self._tracer is not None:
            span = self._tracer.start_agent_span(
                agent.metadata.agent_id,
                step_index,
                context,
            )
        try:
            result = await agent.run(context)
        except Exception as exc:
            if self._tracer is not None:
                self._tracer.capture_error(
                    exc,
                    agent_id=agent.metadata.agent_id,
                    step_index=step_index,
                )
            raise
        if self._tracer is not None:
            self._tracer.finish_agent_span(span, result)
        return result

    @abstractmethod
    async def run(self, state: OrchestratorState) -> OrchestratorState:
        """Run a workflow state through the orchestrator."""

    def _apply_result(
        self,
        state: OrchestratorState,
        *,
        step_index: int,
        result: AgentResult,
    ) -> None:
        state.execution_path.append(result.agent_id)
        state.run_history.append(
            AgentRunRecord(
                step_index=step_index,
                agent_id=result.agent_id,
                status=result.status,
                summary=result.summary,
                next_agent_id=result.next_agent_id,
                output=result.output,
                errors=result.errors,
            )
        )
        state.final_result = result
        record_agent_output(
            state.shared_state,
            agent_id=result.agent_id,
            output=result.output,
        )

        if result.errors:
            state.errors.extend(result.errors)

    def _persist_runtime_records(self, state: OrchestratorState) -> None:
        if self._runtime_record_service is not None:
            self._runtime_record_service.save_runtime_records(state)
        if self._workflow_run_service is not None:
            self._workflow_run_service.save_workflow_run(state)
