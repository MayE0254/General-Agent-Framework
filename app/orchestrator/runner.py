from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.agents.base.schemas import AgentExecutionStatus, AgentResult
from app.orchestrator.base import BaseOrchestrator
from app.orchestrator.shared_state import append_tool_result, set_last_tool_results
from app.orchestrator.state import (
    AgentRunRecord,
    OrchestratorState,
    WorkflowStatus,
    WorkflowTerminalReason,
)
from app.tools.schemas import ToolCallRequest, ToolContext, ToolExecutionStatus

if TYPE_CHECKING:
    from app.agents.base.schemas import PendingToolCall
    from app.tools import ToolExecutor


class SimpleOrchestrator(BaseOrchestrator):
    """Minimal orchestrator that chains agents by next_agent_id."""

    def __init__(self, *args, tool_executor: ToolExecutor | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tool_executor = tool_executor

    async def run(self, state: OrchestratorState) -> OrchestratorState:
        trace = self._start_workflow_trace(state)
        try:
            state.status = WorkflowStatus.RUNNING
            state.current_agent_id = state.initial_agent_id
            self._load_session_context(state)

            for step_index in range(1, state.max_steps + 1):
                if state.current_agent_id is None:
                    state.status = WorkflowStatus.COMPLETED
                    state.terminal_reason = WorkflowTerminalReason.NATURAL_COMPLETION
                    break

                if not self._registry.exists(state.current_agent_id):
                    state.status = WorkflowStatus.FAILED
                    state.terminal_reason = (
                        WorkflowTerminalReason.AGENT_NOT_REGISTERED
                    )
                    state.errors.append(
                        f"Agent '{state.current_agent_id}' is not registered."
                    )
                    break

                agent_id = state.current_agent_id
                agent = self._registry.create(agent_id)
                try:
                    result = await self._run_agent_with_tracing(
                        agent,
                        self._build_agent_context(state),
                        step_index=step_index,
                    )
                except Exception as exc:
                    message = (
                        f"Agent '{agent_id}' raised an exception: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    state.status = WorkflowStatus.FAILED
                    state.terminal_reason = WorkflowTerminalReason.AGENT_EXCEPTION
                    state.errors.append(message)
                    failure_result = AgentResult(
                        agent_id=agent_id,
                        status=AgentExecutionStatus.FAILED,
                        summary=message,
                        errors=[message],
                    )
                    self._apply_result(
                        state,
                        step_index=step_index,
                        result=failure_result,
                    )
                    state.current_agent_id = None
                    break
                self._apply_result(state, step_index=step_index, result=result)

                if result.status == AgentExecutionStatus.FAILED:
                    state.status = WorkflowStatus.FAILED
                    state.terminal_reason = WorkflowTerminalReason.AGENT_FAILED
                    state.current_agent_id = None
                    break

                if (
                    result.requires_human
                    or result.status == AgentExecutionStatus.NEEDS_HUMAN_REVIEW
                ):
                    state.status = WorkflowStatus.NEEDS_HUMAN_REVIEW
                    state.terminal_reason = (
                        WorkflowTerminalReason.NEEDS_HUMAN_REVIEW
                    )
                    break

                if result.tool_calls:
                    tools_ok = await self._execute_pending_tools(
                        state,
                        owner_agent_id=result.agent_id,
                        tool_calls=result.tool_calls,
                    )
                    if not tools_ok:
                        break

                if result.next_agent_id is None:
                    state.status = WorkflowStatus.COMPLETED
                    state.terminal_reason = WorkflowTerminalReason.NATURAL_COMPLETION
                    state.current_agent_id = None
                    break

                state.current_agent_id = result.next_agent_id
            else:
                state.status = WorkflowStatus.FAILED
                state.terminal_reason = WorkflowTerminalReason.MAX_STEPS_EXCEEDED
                state.errors.append("Workflow exceeded max_steps before completion.")

            await self._persist_session_context(state)
            await self._create_human_review_if_needed(state)
            self._sync_knowledge_refs(state)
            self._persist_runtime_records(state)

            return state
        finally:
            self._finish_workflow_trace(trace, state)

    async def _execute_pending_tools(
        self,
        state: OrchestratorState,
        *,
        owner_agent_id: str,
        tool_calls: list[PendingToolCall],
    ) -> bool:
        """Run the tool calls emitted by the current agent.

        Mirrors the LangGraph ``tool_step`` semantics: tool failure fails the
        workflow; success keeps ``next_agent_id`` routing untouched. Returns
        ``False`` when the workflow must stop.
        """
        if self._tool_executor is None:
            state.status = WorkflowStatus.FAILED
            state.terminal_reason = WorkflowTerminalReason.TOOL_EXECUTOR_UNAVAILABLE
            state.current_agent_id = None
            state.errors.append(
                "Simple orchestrator tool step requires a configured ToolExecutor."
            )
            return False

        if not owner_agent_id or not self._registry.exists(owner_agent_id):
            state.status = WorkflowStatus.FAILED
            state.terminal_reason = WorkflowTerminalReason.TOOL_OWNER_NOT_REGISTERED
            state.current_agent_id = None
            state.errors.append(
                f"Agent '{owner_agent_id}' is not registered for tool execution."
            )
            return False

        owner_agent = self._registry.create(owner_agent_id)
        tool_context = ToolContext(
            request_id=state.request_id,
            agent_id=owner_agent_id,
            session_id=state.session_id,
            trace_id=state.trace_id,
            shared_state=state.shared_state,
        )
        executed_results: list[dict[str, Any]] = []
        for item in tool_calls:
            request = ToolCallRequest.model_validate(item.model_dump())
            result = await self._tool_executor.execute(
                agent=owner_agent,
                request=request,
                context=tool_context,
            )
            result_payload = result.model_dump()
            executed_results.append(result_payload)
            append_tool_result(
                state.shared_state,
                agent_id=owner_agent_id,
                result_payload=result_payload,
            )
            if result.status != ToolExecutionStatus.SUCCESS:
                state.status = WorkflowStatus.FAILED
                state.terminal_reason = WorkflowTerminalReason.TOOL_FAILED
                state.current_agent_id = None
                state.errors.extend(
                    result.errors
                    or [
                        "Tool "
                        f"'{request.tool_name}' finished with status "
                        f"'{result.status.value}'."
                    ]
                )
                set_last_tool_results(state.shared_state, executed_results)
                return False
        set_last_tool_results(state.shared_state, executed_results)
        return True
