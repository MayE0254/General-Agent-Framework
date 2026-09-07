from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.agents.base.schemas import AgentExecutionStatus
from app.orchestrator.base import BaseOrchestrator
from app.orchestrator.shared_state import append_tool_result, set_last_tool_results
from app.orchestrator.state import (
    OrchestratorState,
    WorkflowStatus,
    WorkflowTerminalReason,
)
from app.tools.schemas import ToolCallRequest, ToolContext, ToolExecutionStatus

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from app.tools import ToolExecutor


class LangGraphOrchestrator(BaseOrchestrator):
    """LangGraph-based orchestrator skeleton compatible with current state models."""

    def __init__(
        self,
        *args,
        tool_executor: ToolExecutor | None = None,
        max_graph_steps: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._compiled_graph: CompiledStateGraph | None = None
        self._tool_executor = tool_executor
        self._max_graph_steps = max_graph_steps

    async def run(self, state: OrchestratorState) -> OrchestratorState:
        trace = self._start_workflow_trace(state)
        try:
            state.status = WorkflowStatus.RUNNING
            state.current_agent_id = state.initial_agent_id
            self._load_session_context(state)

            graph_input = {
                "orchestrator_state": state,
                "step_index": 0,
            }
            compiled_graph = self._get_compiled_graph()
            result = await compiled_graph.ainvoke(graph_input)
            final_state: OrchestratorState = result["orchestrator_state"]
            await self._persist_session_context(final_state)
            await self._create_human_review_if_needed(final_state)
            self._sync_knowledge_refs(final_state)
            self._persist_runtime_records(final_state)
            return final_state
        finally:
            self._finish_workflow_trace(trace, state)

    def _get_compiled_graph(self):
        if self._compiled_graph is None:
            self._compiled_graph = self._build_graph()
        return self._compiled_graph

    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "LangGraph is not installed in the current environment."
            ) from exc

        graph = StateGraph(dict)
        graph.add_node("agent_step", self._execute_agent_step)
        graph.add_node("tool_step", self._execute_tool_step)
        graph.add_edge(START, "agent_step")
        graph.add_conditional_edges(
            "agent_step",
            self._route_after_agent_step,
            {
                "tool_step": "tool_step",
                "agent_step": "agent_step",
                "__end__": END,
            },
        )
        graph.add_conditional_edges(
            "tool_step",
            self._route_after_tool_step,
            {
                "agent_step": "agent_step",
                "__end__": END,
            },
        )
        return graph.compile()

    async def _execute_agent_step(self, graph_state: dict[str, Any]) -> dict[str, Any]:
        state: OrchestratorState = graph_state["orchestrator_state"]
        step_index = int(graph_state["step_index"]) + 1

        if state.current_agent_id is None:
            state.status = WorkflowStatus.COMPLETED
            state.terminal_reason = WorkflowTerminalReason.NATURAL_COMPLETION
            return {"orchestrator_state": state, "step_index": step_index}

        effective_max_steps = min(state.max_steps, self._max_graph_steps)
        if step_index > effective_max_steps:
            state.status = WorkflowStatus.FAILED
            state.terminal_reason = WorkflowTerminalReason.MAX_STEPS_EXCEEDED
            state.errors.append("Workflow exceeded max_steps before completion.")
            return {"orchestrator_state": state, "step_index": step_index}

        if not self._registry.exists(state.current_agent_id):
            state.status = WorkflowStatus.FAILED
            state.terminal_reason = WorkflowTerminalReason.AGENT_NOT_REGISTERED
            state.errors.append(f"Agent '{state.current_agent_id}' is not registered.")
            return {"orchestrator_state": state, "step_index": step_index}

        agent = self._registry.create(state.current_agent_id)
        try:
            result = await self._run_agent_with_tracing(
                agent,
                self._build_agent_context(state),
                step_index=step_index,
            )
        except Exception as exc:
            message = (
                f"Agent '{state.current_agent_id}' raised an exception: "
                f"{type(exc).__name__}: {exc}"
            )
            from app.agents.base.schemas import AgentResult

            result = AgentResult(
                agent_id=state.current_agent_id,
                status=AgentExecutionStatus.FAILED,
                summary=message,
                errors=[message],
            )
            self._apply_result(state, step_index=step_index, result=result)
            state.status = WorkflowStatus.FAILED
            state.terminal_reason = WorkflowTerminalReason.AGENT_EXCEPTION
            state.current_agent_id = None
            return {
                "orchestrator_state": state,
                "step_index": step_index,
                "pending_tool_calls": [],
                "tool_owner_agent_id": None,
                "pending_next_agent_id": None,
            }
        self._apply_result(state, step_index=step_index, result=result)

        if result.status == AgentExecutionStatus.FAILED:
            state.status = WorkflowStatus.FAILED
            state.terminal_reason = WorkflowTerminalReason.AGENT_FAILED
            state.current_agent_id = None
        elif (
            result.requires_human
            or result.status == AgentExecutionStatus.NEEDS_HUMAN_REVIEW
        ):
            state.status = WorkflowStatus.NEEDS_HUMAN_REVIEW
            state.terminal_reason = WorkflowTerminalReason.NEEDS_HUMAN_REVIEW
            state.current_agent_id = None
        elif result.tool_calls:
            return {
                "orchestrator_state": state,
                "step_index": step_index,
                "pending_tool_calls": [item.model_dump() for item in result.tool_calls],
                "tool_owner_agent_id": result.agent_id,
                "pending_next_agent_id": result.next_agent_id,
            }
        elif result.next_agent_id is None:
            state.status = WorkflowStatus.COMPLETED
            state.terminal_reason = WorkflowTerminalReason.NATURAL_COMPLETION
            state.current_agent_id = None
        else:
            state.current_agent_id = result.next_agent_id

        return {
            "orchestrator_state": state,
            "step_index": step_index,
            "pending_tool_calls": [],
            "tool_owner_agent_id": None,
            "pending_next_agent_id": None,
        }

    async def _execute_tool_step(self, graph_state: dict[str, Any]) -> dict[str, Any]:
        state: OrchestratorState = graph_state["orchestrator_state"]
        pending_tool_calls = graph_state.get("pending_tool_calls", [])
        tool_owner_agent_id = graph_state.get("tool_owner_agent_id")
        pending_next_agent_id = graph_state.get("pending_next_agent_id")

        if not pending_tool_calls:
            if pending_next_agent_id is None:
                state.status = WorkflowStatus.COMPLETED
                state.current_agent_id = None
            else:
                state.current_agent_id = pending_next_agent_id
            return {
                "orchestrator_state": state,
                "step_index": graph_state["step_index"],
                "pending_tool_calls": [],
                "tool_owner_agent_id": None,
                "pending_next_agent_id": None,
            }

        if self._tool_executor is None:
            state.status = WorkflowStatus.FAILED
            state.terminal_reason = WorkflowTerminalReason.TOOL_EXECUTOR_UNAVAILABLE
            state.current_agent_id = None
            state.errors.append(
                "LangGraph tool_step requires a configured ToolExecutor."
            )
            return {
                "orchestrator_state": state,
                "step_index": graph_state["step_index"],
                "pending_tool_calls": [],
                "tool_owner_agent_id": None,
                "pending_next_agent_id": None,
            }

        if not tool_owner_agent_id or not self._registry.exists(tool_owner_agent_id):
            state.status = WorkflowStatus.FAILED
            state.terminal_reason = WorkflowTerminalReason.TOOL_OWNER_NOT_REGISTERED
            state.current_agent_id = None
            state.errors.append(
                f"Agent '{tool_owner_agent_id}' is not registered for tool execution."
            )
            return {
                "orchestrator_state": state,
                "step_index": graph_state["step_index"],
                "pending_tool_calls": [],
                "tool_owner_agent_id": None,
                "pending_next_agent_id": None,
            }

        tool_owner_agent = self._registry.create(tool_owner_agent_id)
        tool_context = ToolContext(
            request_id=state.request_id,
            agent_id=tool_owner_agent_id,
            session_id=state.session_id,
            trace_id=state.trace_id,
            shared_state=state.shared_state,
        )
        executed_results: list[dict[str, Any]] = []

        for item in pending_tool_calls:
            request = ToolCallRequest.model_validate(item)
            result = await self._tool_executor.execute(
                agent=tool_owner_agent,
                request=request,
                context=tool_context,
            )
            result_payload = result.model_dump()
            executed_results.append(result_payload)
            append_tool_result(
                state.shared_state,
                agent_id=tool_owner_agent_id,
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
                        f"'{request.tool_name}' finished with status '{result.status.value}'."
                    ]
                )
                set_last_tool_results(state.shared_state, executed_results)
                return {
                    "orchestrator_state": state,
                    "step_index": graph_state["step_index"],
                    "pending_tool_calls": [],
                    "tool_owner_agent_id": None,
                    "pending_next_agent_id": None,
                }

        set_last_tool_results(state.shared_state, executed_results)
        if pending_next_agent_id is None:
            state.status = WorkflowStatus.COMPLETED
            state.terminal_reason = WorkflowTerminalReason.NATURAL_COMPLETION
            state.current_agent_id = None
        else:
            state.current_agent_id = pending_next_agent_id

        return {
            "orchestrator_state": state,
            "step_index": graph_state["step_index"],
            "pending_tool_calls": [],
            "tool_owner_agent_id": None,
            "pending_next_agent_id": None,
        }

    def _route_after_agent_step(self, graph_state: dict[str, Any]) -> str:
        state: OrchestratorState = graph_state["orchestrator_state"]
        if state.status in {
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.NEEDS_HUMAN_REVIEW,
        }:
            return "__end__"
        if graph_state.get("pending_tool_calls"):
            return "tool_step"
        return "agent_step"

    def _route_after_tool_step(self, graph_state: dict[str, Any]) -> str:
        state: OrchestratorState = graph_state["orchestrator_state"]
        if state.status in {
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.NEEDS_HUMAN_REVIEW,
        }:
            return "__end__"
        return "agent_step"
