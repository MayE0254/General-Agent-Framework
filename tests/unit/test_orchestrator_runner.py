import asyncio

import pytest

from app.agents.base.agent import BaseAgent
from app.agents.base.schemas import AgentContext, AgentMetadata, AgentResult
from app.agents.registry import AgentRegistry, create_default_registry
from app.orchestrator import (
    OrchestratorState,
    SimpleOrchestrator,
    WorkflowStatus,
    WorkflowTerminalReason,
)
from app.tools import ToolExecutor, create_default_tool_registry


class _CrashingAgent(BaseAgent):
    @classmethod
    def build_metadata(cls) -> AgentMetadata:
        return AgentMetadata(
            agent_id="crashing_agent",
            agent_name="Crashing Agent",
            agent_role="test",
        )

    async def _run(self, context: AgentContext) -> AgentResult:
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_simple_orchestrator_converts_agent_exception_to_failed_state() -> None:
    registry = AgentRegistry()
    registry.register(_CrashingAgent)
    orchestrator = SimpleOrchestrator(registry)

    state = OrchestratorState(
        request_id="req-crash-1",
        initial_agent_id="crashing_agent",
        input_text="trigger crash",
    )

    result = await orchestrator.run(state)

    assert result.status == WorkflowStatus.FAILED
    assert result.terminal_reason == WorkflowTerminalReason.AGENT_EXCEPTION
    assert result.final_result is not None
    assert result.final_result.agent_id == "crashing_agent"
    assert result.final_result.status.value == "failed"
    assert any("RuntimeError: boom" in item for item in result.errors)
    assert result.run_history[-1].status.value == "failed"


def test_simple_orchestrator_executes_tool_step_before_next_agent() -> None:
    from app.core.settings import KnowledgeDocument, KnowledgeSettings

    knowledge_settings = KnowledgeSettings(
        documents=[
            KnowledgeDocument(
                id="kb-platform",
                title="Platform",
                content="enterprise multi-agent platform",
            )
        ]
    )
    tool_executor = ToolExecutor(
        create_default_tool_registry(knowledge_settings=knowledge_settings)
    )
    orchestrator = SimpleOrchestrator(
        create_default_registry(),
        tool_executor=tool_executor,
    )
    state = OrchestratorState(
        request_id="req-simple-tool-1",
        initial_agent_id="planner_agent",
        input_text="prepare a reusable platform",
        structured_input={
            "planner_tool_query": "enterprise multi-agent platform",
            "requested_agent": "executor_agent",
        },
    )

    result_state = asyncio.run(orchestrator.run(state))

    assert result_state.status == WorkflowStatus.COMPLETED
    # Tool results drive routing: knowledge_search hits > 0 routes through
    # knowledge_agent before reaching the executor.
    assert result_state.execution_path == [
        "planner_agent",
        "router_agent",
        "knowledge_agent",
        "executor_agent",
        "reviewer_agent",
    ]
    assert result_state.shared_state["last_tool_results"][0]["tool_name"] == (
        "knowledge_search"
    )
    assert result_state.shared_state["tool_results"]["planner_agent"][0]["content"][
        "backend"
    ] == "inmemory"
    router_output = result_state.shared_state["agent_outputs"]["router_agent"]
    assert router_output["knowledge_hits"] == 1
    assert router_output["selected_agent"] == "knowledge_agent"
    assert result_state.knowledge_refs == [
        "knowledge_search:enterprise multi-agent platform"
    ]


def test_simple_orchestrator_fails_when_tools_emitted_without_executor() -> None:
    orchestrator = SimpleOrchestrator(create_default_registry())
    state = OrchestratorState(
        request_id="req-simple-tool-2",
        initial_agent_id="planner_agent",
        input_text="retrieve platform knowledge",
        structured_input={"planner_tool_query": "enterprise platform"},
    )

    result_state = asyncio.run(orchestrator.run(state))

    assert result_state.status == WorkflowStatus.FAILED
    assert (
        result_state.terminal_reason
        == WorkflowTerminalReason.TOOL_EXECUTOR_UNAVAILABLE
    )
    assert any("ToolExecutor" in item for item in result_state.errors)
