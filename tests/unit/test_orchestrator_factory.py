import asyncio

import pytest

from app.agents.registry import create_default_registry
from app.core import Settings
from app.orchestrator import (
    LangGraphOrchestrator,
    OrchestratorState,
    SimpleOrchestrator,
    WorkflowStatus,
    create_orchestrator,
)
from app.tools import ToolExecutor, create_default_tool_registry


def test_orchestrator_factory_defaults_to_simple_backend() -> None:
    settings = Settings()

    orchestrator = create_orchestrator(
        settings=settings,
        registry=create_default_registry(),
    )

    assert isinstance(orchestrator, SimpleOrchestrator)


def test_orchestrator_factory_uses_langgraph_when_configured() -> None:
    pytest.importorskip("langgraph")
    settings = Settings.model_validate(
        {
            "orchestrator": {"backend": "langgraph", "max_graph_steps": 20},
        }
    )

    orchestrator = create_orchestrator(
        settings=settings,
        registry=create_default_registry(),
    )

    assert isinstance(orchestrator, LangGraphOrchestrator)


def test_langgraph_orchestrator_runs_default_chain() -> None:
    pytest.importorskip("langgraph")
    orchestrator = LangGraphOrchestrator(
        create_default_registry(),
        max_graph_steps=20,
    )
    state = OrchestratorState(
        request_id="req-langgraph-1",
        initial_agent_id="planner_agent",
        input_text="build a reusable enterprise multi-agent framework",
        structured_input={"requested_agent": "executor_agent"},
    )

    result_state = asyncio.run(orchestrator.run(state))

    assert result_state.status == WorkflowStatus.COMPLETED
    assert result_state.execution_path == [
        "planner_agent",
        "router_agent",
        "executor_agent",
        "reviewer_agent",
    ]


def test_langgraph_orchestrator_executes_tool_step_before_next_agent() -> None:
    pytest.importorskip("langgraph")
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
    orchestrator = LangGraphOrchestrator(
        create_default_registry(),
        tool_executor=tool_executor,
        max_graph_steps=20,
    )
    state = OrchestratorState(
        request_id="req-langgraph-tool-1",
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
    # Router consumed the tool results and routed to the knowledge agent.
    router_output = result_state.shared_state["agent_outputs"]["router_agent"]
    assert router_output["knowledge_hits"] == 1
    assert router_output["selected_agent"] == "knowledge_agent"
    # Knowledge agent materialized retrieval refs for downstream agents.
    knowledge_output = result_state.shared_state["agent_outputs"]["knowledge_agent"]
    assert knowledge_output["knowledge_refs"] == [
        "knowledge_search:enterprise multi-agent platform"
    ]
    # Executor agent consumed the retrieval refs from the knowledge agent.
    executor_output = result_state.shared_state["agent_outputs"]["executor_agent"]
    assert executor_output["knowledge_refs"] == [
        "knowledge_search:enterprise multi-agent platform"
    ]
    # Orchestration synced the runtime refs back into the auditable state.
    assert result_state.knowledge_refs == [
        "knowledge_search:enterprise multi-agent platform"
    ]


def test_langgraph_orchestrator_keeps_default_chain_without_tool_hits() -> None:
    pytest.importorskip("langgraph")
    tool_executor = ToolExecutor(create_default_tool_registry())
    orchestrator = LangGraphOrchestrator(
        create_default_registry(),
        tool_executor=tool_executor,
        max_graph_steps=20,
    )
    # No planner_tool_query -> no tool calls -> router falls back to the
    # requested agent without invoking the knowledge agent.
    state = OrchestratorState(
        request_id="req-langgraph-tool-2",
        initial_agent_id="planner_agent",
        input_text="build a reusable platform",
        structured_input={"requested_agent": "executor_agent"},
    )

    result_state = asyncio.run(orchestrator.run(state))

    assert result_state.status == WorkflowStatus.COMPLETED
    assert result_state.execution_path == [
        "planner_agent",
        "router_agent",
        "executor_agent",
        "reviewer_agent",
    ]
    router_output = result_state.shared_state["agent_outputs"]["router_agent"]
    assert router_output["knowledge_hits"] == 0
    assert router_output["selected_agent"] == "executor_agent"
