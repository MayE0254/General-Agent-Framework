import asyncio

import pytest

from app.agents.registry import create_default_registry
from app.agents.system.builtin_agents import PlannerAgent
from app.core.settings import ObservabilitySettings
from app.observability import LangfuseTracer
from app.orchestrator import (
    LangGraphOrchestrator,
    OrchestratorState,
    SimpleOrchestrator,
    WorkflowStatus,
)
from app.tools import ToolExecutor, create_default_tool_registry
from app.tools.schemas import ToolCallRequest, ToolContext, ToolExecutionStatus


class FakeTracer:
    """Records span lifecycle events for assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def start_workflow_span(self, state: object) -> object:
        self.events.append(("start_workflow", {"request_id": state.request_id}))
        return {"kind": "workflow"}

    def finish_workflow_span(self, span: object, state: object) -> None:
        self.events.append(("finish_workflow", {"status": str(state.status)}))

    def start_agent_span(self, agent_id: str, step_index: int, context: object) -> object:
        self.events.append(
            ("start_agent", {"agent_id": agent_id, "step_index": step_index})
        )
        return {"kind": "agent"}

    def finish_agent_span(self, span: object, result: object) -> None:
        self.events.append(("finish_agent", {"agent_id": result.agent_id}))

    def start_tool_span(self, tool_name: str, agent_id: str, request: object) -> object:
        self.events.append(
            ("start_tool", {"tool_name": tool_name, "agent_id": agent_id})
        )
        return {"kind": "tool"}

    def finish_tool_span(self, span: object, result: object) -> None:
        self.events.append(("finish_tool", {"tool_name": result.tool_name}))

    def start_llm_span(self, request: object) -> object:
        return None

    def finish_llm_span(self, span: object, response: object) -> None:
        return None

    def capture_error(self, error: BaseException, **context: object) -> None:
        self.events.append(("error", context))


def _state(**overrides) -> OrchestratorState:
    base = {
        "request_id": "req-trace-1",
        "initial_agent_id": "planner_agent",
        "input_text": "build a reusable enterprise multi-agent framework",
        "structured_input": {"requested_agent": "executor_agent"},
    }
    base.update(overrides)
    return OrchestratorState(**base)


def test_simple_orchestrator_emits_workflow_and_agent_spans() -> None:
    tracer = FakeTracer()
    orchestrator = SimpleOrchestrator(create_default_registry(), tracer=tracer)

    result = asyncio.run(orchestrator.run(_state()))

    assert result.status == WorkflowStatus.COMPLETED
    kinds = [event[0] for event in tracer.events]
    assert "start_workflow" in kinds
    assert "finish_workflow" in kinds
    agent_ids = [
        event[1]["agent_id"]
        for event in tracer.events
        if event[0] == "start_agent"
    ]
    assert agent_ids == [
        "planner_agent",
        "router_agent",
        "executor_agent",
        "reviewer_agent",
    ]
    assert len([e for e in tracer.events if e[0] == "start_agent"]) == len(
        [e for e in tracer.events if e[0] == "finish_agent"]
    )


def test_langgraph_orchestrator_emits_workflow_and_agent_spans() -> None:
    pytest.importorskip("langgraph")
    tracer = FakeTracer()
    orchestrator = LangGraphOrchestrator(
        create_default_registry(),
        tracer=tracer,
        max_graph_steps=20,
    )

    result = asyncio.run(orchestrator.run(_state()))

    assert result.status == WorkflowStatus.COMPLETED
    agent_ids = [
        event[1]["agent_id"]
        for event in tracer.events
        if event[0] == "start_agent"
    ]
    assert agent_ids == [
        "planner_agent",
        "router_agent",
        "executor_agent",
        "reviewer_agent",
    ]
    assert [e[0] for e in tracer.events].count("start_workflow") == 1
    assert [e[0] for e in tracer.events].count("finish_workflow") == 1


def test_tool_executor_emits_tool_span() -> None:
    tracer = FakeTracer()
    executor = ToolExecutor(
        create_default_tool_registry(),
        tracer=tracer,
    )
    agent = PlannerAgent()
    request = ToolCallRequest(
        tool_name="knowledge_search",
        arguments={"query": "enterprise multi-agent platform"},
    )
    context = ToolContext(
        request_id="req-tool-1",
        agent_id=agent.metadata.agent_id,
    )

    result = asyncio.run(
        executor.execute(agent=agent, request=request, context=context)
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert [e[0] for e in tracer.events] == ["start_tool", "finish_tool"]


def test_langfuse_tracer_disabled_chain_spans_are_noop() -> None:
    tracer = LangfuseTracer(
        ObservabilitySettings(
            langfuse_host="https://cloud.langfuse.com",
            langfuse_public_key="pk-xxx",
            langfuse_secret_key="",  # missing secret -> disabled
        )
    )
    assert tracer.enabled is False
    assert tracer.start_workflow_span(object()) is None
    assert tracer.start_agent_span("planner_agent", 1, object()) is None
    assert tracer.start_tool_span("knowledge_search", "planner_agent", object()) is None
    tracer.finish_workflow_span(None, object())
    tracer.finish_agent_span(None, object())
    tracer.finish_tool_span(None, object())
