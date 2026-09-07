import asyncio

from app.agents.base.agent import BaseAgent
from app.agents.base.schemas import AgentContext, AgentMetadata, AgentResult
from app.agents.system import KnowledgeAgent, PlannerAgent
from app.tools import (
    BaseTool,
    ToolCallRequest,
    ToolContext,
    ToolExecutionStatus,
    ToolExecutor,
    ToolErrorCategory,
    ToolMetadata,
    ToolRegistry,
    ToolResult,
)


class FakeKnowledgeSearchTool(BaseTool):
    @classmethod
    def build_metadata(cls) -> ToolMetadata:
        return ToolMetadata(
            tool_name="knowledge_search",
            description="Returns a fake search payload.",
            tags=["test"],
        )

    async def _run(self, context: ToolContext, arguments: dict) -> ToolResult:
        return ToolResult(
            tool_name=self.metadata.tool_name,
            status=ToolExecutionStatus.SUCCESS,
            content={
                "agent_id": context.agent_id,
                "query": arguments.get("query"),
            },
            message="fake tool executed",
        )


class CrashingTool(BaseTool):
    @classmethod
    def build_metadata(cls) -> ToolMetadata:
        return ToolMetadata(
            tool_name="crashing_tool",
            description="Raises a runtime error.",
            tags=["test"],
        )

    async def _run(self, context: ToolContext, arguments: dict) -> ToolResult:
        raise RuntimeError("tool boom")


class TimeoutTool(BaseTool):
    @classmethod
    def build_metadata(cls) -> ToolMetadata:
        return ToolMetadata(
            tool_name="timeout_tool",
            description="Raises a timeout error.",
            tags=["test"],
        )

    async def _run(self, context: ToolContext, arguments: dict) -> ToolResult:
        raise TimeoutError("tool timed out")


class PermissiveToolAgent(BaseAgent):
    @classmethod
    def build_metadata(cls) -> AgentMetadata:
        return AgentMetadata(
            agent_id="permissive_tool_agent",
            agent_name="Permissive Tool Agent",
            agent_role="test",
            allowed_tools=["missing_tool", "crashing_tool", "timeout_tool"],
        )

    async def _run(self, context: AgentContext) -> AgentResult:
        return AgentResult(agent_id=self.metadata.agent_id)


def test_tool_executor_allows_permitted_tool() -> None:
    registry = ToolRegistry()
    registry.register(FakeKnowledgeSearchTool)
    executor = ToolExecutor(registry)
    agent = PlannerAgent()

    result = asyncio.run(
        executor.execute(
            agent=agent,
            request=ToolCallRequest(
                tool_name="knowledge_search",
                arguments={"query": "multi-agent framework"},
            ),
            context=ToolContext(
                request_id="req-1",
                agent_id=agent.metadata.agent_id,
            ),
        )
    )

    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.content["query"] == "multi-agent framework"


def test_tool_executor_denies_unpermitted_tool() -> None:
    registry = ToolRegistry()
    registry.register(FakeKnowledgeSearchTool)
    executor = ToolExecutor(registry)
    agent = KnowledgeAgent()

    result = asyncio.run(
        executor.execute(
            agent=agent,
            request=ToolCallRequest(
                tool_name="knowledge_search",
                arguments={"query": "denied"},
            ),
            context=ToolContext(
                request_id="req-2",
                agent_id=agent.metadata.agent_id,
            ),
        )
    )

    assert result.status == ToolExecutionStatus.DENIED
    assert result.errors == ["tool_permission_denied"]
    assert result.error_category == ToolErrorCategory.POLICY_DENIED
    assert result.retryable is False


def test_tool_executor_returns_configuration_failure_for_unknown_tool() -> None:
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    agent = PermissiveToolAgent()

    result = asyncio.run(
        executor.execute(
            agent=agent,
            request=ToolCallRequest(
                tool_name="missing_tool",
                arguments={"query": "missing"},
            ),
            context=ToolContext(
                request_id="req-3",
                agent_id=agent.metadata.agent_id,
            ),
        )
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.errors == ["tool_not_registered"]
    assert result.error_category == ToolErrorCategory.CONFIGURATION
    assert result.retryable is False


def test_tool_executor_wraps_tool_exception() -> None:
    registry = ToolRegistry()
    registry.register(CrashingTool)
    executor = ToolExecutor(registry)
    agent = PermissiveToolAgent()

    result = asyncio.run(
        executor.execute(
            agent=agent,
            request=ToolCallRequest(
                tool_name="crashing_tool",
                arguments={},
            ),
            context=ToolContext(
                request_id="req-4",
                agent_id=agent.metadata.agent_id,
            ),
        )
    )

    assert result.status == ToolExecutionStatus.FAILED
    assert result.errors == ["tool_execution_exception"]
    assert result.error_category == ToolErrorCategory.EXECUTION_EXCEPTION
    assert result.retryable is False
    assert "RuntimeError: tool boom" == result.error_detail


def test_tool_executor_wraps_tool_timeout() -> None:
    registry = ToolRegistry()
    registry.register(TimeoutTool)
    executor = ToolExecutor(registry)
    agent = PermissiveToolAgent()

    result = asyncio.run(
        executor.execute(
            agent=agent,
            request=ToolCallRequest(
                tool_name="timeout_tool",
                arguments={},
            ),
            context=ToolContext(
                request_id="req-5",
                agent_id=agent.metadata.agent_id,
            ),
        )
    )

    assert result.status == ToolExecutionStatus.TIMEOUT
    assert result.errors == ["tool_timeout"]
    assert result.error_category == ToolErrorCategory.TIMEOUT
    assert result.retryable is True
