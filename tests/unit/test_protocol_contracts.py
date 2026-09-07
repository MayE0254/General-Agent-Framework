import pytest
from pydantic import ValidationError

from app.agents.base.schemas import AgentContext, AgentResult
from app.tools.schemas import ToolCallRequest, ToolContext, ToolResult


def test_agent_context_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AgentContext(
            request_id="req-1",
            rogue="value",
        )


def test_agent_result_validates_next_agent_id_pattern() -> None:
    with pytest.raises(ValidationError):
        AgentResult(
            agent_id="planner_agent",
            next_agent_id="Executor-Agent",
        )


def test_tool_contracts_validate_identifiers() -> None:
    with pytest.raises(ValidationError):
        ToolContext(
            request_id="req-1",
            agent_id="Planner-Agent",
        )

    with pytest.raises(ValidationError):
        ToolCallRequest(tool_name="Document-Parser")

    with pytest.raises(ValidationError):
        ToolResult(tool_name="Document-Parser")
