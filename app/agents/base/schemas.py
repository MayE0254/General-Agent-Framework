from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from app.llm import LLMService


class AgentKind(StrEnum):
    SYSTEM = "system"
    DOMAIN = "domain"


class AgentExecutionStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class LLMProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str | None = Field(default=None, description="LLM service provider.")
    model: str | None = Field(default=None, description="LLM model identifier.")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)


class AgentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable and unique agent identifier.",
    )
    agent_name: str = Field(..., min_length=1, max_length=128)
    agent_role: str = Field(..., min_length=1, max_length=128)
    agent_kind: AgentKind = Field(default=AgentKind.SYSTEM)
    description: str = Field(default="", max_length=512)
    version: str = Field(default="0.1.0", max_length=32)
    allowed_tools: list[str] = Field(default_factory=list)
    llm_profile: LLMProfile = Field(default_factory=LLMProfile)
    tags: list[str] = Field(default_factory=list)


class AgentContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., min_length=1, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)
    user_id: str | None = Field(default=None, max_length=128)
    workflow_id: str | None = Field(default=None, max_length=128)
    trace_id: str | None = Field(default=None, max_length=128)
    input_text: str | None = Field(default=None)
    structured_input: dict[str, Any] = Field(default_factory=dict)
    shared_state: dict[str, Any] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)
    knowledge_refs: list[str] = Field(default_factory=list)
    # Optional request-level LLM profile override for multi-model routing.
    # When set it wins over the agent's own llm_profile for provider/model.
    llm_profile: LLMProfile | None = Field(default=None)
    llm_service: Any | None = Field(
        default=None,
        description="Optional LLMService for agents that need model calls.",
    )


class PendingToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(..., min_length=3, max_length=64)
    status: AgentExecutionStatus = Field(default=AgentExecutionStatus.SUCCESS)
    summary: str = Field(default="", max_length=2000)
    output: dict[str, Any] = Field(default_factory=dict)
    messages: list[str] = Field(default_factory=list)
    tool_calls: list[PendingToolCall] = Field(default_factory=list)
    next_agent_id: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    requires_human: bool = Field(default=False)
    errors: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
