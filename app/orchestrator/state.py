from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agents.base.schemas import AgentContext, AgentExecutionStatus, AgentResult, LLMProfile
from app.orchestrator.shared_state import normalize_shared_state


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WorkflowStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


class WorkflowTerminalReason(StrEnum):
    NATURAL_COMPLETION = "natural_completion"
    AGENT_FAILED = "agent_failed"
    AGENT_EXCEPTION = "agent_exception"
    AGENT_NOT_REGISTERED = "agent_not_registered"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    TOOL_FAILED = "tool_failed"
    TOOL_EXECUTOR_UNAVAILABLE = "tool_executor_unavailable"
    TOOL_OWNER_NOT_REGISTERED = "tool_owner_not_registered"


class AgentRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: int = Field(..., ge=1)
    agent_id: str = Field(..., min_length=3, max_length=64)
    status: AgentExecutionStatus
    summary: str = Field(default="", max_length=2000)
    next_agent_id: str | None = Field(default=None, max_length=64)
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime = Field(default_factory=utc_now)
    output: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class OrchestratorState(BaseModel):
    model_config = ConfigDict(extra="allow")

    request_id: str = Field(..., min_length=1, max_length=128)
    initial_agent_id: str = Field(..., min_length=3, max_length=64)
    session_id: str | None = Field(default=None, max_length=128)
    user_id: str | None = Field(default=None, max_length=128)
    project_key: str = Field(default="default", max_length=128)
    workflow_id: str | None = Field(default=None, max_length=128)
    trace_id: str | None = Field(default=None, max_length=128)
    input_text: str | None = Field(default=None)
    structured_input: dict[str, Any] = Field(default_factory=dict)
    shared_state: dict[str, Any] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)
    knowledge_refs: list[str] = Field(default_factory=list)
    current_agent_id: str | None = Field(default=None, max_length=64)
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    terminal_reason: WorkflowTerminalReason | None = Field(default=None)
    execution_path: list[str] = Field(default_factory=list)
    run_history: list[AgentRunRecord] = Field(default_factory=list)
    final_result: AgentResult | None = Field(default=None)
    errors: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=10, ge=1, le=100)
    # Request-level model routing override propagated to every agent context.
    llm_profile: LLMProfile | None = Field(default=None)

    @model_validator(mode="after")
    def _normalize_protocol_state(self) -> "OrchestratorState":
        self.shared_state = normalize_shared_state(self.shared_state)
        return self

    def to_agent_context(self) -> AgentContext:
        if self.current_agent_id is None:
            raise ValueError("current_agent_id must be set before building context.")

        return AgentContext(
            request_id=self.request_id,
            session_id=self.session_id,
            user_id=self.user_id,
            workflow_id=self.workflow_id,
            trace_id=self.trace_id,
            input_text=self.input_text,
            structured_input=self.structured_input,
            shared_state=self.shared_state,
            memory=self.memory,
            knowledge_refs=self.knowledge_refs,
            llm_profile=self.llm_profile,
        )
