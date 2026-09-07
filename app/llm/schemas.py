from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LLMRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LLMMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: LLMRole
    content: str = Field(..., max_length=200_000)
    name: str | None = Field(default=None, max_length=128)
    tool_call_id: str | None = Field(default=None, max_length=128)


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., min_length=1, max_length=128)
    messages: list[LLMMessage] = Field(..., min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    timeout_seconds: int = Field(default=60, ge=1, le=600)
    # Optional named provider (from [llm].providers) for multi-model routing.
    # When unset the default [llm] block is used.
    provider: str | None = Field(default=None, max_length=64)
    # Audit context: populated by callers so LLM call logs can be traced
    # back to the originating workflow step and request.
    request_id: str | None = Field(default=None, max_length=128)
    session_id: str | None = Field(default=None, max_length=128)
    workflow_id: str | None = Field(default=None, max_length=128)
    trace_id: str | None = Field(default=None, max_length=128)
    agent_id: str | None = Field(default=None, max_length=64)


class LLMUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., min_length=1, max_length=128)
    content: str = Field(default="", max_length=500_000)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    latency_ms: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    raw: dict[str, Any] = Field(default_factory=dict)


class LLMError(RuntimeError):
    """Raised when an LLM call fails or the service is not configured."""


class LLMCallAudit(BaseModel):
    """Structured audit event emitted once per LLM call.

    Emitted by LLMService regardless of outcome; a recorder (usually backed
    by the runtime audit store) decides how to persist it. Kept independent
    of the persistence layer so ``app.llm`` stays storage-agnostic.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    session_id: str | None = None
    workflow_id: str | None = None
    trace_id: str | None = None
    agent_id: str | None = None
    model: str = Field(..., min_length=1, max_length=128)
    status: str = Field(..., pattern="^(success|failed)$")
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
