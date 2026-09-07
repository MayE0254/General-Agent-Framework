from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolExecutionStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    DENIED = "denied"
    TIMEOUT = "timeout"


class ToolErrorCategory(StrEnum):
    INPUT_VALIDATION = "input_validation"
    POLICY_DENIED = "policy_denied"
    CONFIGURATION = "configuration"
    TIMEOUT = "timeout"
    EXTERNAL_DEPENDENCY = "external_dependency"
    EXECUTION_EXCEPTION = "execution_exception"


class ToolMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable and unique tool identifier.",
    )
    description: str = Field(default="", max_length=512)
    version: str = Field(default="0.1.0", max_length=32)
    timeout_seconds: int = Field(default=30, ge=1, le=3600)
    tags: list[str] = Field(default_factory=list)


class ToolContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., min_length=1, max_length=128)
    agent_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    session_id: str | None = Field(default=None, max_length=128)
    trace_id: str | None = Field(default=None, max_length=128)
    shared_state: dict[str, Any] = Field(default_factory=dict)


class ToolCallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    status: ToolExecutionStatus = Field(default=ToolExecutionStatus.SUCCESS)
    content: dict[str, Any] = Field(default_factory=dict)
    message: str = Field(default="", max_length=2000)
    errors: list[str] = Field(default_factory=list)
    error_category: ToolErrorCategory | None = Field(default=None)
    retryable: bool = Field(default=False)
    error_detail: str | None = Field(default=None, max_length=2000)
    metrics: dict[str, Any] = Field(default_factory=dict)
