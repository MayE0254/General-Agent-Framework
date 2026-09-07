from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agents.base.schemas import AgentKind, AgentResult, LLMProfile
from app.orchestrator import WorkflowStatus, WorkflowTerminalReason


class AgentMetadataResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str
    agent_name: str
    agent_role: str
    agent_kind: AgentKind
    description: str
    allowed_tools: list[str]
    tags: list[str]


class WorkflowRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., min_length=1, max_length=128)
    initial_agent_id: str = Field(
        default="planner_agent",
        min_length=3,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    session_id: str | None = Field(default=None, max_length=128)
    user_id: str | None = Field(default=None, max_length=128)
    workflow_id: str | None = Field(default=None, max_length=128)
    trace_id: str | None = Field(default=None, max_length=128)
    input_text: str | None = Field(default=None)
    structured_input: dict[str, Any] = Field(default_factory=dict)
    shared_state: dict[str, Any] = Field(default_factory=dict)
    memory: dict[str, Any] = Field(default_factory=dict)
    knowledge_refs: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=10, ge=1, le=100)
    # Optional request-level model routing: overrides each agent's own
    # llm_profile (provider/model/temperature) for this run.
    llm_profile: LLMProfile | None = Field(default=None)


class WorkflowRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: WorkflowStatus
    terminal_reason: WorkflowTerminalReason | None = None
    review_id: str | None = None
    source_review_id: str | None = None
    execution_path: list[str]
    errors: list[str]
    final_result: AgentResult | None


class WorkflowRunAsyncResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    request_id: str
    status: str
    workflow_status: WorkflowStatus | None = None
    terminal_reason: WorkflowTerminalReason | None = None
    review_id: str | None = None
    message: str


class WorkflowTaskStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    request_id: str | None
    session_id: str | None
    user_id: str | None
    initial_agent_id: str | None
    task_name: str | None
    queue_name: str | None
    status: str
    workflow_status: WorkflowStatus | None = None
    terminal_reason: WorkflowTerminalReason | None = None
    review_id: str | None = None
    message: str | None
    result: dict[str, Any] | None
    traceback: str | None
    created_at: datetime | None
    updated_at: datetime | None


class AsyncWorkflowSubmissionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    request_id: str
    session_id: str | None
    user_id: str | None
    initial_agent_id: str
    task_name: str
    queue_name: str
    status: str
    workflow_status: WorkflowStatus | None = None
    terminal_reason: WorkflowTerminalReason | None = None
    review_id: str | None = None
    message: str
    result_payload: dict[str, Any] | None
    traceback: str | None
    metadata_payload: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class AsyncWorkflowSubmissionListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[AsyncWorkflowSubmissionResponse]
    pagination: PaginationMetaResponse


class TaskRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    session_id: str | None
    workflow_id: str | None
    trace_id: str | None
    task_type: str
    status: str
    terminal_reason: str | None
    initial_agent_id: str
    final_agent_id: str | None
    review_id: str | None
    summary: str
    errors: list[str]
    metadata_payload: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class AgentRunLogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    session_id: str | None
    workflow_id: str | None
    trace_id: str | None
    step_index: int
    agent_id: str
    status: str
    next_agent_id: str | None
    summary: str
    output: dict[str, Any]
    errors: list[str]
    started_at_runtime: datetime
    completed_at_runtime: datetime
    created_at: datetime


class ToolCallLogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    session_id: str | None
    workflow_id: str | None
    trace_id: str | None
    agent_id: str
    tool_name: str
    status: str
    message: str
    arguments: dict[str, Any]
    content: dict[str, Any]
    errors: list[str]
    error_category: str | None = None
    retryable: bool = False
    error_detail: str | None = None
    metrics: dict[str, Any]
    created_at: datetime


class LLMCallLogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None
    session_id: str | None
    workflow_id: str | None
    trace_id: str | None
    agent_id: str | None
    model: str
    status: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int
    cost_usd: float
    error: str | None
    created_at: datetime


class RuntimeAggregatesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_run_count: int = 0
    agent_run_count: int = 0
    tool_call_count: int = 0
    llm_call_count: int = 0
    total_llm_tokens: int = 0
    total_cost_usd: float = 0.0


class RuntimeTaskDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_run: TaskRunResponse
    agent_runs: list[AgentRunLogResponse]
    tool_calls: list[ToolCallLogResponse]
    llm_calls: list[LLMCallLogResponse]
    aggregates: RuntimeAggregatesResponse


class PaginationMetaResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int
    limit: int
    offset: int
    filters: dict[str, Any]


class TaskRunListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[TaskRunResponse]
    pagination: PaginationMetaResponse


class AgentRunLogListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[AgentRunLogResponse]
    pagination: PaginationMetaResponse


class ToolCallLogListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ToolCallLogResponse]
    pagination: PaginationMetaResponse


class LLMCallLogListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[LLMCallLogResponse]
    pagination: PaginationMetaResponse


class WorkflowRunRecordResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    workflow_id: str | None
    trace_id: str | None
    initial_agent_id: str
    current_agent_id: str | None
    status: str
    terminal_reason: str | None
    execution_path: list[str]
    final_agent_id: str | None
    review_id: str | None
    summary: str
    errors: list[str]
    metadata_payload: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class WorkflowRunListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[WorkflowRunRecordResponse]
    pagination: PaginationMetaResponse


class WorkflowRunDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_run: WorkflowRunRecordResponse
    agent_runs: list[AgentRunLogResponse]
    llm_calls: list[LLMCallLogResponse]
    aggregates: RuntimeAggregatesResponse


class SessionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    user_id: str | None
    status: str
    last_request_id: str | None
    metadata_payload: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class SessionListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[SessionResponse]
    pagination: PaginationMetaResponse


class SessionDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session: SessionResponse
    task_runs: list[TaskRunResponse]
    aggregates: RuntimeAggregatesResponse


class SessionMessageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    session_id: str
    request_id: str
    role: str
    content: str
    step_index: int
    metadata_payload: dict[str, Any]
    created_at: datetime


class SessionMessageListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    items: list[SessionMessageResponse]
    pagination: PaginationMetaResponse


class MemoryEntryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    memory_type: str
    scope: str
    scope_key: str
    content: str
    source: str
    importance: int
    tags: list[str]
    metadata_payload: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class MemoryEntryCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_type: str = Field(..., pattern="^(preference|fact|rule)$")
    content: str = Field(..., min_length=1, max_length=4000)
    scope: Literal["user", "project", "global"] = "user"
    scope_key: str = Field(..., min_length=1, max_length=128)
    source: Literal["user", "llm", "manual"] = "manual"
    importance: int = Field(default=3, ge=1, le=5)
    tags: list[str] = Field(default_factory=list)
    metadata_payload: dict[str, Any] = Field(default_factory=dict)


class MemoryEntryUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str | None = Field(default=None, min_length=1, max_length=4000)
    importance: int | None = Field(default=None, ge=1, le=5)
    tags: list[str] | None = Field(default=None)
    metadata_payload: dict[str, Any] | None = Field(default=None)


class MemoryListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[MemoryEntryResponse]
    pagination: PaginationMetaResponse


class MemoryDeleteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str
    deleted: bool


class HumanReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    request_id: str
    session_id: str | None
    user_id: str | None
    status: str
    agent_id: str
    summary: str
    payload: dict[str, Any]
    decided_by: str | None
    decision_note: str
    decided_at: datetime | None
    source_review_id: str | None = None
    continuation_request_id: str | None = None
    continuation_status: str | None = None
    continuation_terminal_reason: str | None = None
    continuation_run_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class HumanReviewListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[HumanReviewResponse]
    pagination: PaginationMetaResponse


class ReviewContinuationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    agent_id: str = Field(
        ...,
        min_length=3,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    )
    input_text: str = Field(..., min_length=1)
    structured_input: dict[str, Any] = Field(default_factory=dict)


class ReviewContinuationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    source_review_id: str
    source_request_id: str
    source_workflow_id: str | None
    source_trace_id: str | None
    source_workflow_status: WorkflowStatus | None = None
    source_terminal_reason: WorkflowTerminalReason | None = None
    resumed_agent_id: str
    decided_by: str | None
    decision_note: str


class ReviewDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decided_by: str | None = Field(default=None, max_length=128)
    note: str = Field(default="", max_length=2000)


class ApproveReviewRequest(ReviewDecisionRequest):
    model_config = ConfigDict(extra="forbid")

    continuation: ReviewContinuationRequest | None = Field(default=None)


class ApproveReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review: HumanReviewResponse
    continuation: ReviewContinuationResponse | None = Field(default=None)
    continuation_run: WorkflowRunResponse | None = Field(default=None)
