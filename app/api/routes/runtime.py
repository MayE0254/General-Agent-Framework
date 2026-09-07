from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_memory_service, get_runtime_query_service
from app.schemas import (
    AgentRunLogResponse,
    AgentRunLogListResponse,
    LLMCallLogResponse,
    LLMCallLogListResponse,
    MemoryDeleteResponse,
    MemoryEntryCreateRequest,
    MemoryEntryResponse,
    MemoryEntryUpdateRequest,
    MemoryListResponse,
    PaginationMetaResponse,
    RuntimeTaskDetailResponse,
    SessionDetailResponse,
    SessionListResponse,
    SessionMessageListResponse,
    SessionMessageResponse,
    SessionResponse,
    TaskRunResponse,
    TaskRunListResponse,
    ToolCallLogResponse,
    ToolCallLogListResponse,
    WorkflowRunDetailResponse,
    WorkflowRunListResponse,
    WorkflowRunRecordResponse,
)
from app.services import MemoryService, RuntimeQueryService

router = APIRouter(tags=["runtime"])


def _to_task_run_response(record) -> TaskRunResponse:
    trace_id = record.metadata_payload.get("trace_id")
    terminal_reason = record.metadata_payload.get("terminal_reason")
    review_id = record.metadata_payload.get("review_id")
    return TaskRunResponse(
        request_id=record.request_id,
        session_id=record.session_id,
        workflow_id=record.workflow_id,
        trace_id=trace_id,
        task_type=record.task_type,
        status=record.status,
        terminal_reason=terminal_reason,
        initial_agent_id=record.initial_agent_id,
        final_agent_id=record.final_agent_id,
        review_id=review_id,
        summary=record.summary,
        errors=record.errors,
        metadata_payload=record.metadata_payload,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _to_agent_run_response(record) -> AgentRunLogResponse:
    return AgentRunLogResponse(
        request_id=record.request_id,
        session_id=record.session_id,
        workflow_id=record.workflow_id,
        trace_id=record.trace_id,
        step_index=record.step_index,
        agent_id=record.agent_id,
        status=record.status,
        next_agent_id=record.next_agent_id,
        summary=record.summary,
        output=record.output,
        errors=record.errors,
        started_at_runtime=record.started_at_runtime,
        completed_at_runtime=record.completed_at_runtime,
        created_at=record.created_at,
    )


def _to_tool_call_response(record) -> ToolCallLogResponse:
    error_category = record.metrics.get("error_category")
    retryable = bool(record.metrics.get("retryable", False))
    error_detail = record.metrics.get("error_detail")
    return ToolCallLogResponse(
        request_id=record.request_id,
        session_id=record.session_id,
        workflow_id=record.workflow_id,
        trace_id=record.trace_id,
        agent_id=record.agent_id,
        tool_name=record.tool_name,
        status=record.status,
        message=record.message,
        arguments=record.arguments,
        content=record.content,
        errors=record.errors,
        error_category=error_category,
        retryable=retryable,
        error_detail=error_detail,
        metrics=record.metrics,
        created_at=record.created_at,
    )


def _to_llm_call_response(record) -> LLMCallLogResponse:
    return LLMCallLogResponse(
        request_id=record.request_id,
        session_id=record.session_id,
        workflow_id=record.workflow_id,
        trace_id=record.trace_id,
        agent_id=record.agent_id,
        model=record.model,
        status=record.status,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        total_tokens=record.total_tokens,
        latency_ms=record.latency_ms,
        error=record.error,
        cost_usd=record.cost_usd,
        created_at=record.created_at,
    )


def _to_workflow_run_response(record) -> WorkflowRunRecordResponse:
    trace_id = record.metadata_payload.get("trace_id")
    terminal_reason = record.metadata_payload.get("terminal_reason")
    review_id = record.metadata_payload.get("review_id")
    return WorkflowRunRecordResponse(
        request_id=record.request_id,
        workflow_id=record.workflow_id,
        trace_id=trace_id,
        initial_agent_id=record.initial_agent_id,
        current_agent_id=record.current_agent_id,
        status=record.status,
        terminal_reason=terminal_reason,
        execution_path=record.execution_path,
        final_agent_id=record.final_agent_id,
        review_id=review_id,
        summary=record.summary,
        errors=record.errors,
        metadata_payload=record.metadata_payload,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _to_session_response(record) -> SessionResponse:
    return SessionResponse(
        session_id=record.session_id,
        user_id=record.user_id,
        status=record.status,
        last_request_id=record.last_request_id,
        metadata_payload=record.metadata_payload,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.get("/runtime/tasks", response_model=TaskRunListResponse)
def list_runtime_tasks(
    status: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    workflow_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: RuntimeQueryService = Depends(get_runtime_query_service),
) -> TaskRunListResponse:
    result = service.list_task_runs(
        status=status,
        session_id=session_id,
        workflow_id=workflow_id,
        limit=limit,
        offset=offset,
    )
    return TaskRunListResponse(
        items=[_to_task_run_response(item) for item in result["items"]],
        pagination=PaginationMetaResponse(
            total=result["total"],
            limit=result["limit"],
            offset=result["offset"],
            filters=result["filters"],
        ),
    )


@router.get("/runtime/tasks/{request_id}", response_model=RuntimeTaskDetailResponse)
def get_runtime_task_detail(
    request_id: str,
    service: RuntimeQueryService = Depends(get_runtime_query_service),
) -> RuntimeTaskDetailResponse:
    detail = service.get_task_run_detail(request_id)
    return RuntimeTaskDetailResponse(
        task_run=_to_task_run_response(detail["task_run"]),
        agent_runs=[_to_agent_run_response(item) for item in detail["agent_runs"]],
        tool_calls=[_to_tool_call_response(item) for item in detail["tool_calls"]],
        llm_calls=[_to_llm_call_response(item) for item in detail["llm_calls"]],
        aggregates=detail["aggregates"],
    )


@router.get("/runtime/agent-runs", response_model=AgentRunLogListResponse)
def list_runtime_agent_runs(
    request_id: str | None = Query(default=None),
    agent_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: RuntimeQueryService = Depends(get_runtime_query_service),
) -> AgentRunLogListResponse:
    result = service.list_agent_runs(
        request_id=request_id,
        agent_id=agent_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    return AgentRunLogListResponse(
        items=[_to_agent_run_response(item) for item in result["items"]],
        pagination=PaginationMetaResponse(
            total=result["total"],
            limit=result["limit"],
            offset=result["offset"],
            filters=result["filters"],
        ),
    )


@router.get("/runtime/tool-calls", response_model=ToolCallLogListResponse)
def list_runtime_tool_calls(
    request_id: str | None = Query(default=None),
    agent_id: str | None = Query(default=None),
    tool_name: str | None = Query(default=None),
    status: str | None = Query(default=None),
    error_category: str | None = Query(default=None),
    retryable: bool | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: RuntimeQueryService = Depends(get_runtime_query_service),
) -> ToolCallLogListResponse:
    result = service.list_tool_calls(
        request_id=request_id,
        agent_id=agent_id,
        tool_name=tool_name,
        status=status,
        error_category=error_category,
        retryable=retryable,
        limit=limit,
        offset=offset,
    )
    return ToolCallLogListResponse(
        items=[_to_tool_call_response(item) for item in result["items"]],
        pagination=PaginationMetaResponse(
            total=result["total"],
            limit=result["limit"],
            offset=result["offset"],
            filters=result["filters"],
        ),
    )


@router.get("/runtime/llm-calls", response_model=LLMCallLogListResponse)
def list_runtime_llm_calls(
    request_id: str | None = Query(default=None),
    agent_id: str | None = Query(default=None),
    model: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    service: RuntimeQueryService = Depends(get_runtime_query_service),
) -> LLMCallLogListResponse:
    result = service.list_llm_calls(
        request_id=request_id,
        agent_id=agent_id,
        model=model,
        status=status,
        limit=limit,
        offset=offset,
    )
    return LLMCallLogListResponse(
        items=[_to_llm_call_response(item) for item in result["items"]],
        pagination=PaginationMetaResponse(
            total=result["total"],
            limit=result["limit"],
            offset=result["offset"],
            filters=result["filters"],
        ),
    )


@router.get("/runtime/workflow-runs", response_model=WorkflowRunListResponse)
def list_runtime_workflow_runs(
    request_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: RuntimeQueryService = Depends(get_runtime_query_service),
) -> WorkflowRunListResponse:
    result = service.list_workflow_runs(
        request_id=request_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    return WorkflowRunListResponse(
        items=[_to_workflow_run_response(item) for item in result["items"]],
        pagination=PaginationMetaResponse(
            total=result["total"],
            limit=result["limit"],
            offset=result["offset"],
            filters=result["filters"],
        ),
    )


@router.get(
    "/runtime/workflow-runs/{request_id}",
    response_model=WorkflowRunDetailResponse,
)
def get_runtime_workflow_run_detail(
    request_id: str,
    service: RuntimeQueryService = Depends(get_runtime_query_service),
) -> WorkflowRunDetailResponse:
    detail = service.get_workflow_run_detail(request_id)
    return WorkflowRunDetailResponse(
        workflow_run=_to_workflow_run_response(detail["workflow_run"]),
        agent_runs=[_to_agent_run_response(item) for item in detail["agent_runs"]],
        llm_calls=[_to_llm_call_response(item) for item in detail["llm_calls"]],
        aggregates=detail["aggregates"],
    )


@router.get("/runtime/sessions", response_model=SessionListResponse)
def list_runtime_sessions(
    session_id: str | None = Query(default=None),
    user_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: RuntimeQueryService = Depends(get_runtime_query_service),
) -> SessionListResponse:
    result = service.list_sessions(
        session_id=session_id,
        user_id=user_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    return SessionListResponse(
        items=[_to_session_response(item) for item in result["items"]],
        pagination=PaginationMetaResponse(
            total=result["total"],
            limit=result["limit"],
            offset=result["offset"],
            filters=result["filters"],
        ),
    )


@router.get(
    "/runtime/sessions/{session_id}",
    response_model=SessionDetailResponse,
)
def get_runtime_session_detail(
    session_id: str,
    service: RuntimeQueryService = Depends(get_runtime_query_service),
) -> SessionDetailResponse:
    detail = service.get_session_detail(session_id)
    return SessionDetailResponse(
        session=_to_session_response(detail["session"]),
        task_runs=[_to_task_run_response(item) for item in detail["task_runs"]],
        aggregates=detail["aggregates"],
    )


@router.get(
    "/runtime/sessions/{session_id}/messages",
    response_model=SessionMessageListResponse,
)
def list_session_messages(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: MemoryService = Depends(get_memory_service),
) -> SessionMessageListResponse:
    items = service.list_session_messages(
        session_id, limit=limit + offset
    )
    return SessionMessageListResponse(
        session_id=session_id,
        items=[
            SessionMessageResponse(
                id=item.id,
                session_id=item.session_id,
                request_id=item.request_id,
                role=item.role,
                content=item.content,
                step_index=item.step_index,
                metadata_payload=item.metadata_payload,
                created_at=item.created_at,
            )
            for item in items[offset:]
        ],
        pagination=PaginationMetaResponse(
            total=service.count_session_messages(session_id),
            limit=limit,
            offset=offset,
            filters={"session_id": session_id},
        ),
    )


@router.get("/runtime/memories", response_model=MemoryListResponse)
def list_memories(
    scope: str | None = Query(default=None),
    scope_key: str | None = Query(default=None),
    memory_type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: MemoryService = Depends(get_memory_service),
) -> MemoryListResponse:
    result = service.list_memories(
        scope=scope,
        scope_key=scope_key,
        memory_type=memory_type,
        limit=limit,
        offset=offset,
    )
    return MemoryListResponse(
        items=[_to_memory_response(item) for item in result["items"]],
        pagination=PaginationMetaResponse(
            total=result["total"],
            limit=result["limit"],
            offset=result["offset"],
            filters=result["filters"],
        ),
    )


@router.post(
    "/runtime/memories",
    response_model=MemoryEntryResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_memory(
    payload: MemoryEntryCreateRequest,
    service: MemoryService = Depends(get_memory_service),
) -> MemoryEntryResponse:
    record = service.add_memory(
        memory_type=payload.memory_type,
        content=payload.content,
        scope=payload.scope,
        scope_key=payload.scope_key,
        source=payload.source,
        importance=payload.importance,
        tags=payload.tags,
        metadata_payload=payload.metadata_payload,
    )
    return _to_memory_response(record)


@router.patch(
    "/runtime/memories/{memory_id}",
    response_model=MemoryEntryResponse,
)
def update_memory(
    memory_id: str,
    payload: MemoryEntryUpdateRequest,
    service: MemoryService = Depends(get_memory_service),
) -> MemoryEntryResponse:
    updates = payload.model_dump(exclude_unset=True)
    record = service.update_memory(memory_id, **updates)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory entry '{memory_id}' was not found.",
        )
    return _to_memory_response(record)


@router.delete(
    "/runtime/memories/{memory_id}",
    response_model=MemoryDeleteResponse,
)
def delete_memory(
    memory_id: str,
    service: MemoryService = Depends(get_memory_service),
) -> MemoryDeleteResponse:
    deleted = service.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory entry '{memory_id}' was not found.",
        )
    return MemoryDeleteResponse(memory_id=memory_id, deleted=True)


def _to_memory_response(record) -> MemoryEntryResponse:
    return MemoryEntryResponse(
        id=record.id,
        memory_type=record.memory_type,
        scope=record.scope,
        scope_key=record.scope_key,
        content=record.content,
        source=record.source,
        importance=record.importance,
        tags=record.tags,
        metadata_payload=record.metadata_payload,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
