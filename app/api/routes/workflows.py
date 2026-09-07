from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from celery.result import AsyncResult

from app.api.dependencies import (
    get_app_settings,
    get_async_workflow_submission_service,
    get_orchestrator,
)
from app.core import NotFoundError
from app.infra import is_redis_reachable, probe_redis_connection
from app.orchestrator import (
    OrchestratorState,
    SimpleOrchestrator,
    WorkflowStatus,
    WorkflowTerminalReason,
)
from app.schemas import (
    AsyncWorkflowSubmissionListResponse,
    AsyncWorkflowSubmissionResponse,
    PaginationMetaResponse,
    WorkflowRunAsyncResponse,
    WorkflowRunRequest,
    WorkflowRunResponse,
    WorkflowTaskStatusResponse,
)
from app.services import AsyncWorkflowSubmissionService
from app.tasks.celery_app import celery_app

router = APIRouter(tags=["workflows"])

_REDIS_UNAVAILABLE = (
    "Redis is not reachable. Start the Redis server before using async "
    "workflow submission."
)


def _to_async_submission_response(record) -> AsyncWorkflowSubmissionResponse:
    workflow_status = _derive_workflow_status(
        queue_status=record.status,
        result_payload=record.result_payload,
    )
    return AsyncWorkflowSubmissionResponse(
        task_id=record.task_id,
        request_id=record.request_id,
        session_id=record.session_id,
        user_id=record.user_id,
        initial_agent_id=record.initial_agent_id,
        task_name=record.task_name,
        queue_name=record.queue_name,
        status=record.status,
        workflow_status=workflow_status,
        terminal_reason=_derive_terminal_reason(record.result_payload),
        review_id=_derive_review_id(record.result_payload),
        message=record.message,
        result_payload=record.result_payload,
        traceback=record.traceback,
        metadata_payload=record.metadata_payload,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _derive_workflow_status(
    *,
    queue_status: str,
    result_payload: dict | None,
) -> WorkflowStatus | None:
    normalized = (queue_status or "").lower()
    if normalized == "pending":
        return WorkflowStatus.PENDING
    if normalized in {"received", "started", "retry"}:
        return WorkflowStatus.RUNNING
    if normalized == "failure":
        return WorkflowStatus.FAILED
    if normalized == "success":
        result_status = (result_payload or {}).get("status")
        if result_status:
            try:
                return WorkflowStatus(result_status)
            except ValueError:
                return WorkflowStatus.COMPLETED
        return WorkflowStatus.COMPLETED
    return None


def _derive_terminal_reason(
    result_payload: dict | None,
) -> WorkflowTerminalReason | None:
    raw = (result_payload or {}).get("terminal_reason")
    if raw is None:
        return None
    try:
        return WorkflowTerminalReason(raw)
    except ValueError:
        return None


def _derive_review_id(result_payload: dict | None) -> str | None:
    review_id = (result_payload or {}).get("review_id")
    return str(review_id) if review_id else None


@router.post("/workflows/run", response_model=WorkflowRunResponse)
async def run_workflow(
    request: WorkflowRunRequest,
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator),
) -> WorkflowRunResponse:
    state = OrchestratorState(
        request_id=request.request_id,
        initial_agent_id=request.initial_agent_id,
        session_id=request.session_id,
        user_id=request.user_id,
        workflow_id=request.workflow_id,
        trace_id=request.trace_id,
        input_text=request.input_text,
        structured_input=request.structured_input,
        shared_state=request.shared_state,
        memory=request.memory,
        knowledge_refs=request.knowledge_refs,
        max_steps=request.max_steps,
        llm_profile=request.llm_profile,
    )
    result_state = await orchestrator.run(state)

    return WorkflowRunResponse(
        request_id=result_state.request_id,
        status=result_state.status,
        terminal_reason=result_state.terminal_reason,
        review_id=result_state.memory.get("human_review", {}).get("review_id"),
        source_review_id=result_state.memory.get("human_review", {}).get(
            "source_review_id"
        ),
        execution_path=result_state.execution_path,
        errors=result_state.errors,
        final_result=result_state.final_result,
    )


@router.post(
    "/workflows/run/async",
    response_model=WorkflowRunAsyncResponse,
    status_code=202,
)
async def run_workflow_async(
    request: WorkflowRunRequest,
    settings=Depends(get_app_settings),
    submission_service: AsyncWorkflowSubmissionService = Depends(
        get_async_workflow_submission_service
    ),
) -> WorkflowRunAsyncResponse:
    """Submit a workflow to the Celery queue and return immediately."""
    redis_probe = probe_redis_connection(settings)
    if not redis_probe["reachable"]:
        detail = _REDIS_UNAVAILABLE
        if redis_probe["error_type"]:
            detail = (
                f"{detail} Probe failed: "
                f"{redis_probe['error_type']}: {redis_probe['error_message']}"
            )
        raise HTTPException(status_code=503, detail=detail)

    if submission_service.get_by_request_id(request.request_id) is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                "request_id already exists in async workflow task center. "
                "Use a new request_id before resubmitting."
            ),
        )

    task_id = str(uuid4())
    async_result = celery_app.send_task(
        "workflow.run",
        args=[request.model_dump(mode="json")],
        task_id=task_id,
    )
    submission = submission_service.create_submission(
        request,
        task_id=async_result.id,
        task_name="workflow.run",
        queue_name=settings.celery.task_default_queue,
    )
    return WorkflowRunAsyncResponse(
        task_id=async_result.id,
        request_id=submission.request_id,
        status=submission.status,
        workflow_status=WorkflowStatus.PENDING,
        terminal_reason=None,
        review_id=None,
        message=submission.message,
    )


@router.get("/workflows/tasks/{task_id}", response_model=WorkflowTaskStatusResponse)
async def get_workflow_task_status(
    task_id: str,
    settings=Depends(get_app_settings),
    submission_service: AsyncWorkflowSubmissionService = Depends(
        get_async_workflow_submission_service
    ),
) -> WorkflowTaskStatusResponse:
    """Poll the result of an async workflow submission."""
    record = None
    if is_redis_reachable(settings):
        try:
            record = submission_service.refresh_submission_status(task_id)
        except NotFoundError:
            async_result = AsyncResult(task_id, app=celery_app)
            status = async_result.status.lower()
            return WorkflowTaskStatusResponse(
                task_id=task_id,
                request_id=None,
                session_id=None,
                user_id=None,
                initial_agent_id=None,
                task_name="workflow.run",
                queue_name=settings.celery.task_default_queue,
                status=status,
                workflow_status=_derive_workflow_status(
                    queue_status=status,
                    result_payload=(
                        async_result.result
                        if isinstance(async_result.result, dict)
                        else None
                    ),
                ),
                terminal_reason=_derive_terminal_reason(
                    async_result.result if isinstance(async_result.result, dict) else None
                ),
                review_id=_derive_review_id(
                    async_result.result if isinstance(async_result.result, dict) else None
                ),
                message=None,
                result=(
                    async_result.result
                    if async_result.successful()
                    and isinstance(async_result.result, dict)
                    else None
                ),
                traceback=async_result.traceback if async_result.failed() else None,
                created_at=None,
                updated_at=None,
            )
    else:
        try:
            record = submission_service.get_submission(task_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=503, detail=_REDIS_UNAVAILABLE) from exc

    if record is None:
        raise HTTPException(status_code=404, detail="Async submission was not found.")

    return WorkflowTaskStatusResponse(
        task_id=task_id,
        request_id=record.request_id,
        session_id=record.session_id,
        user_id=record.user_id,
        initial_agent_id=record.initial_agent_id,
        task_name=record.task_name,
        queue_name=record.queue_name,
        status=record.status,
        workflow_status=_derive_workflow_status(
            queue_status=record.status,
            result_payload=record.result_payload,
        ),
        terminal_reason=_derive_terminal_reason(record.result_payload),
        review_id=_derive_review_id(record.result_payload),
        message=record.message,
        result=record.result_payload,
        traceback=record.traceback,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.get(
    "/workflows/submissions",
    response_model=AsyncWorkflowSubmissionListResponse,
)
async def list_async_workflow_submissions(
    status: str | None = Query(default=None),
    request_id: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    user_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    settings=Depends(get_app_settings),
    submission_service: AsyncWorkflowSubmissionService = Depends(
        get_async_workflow_submission_service
    ),
) -> AsyncWorkflowSubmissionListResponse:
    result = submission_service.list_submissions(
        status=status,
        request_id=request_id,
        session_id=session_id,
        user_id=user_id,
        limit=limit,
        offset=offset,
        refresh_statuses=is_redis_reachable(settings),
    )
    return AsyncWorkflowSubmissionListResponse(
        items=[_to_async_submission_response(item) for item in result["items"]],
        pagination=PaginationMetaResponse(
            total=result["total"],
            limit=result["limit"],
            offset=result["offset"],
            filters=result["filters"],
        ),
    )
