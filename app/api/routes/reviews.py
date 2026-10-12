from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import (
    AuthenticatedUser,
    get_current_user,
    get_human_review_service,
    get_orchestrator,
)
from app.orchestrator import WorkflowStatus, WorkflowTerminalReason
from app.schemas import (
    ApproveReviewRequest,
    ApproveReviewResponse,
    HumanReviewListResponse,
    HumanReviewResponse,
    PaginationMetaResponse,
    ReviewContinuationResponse,
    ReviewDecisionRequest,
    WorkflowRunResponse,
)
from app.services import HumanReviewService, event_to_sse_frame

router = APIRouter(tags=["reviews"])


def _ensure_review_operator(
    service: HumanReviewService,
    review_id: str,
    user: AuthenticatedUser,
) -> None:
    """Only the submitting user may decide a review; admins are read-only."""
    if user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrators review outcomes read-only; decisions belong to the submitting user.",
        )
    task = service.get_review(review_id)
    if task is None or task.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review task '{review_id}' was not found.",
        )


def _to_review_response(record) -> HumanReviewResponse:
    return HumanReviewResponse(
        id=record.id,
        request_id=record.request_id,
        session_id=record.session_id,
        user_id=record.user_id,
        status=record.status,
        agent_id=record.agent_id,
        summary=record.summary,
        payload=record.payload,
        decided_by=record.decided_by,
        decision_note=record.decision_note,
        decided_at=record.decided_at,
        source_review_id=record.source_review_id,
        continuation_request_id=record.continuation_request_id,
        continuation_status=record.continuation_status,
        continuation_terminal_reason=record.continuation_terminal_reason,
        continuation_run_at=record.continuation_run_at,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.get("/runtime/reviews", response_model=HumanReviewListResponse)
def list_reviews(
    status_filter: str | None = Query(default=None, alias="status"),
    user_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: HumanReviewService = Depends(get_human_review_service),
    user: AuthenticatedUser = Depends(get_current_user),
) -> HumanReviewListResponse:
    # Non-admin users only ever see review tasks they submitted; admins can
    # filter by member (user_id) to group review outcomes per user.
    effective_user_id = user.id if not user.is_admin else user_id
    result = service.list_reviews(
        status=status_filter,
        user_id=effective_user_id,
        limit=limit,
        offset=offset,
    )
    return HumanReviewListResponse(
        items=[_to_review_response(item) for item in result["items"]],
        pagination=PaginationMetaResponse(
            total=result["total"],
            limit=result["limit"],
            offset=result["offset"],
            filters=result["filters"],
        ),
    )


@router.get("/runtime/reviews/events")
async def stream_review_events(
    service: HumanReviewService = Depends(get_human_review_service),
    _: AuthenticatedUser = Depends(get_current_user),
) -> StreamingResponse:
    """SSE stream of review lifecycle events (created/decided) for UI prompts."""

    async def event_stream():
        async for event in service.stream_events():
            yield event_to_sse_frame(event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/runtime/reviews/{review_id}", response_model=HumanReviewResponse)
def get_review(
    review_id: str,
    service: HumanReviewService = Depends(get_human_review_service),
    user: AuthenticatedUser = Depends(get_current_user),
) -> HumanReviewResponse:
    task = service.get_review(review_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review task '{review_id}' was not found.",
        )
    if not user.is_admin and task.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review task '{review_id}' was not found.",
        )
    return _to_review_response(task)


@router.post(
    "/runtime/reviews/{review_id}/approve",
    response_model=ApproveReviewResponse,
)
async def approve_review(
    review_id: str,
    payload: ApproveReviewRequest,
    service: HumanReviewService = Depends(get_human_review_service),
    orchestrator=Depends(get_orchestrator),
    user: AuthenticatedUser = Depends(get_current_user),
) -> ApproveReviewResponse:
    _ensure_review_operator(service, review_id, user)
    try:
        task = service.approve(
            review_id,
            decided_by=user.username,
            note=payload.note,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    continuation_run = None
    continuation = None
    if payload.continuation is not None:
        state = service.build_continuation_state(
            task,
            continuation=payload.continuation,
        )
        review_payload = task.payload or {}
        result_state = await orchestrator.run(state)
        # Persist the approve -> continuation linkage so later queries of
        # this review know which request resumed the workflow.
        task = service.record_continuation(task, result_state)
        continuation = ReviewContinuationResponse(
            request_id=state.request_id,
            source_review_id=task.id,
            source_request_id=task.request_id,
            source_workflow_id=review_payload.get("workflow_id"),
            source_trace_id=review_payload.get("trace_id"),
            source_workflow_status=(
                WorkflowStatus(review_payload["workflow_status"])
                if review_payload.get("workflow_status")
                else None
            ),
            source_terminal_reason=(
                WorkflowTerminalReason(review_payload["terminal_reason"])
                if review_payload.get("terminal_reason")
                else None
            ),
            resumed_agent_id=payload.continuation.agent_id,
            decided_by=task.decided_by,
            decision_note=task.decision_note,
        )
        continuation_run = WorkflowRunResponse(
            request_id=result_state.request_id,
            status=result_state.status,
            terminal_reason=result_state.terminal_reason,
            review_id=result_state.memory.get("human_review", {}).get("review_id"),
            source_review_id=task.id,
            execution_path=result_state.execution_path,
            errors=result_state.errors,
            final_result=result_state.final_result,
        )

    return ApproveReviewResponse(
        review=_to_review_response(task),
        continuation=continuation,
        continuation_run=continuation_run,
    )


@router.post(
    "/runtime/reviews/{review_id}/reject",
    response_model=HumanReviewResponse,
)
def reject_review(
    review_id: str,
    payload: ReviewDecisionRequest,
    service: HumanReviewService = Depends(get_human_review_service),
    user: AuthenticatedUser = Depends(get_current_user),
) -> HumanReviewResponse:
    _ensure_review_operator(service, review_id, user)
    try:
        task = service.reject(
            review_id,
            decided_by=user.username,
            note=payload.note,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return _to_review_response(task)
