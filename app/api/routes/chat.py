"""Chat console API.

Two shapes live here:

* ordinary CRUD over the caller's chat sessions and their transcript, and
* ``POST /chat/turns`` -- a Server-Sent Events stream for one turn.

The turn endpoint streams regardless of what the turn turns out to be: a
plain conversation arrives as ``delta`` frames (typewriter effect) while a
hand-off to a specialist agent arrives as a single ``task`` frame the client
then polls. Deciding between the two is ``ChatService``'s job, so this module
never enumerates verticals.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import (
    AuthenticatedUser,
    check_session_ownership,
    get_chat_service,
    get_current_user,
    get_memory_service,
    get_model_pool_service,
)
from app.core import AppError
from app.schemas import (
    ChatSessionCreateRequest,
    ChatSessionDeleteResponse,
    ChatTurnRequest,
    PaginationMetaResponse,
    SessionListResponse,
    SessionMessageListResponse,
    SessionMessageResponse,
    SessionResponse,
)
from app.services import (
    ChatService,
    MemoryService,
    ModelPoolService,
    event_to_sse_frame,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Streaming proxies (nginx) buffer by default and would defeat the
    # typewriter effect entirely.
    "X-Accel-Buffering": "no",
}


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


@router.post(
    "/chat/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_chat_session(
    payload: ChatSessionCreateRequest,
    service: ChatService = Depends(get_chat_service),
    user: AuthenticatedUser = Depends(get_current_user),
) -> SessionResponse:
    record = service.create_session(user_id=user.id, title=payload.title)
    return _to_session_response(record)


@router.get("/chat/sessions", response_model=SessionListResponse)
def list_chat_sessions(
    user_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: ChatService = Depends(get_chat_service),
    user: AuthenticatedUser = Depends(get_current_user),
) -> SessionListResponse:
    # Members always see their own sessions only; administrators may group by
    # member, mirroring the task centre and the review list.
    effective_user_id = user_id if user.is_admin else user.id
    result = service.list_sessions(
        user_id=effective_user_id,
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
    "/chat/sessions/{session_id}/messages",
    response_model=SessionMessageListResponse,
)
def list_chat_messages(
    session_id: str,
    http_request: Request,
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: MemoryService = Depends(get_memory_service),
    user: AuthenticatedUser = Depends(get_current_user),
) -> SessionMessageListResponse:
    check_session_ownership(http_request, user, session_id)
    items = service.list_session_messages(session_id, limit=limit + offset)
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


@router.delete(
    "/chat/sessions/{session_id}",
    response_model=ChatSessionDeleteResponse,
)
def delete_chat_session(
    session_id: str,
    http_request: Request,
    service: ChatService = Depends(get_chat_service),
    user: AuthenticatedUser = Depends(get_current_user),
) -> ChatSessionDeleteResponse:
    check_session_ownership(http_request, user, session_id)
    service.delete_session(session_id)
    return ChatSessionDeleteResponse(session_id=session_id, deleted=True)


@router.post("/chat/turns")
async def create_chat_turn(
    payload: ChatTurnRequest,
    service: ChatService = Depends(get_chat_service),
    pool_service: ModelPoolService = Depends(get_model_pool_service),
    user: AuthenticatedUser = Depends(get_current_user),
) -> StreamingResponse:
    """Answer one turn as an SSE stream.

    Session creation, ownership checks and model resolution all run before the
    stream opens, so a bad session or an unavailable model still reports a
    real 4xx status instead of a 200 with an error frame in it.
    """
    session = service.ensure_session(
        session_id=payload.session_id,
        user_id=user.id,
    )
    model_entry = pool_service.resolve_entry(
        entry_id=payload.model_entry_id,
        name=payload.model_name,
    )

    async def event_stream():
        try:
            async for event in service.handle_turn(
                session=session,
                content=payload.content,
                user_id=user.id,
                model_entry=model_entry,
            ):
                yield event_to_sse_frame(event)
        except Exception as exc:  # noqa: BLE001 - surfaced to the client below
            # Headers are long gone: an SSE stream cannot change its status
            # code mid-flight, so the failure travels as an error frame.
            logger.exception("chat turn failed")
            yield event_to_sse_frame(
                {
                    "type": "error",
                    "message": _client_message(exc),
                    "session_id": session.session_id,
                }
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


def _client_message(exc: Exception) -> str:
    """App errors carry a message written for callers; anything else is not."""
    if isinstance(exc, AppError):
        return exc.message
    return "服务端处理该轮对话时发生异常，请稍后重试。"
