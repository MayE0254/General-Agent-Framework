from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fastapi import Depends, HTTPException, Request, status

from app.agents.registry import AgentRegistry
from app.core import ResourceNotReadyError, Settings
from app.core.security import verify_session_token
from app.infra import AppResources, is_redis_reachable
from app.orchestrator import BaseOrchestrator
from app.repositories import (
    AsyncWorkflowSubmissionRepository,
    HumanReviewRepository,
    MemoryRepository,
    ModelPoolRepository,
    RuntimeAuditRepository,
    UserRepository,
    WorkflowRunRepository,
)
from app.schemas import WorkflowRunRequest
from app.services import (
    AsyncWorkflowSubmissionService,
    ChatService,
    DispatchedTask,
    HumanReviewService,
    MemoryService,
    ModelPoolService,
    RuntimeQueryService,
)
from app.tasks.celery_app import celery_app
from app.tools.registry import ToolRegistry


@dataclass(frozen=True)
class AuthenticatedUser:
    """Identity resolved from the signed session cookie."""

    id: str
    username: str
    role: str
    display_name: str | None

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def get_current_user(request: Request) -> AuthenticatedUser:
    settings = get_app_settings(request)
    token = request.cookies.get(settings.auth.session_cookie_name)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
        )
    user_id = verify_session_token(token, settings.auth.session_secret)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session is invalid or expired.",
        )
    session_factory = get_database_session_factory(request)
    session = session_factory()
    try:
        user = UserRepository(session).get(user_id)
        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is disabled or no longer exists.",
            )
        return AuthenticatedUser(
            id=user.id,
            username=user.username,
            role=user.role,
            display_name=user.display_name,
        )
    finally:
        session.close()


def require_admin(user: AuthenticatedUser) -> AuthenticatedUser:
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator permission required.",
        )
    return user


def admin_required(
    user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    """FastAPI dependency combining login and admin-role checks."""
    return require_admin(user)


def check_request_ownership(
    request: Request, user: AuthenticatedUser, request_id: str
) -> None:
    """Raise 404 when a non-admin user does not own the workflow request.

    Ownership is resolved from the async submission first, then from the
    persisted workflow run metadata (covers synchronous runs).
    """
    if user.is_admin:
        return
    session_factory = get_database_session_factory(request)
    session = session_factory()
    try:
        submission = AsyncWorkflowSubmissionRepository(session).get_by_request_id(
            request_id
        )
        if submission is not None and submission.user_id == user.id:
            return
        workflow_run = WorkflowRunRepository(session).get_by_request_id(request_id)
        owner = (
            (workflow_run.metadata_payload or {}).get("user_id")
            if workflow_run is not None
            else None
        )
        if owner == user.id:
            return
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resource was not found.",
        )
    finally:
        session.close()


def check_session_ownership(
    request: Request, user: AuthenticatedUser, session_id: str
) -> None:
    """Raise 404 when a non-admin user does not own the session."""
    if user.is_admin:
        return
    session_factory = get_database_session_factory(request)
    session = session_factory()
    try:
        record = RuntimeAuditRepository(session).get_session_by_session_id(session_id)
        if record is None or record.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resource was not found.",
            )
    finally:
        session.close()


def get_memory_service(request: Request) -> MemoryService:
    service = getattr(request.app.state, "memory_service", None)
    if service is None:
        raise ResourceNotReadyError(
            "Memory service is not available.",
            details={"resource": "memory_service"},
        )
    return service


def get_app_settings(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise ResourceNotReadyError(
            "Application settings are not available.",
            details={"resource": "settings"},
        )
    return settings


def get_app_resources(request: Request) -> AppResources:
    resources = getattr(request.app.state, "resources", None)
    if resources is None:
        raise ResourceNotReadyError(
            "Application resources are not available.",
            details={"resource": "resources"},
        )
    return resources


def get_agent_registry(request: Request) -> AgentRegistry:
    registry = getattr(request.app.state, "agent_registry", None)
    if registry is None:
        raise ResourceNotReadyError(
            "Agent registry is not available.",
            details={"resource": "agent_registry"},
        )
    return registry


def get_tool_registry(request: Request) -> ToolRegistry:
    registry = getattr(request.app.state, "tool_registry", None)
    if registry is None:
        raise ResourceNotReadyError(
            "Tool registry is not available.",
            details={"resource": "tool_registry"},
        )
    return registry


def get_orchestrator(request: Request) -> BaseOrchestrator:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise ResourceNotReadyError(
            "Orchestrator is not available.",
            details={"resource": "orchestrator"},
        )
    return orchestrator


def get_database_session_factory(request: Request):
    resources = get_app_resources(request)
    if resources.database_session_factory is None:
        raise ResourceNotReadyError(
            "Database session factory is not available.",
            details={"resource": "database_session_factory"},
        )
    return resources.database_session_factory


def get_runtime_query_service(
    request: Request,
) -> Generator[RuntimeQueryService, None, None]:
    session_factory = get_database_session_factory(request)
    session = session_factory()
    service = RuntimeQueryService(
        RuntimeAuditRepository(session),
        workflow_run_repository=WorkflowRunRepository(session),
    )
    try:
        yield service
    finally:
        service.close()


def get_human_review_service(request: Request) -> HumanReviewService:
    service = getattr(request.app.state, "human_review_service", None)
    if service is None:
        raise ResourceNotReadyError(
            "Human review service is not available.",
            details={"resource": "human_review_service"},
        )
    return service


def get_async_workflow_submission_service(
    request: Request,
) -> Generator[AsyncWorkflowSubmissionService, None, None]:
    session_factory = get_database_session_factory(request)
    session = session_factory()
    service = AsyncWorkflowSubmissionService(
        AsyncWorkflowSubmissionRepository(session)
    )
    try:
        yield service
    finally:
        service.close()


def get_model_pool_service(
    request: Request,
) -> Generator[ModelPoolService, None, None]:
    session_factory = get_database_session_factory(request)
    session = session_factory()
    service = ModelPoolService(ModelPoolRepository(session))
    try:
        yield service
    finally:
        service.close()


def build_chat_task_dispatcher(
    request: Request,
) -> Callable[[str, dict[str, Any]], DispatchedTask]:
    """Build the transport chat uses to hand a turn to a specialist agent.

    Lives in the API layer on purpose: ``ChatService`` must stay broker-free
    and testable, so it only ever calls this callable. The submission row is
    written here as well as the Celery task, which is what makes a chat
    hand-off visible in the console task center under the same request id.
    """
    settings = get_app_settings(request)
    session_factory = get_database_session_factory(request)
    queue_name = settings.celery.task_default_queue

    def dispatch(task_name: str, payload: dict[str, Any]) -> DispatchedTask:
        if not is_redis_reachable(settings):
            raise ResourceNotReadyError(
                "Redis is not reachable. Start the Redis server before using "
                "async workflow submission.",
                details={"resource": "redis"},
            )
        workflow_request = WorkflowRunRequest.model_validate(payload)
        async_result = celery_app.send_task(
            task_name,
            args=[workflow_request.model_dump(mode="json")],
            task_id=str(uuid4()),
        )
        session = session_factory()
        try:
            service = AsyncWorkflowSubmissionService(
                AsyncWorkflowSubmissionRepository(session)
            )
            try:
                submission = service.create_submission(
                    workflow_request,
                    task_id=async_result.id,
                    task_name=task_name,
                    queue_name=queue_name,
                )
            finally:
                service.close()
        finally:
            session.close()
        return DispatchedTask(task_id=async_result.id, status=submission.status)

    return dispatch


def get_chat_service(request: Request) -> Generator[ChatService, None, None]:
    session_factory = get_database_session_factory(request)
    session = session_factory()
    service = ChatService(
        RuntimeAuditRepository(session),
        get_memory_service(request),
        llm_service=getattr(request.app.state, "llm_service", None),
        task_dispatcher=build_chat_task_dispatcher(request),
        knowledge_retriever=getattr(request.app.state, "knowledge_retriever", None),
        knowledge_settings=request.app.state.settings.knowledge,
    )
    try:
        yield service
    finally:
        service.close()
