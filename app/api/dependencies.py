from __future__ import annotations

from collections.abc import Generator

from fastapi import Request

from app.agents.registry import AgentRegistry
from app.core import ResourceNotReadyError, Settings
from app.infra import AppResources
from app.orchestrator import BaseOrchestrator
from app.repositories import (
    AsyncWorkflowSubmissionRepository,
    HumanReviewRepository,
    MemoryRepository,
    RuntimeAuditRepository,
    WorkflowRunRepository,
)
from app.services import (
    AsyncWorkflowSubmissionService,
    HumanReviewService,
    MemoryService,
    RuntimeQueryService,
)
from app.tools.registry import ToolRegistry


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
