"""Reusable application runtime stack.

Both the FastAPI process (lifespan) and the Celery worker process need the
same wiring: resources, registries, services and the orchestrator. Keeping the
construction in one place prevents the two entry points from drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents.registry import AgentRegistry, create_default_registry
from app.agents.registry.routing import sync_domain_routes
from app.business import register_business_agents
from app.core import Settings, get_settings
from app.infra import AppResources, close_app_resources, initialize_app_resources
from app.llm import LLMService
from app.llm.factory import create_llm_service
from app.observability import create_tracer
from app.orchestrator import BaseOrchestrator, create_orchestrator
from app.repositories import (
    HumanReviewRepository,
    MemoryRepository,
    RuntimeAuditRepository,
    WorkflowRunRepository,
)
from app.services import (
    HumanReviewService,
    MemoryService,
    RuntimeRecordService,
    WorkflowRunService,
)
from app.tools import ToolExecutor, ToolRegistry, create_default_tool_registry


@dataclass
class RuntimeStack:
    """Everything a runtime process needs to execute workflows."""

    settings: Settings
    resources: AppResources
    agent_registry: AgentRegistry
    tool_registry: ToolRegistry
    tool_executor: ToolExecutor
    llm_service: LLMService
    orchestrator: BaseOrchestrator
    runtime_record_service: RuntimeRecordService | None
    workflow_run_service: WorkflowRunService | None
    memory_service: MemoryService | None
    human_review_service: HumanReviewService | None
    tracer: Any | None = None

    def close(self) -> None:
        for service in (
            self.runtime_record_service,
            self.memory_service,
            self.human_review_service,
        ):
            if service is not None:
                service.close()
        close_app_resources(self.resources)


def build_runtime_stack(settings: Settings | None = None) -> RuntimeStack:
    """Construct the full runtime stack (resources -> services -> orchestrator).

    Database-backed services are only created when the database probe
    succeeded; otherwise they stay ``None`` and the orchestrator keeps serving
    without persistence (preview mode).
    """
    current_settings = settings or get_settings()
    resources = initialize_app_resources(current_settings)
    agent_registry = create_default_registry()
    tool_registry = create_default_tool_registry(
        tool_settings=current_settings.tools,
        knowledge_settings=current_settings.knowledge,
    )
    # Business agents (vertical domains) + their routing registration.
    # Business packages under app/business are optional plugins: each one
    # mounts itself when present, a bare infrastructure checkout skips them.
    register_business_agents(
        agent_registry=agent_registry,
        tool_registry=tool_registry,
    )
    sync_domain_routes(agent_registry)

    runtime_record_service: RuntimeRecordService | None = None
    workflow_run_service: WorkflowRunService | None = None
    memory_service: MemoryService | None = None
    human_review_service: HumanReviewService | None = None
    if (
        resources.database.status.value == "ready"
        and resources.database_session_factory is not None
    ):
        session = resources.database_session_factory()
        runtime_record_service = RuntimeRecordService(RuntimeAuditRepository(session))
        workflow_run_service = WorkflowRunService(WorkflowRunRepository(session))
        session = resources.database_session_factory()
        human_review_service = HumanReviewService(HumanReviewRepository(session))

    tracer = create_tracer(current_settings.observability)
    tool_executor = ToolExecutor(
        tool_registry,
        runtime_record_service=runtime_record_service,
        tracer=tracer,
    )
    llm_audit_recorder = None
    if runtime_record_service is not None:
        llm_audit_recorder = runtime_record_service.save_llm_call_log
    llm_service = create_llm_service(
        current_settings,
        tracer=tracer,
        audit_recorder=llm_audit_recorder,
    )
    if (
        resources.database.status.value == "ready"
        and resources.database_session_factory is not None
    ):
        session = resources.database_session_factory()
        memory_service = MemoryService(
            MemoryRepository(session),
            llm_service=llm_service,
            settings=current_settings.memory,
        )

    orchestrator = create_orchestrator(
        settings=current_settings,
        registry=agent_registry,
        runtime_record_service=runtime_record_service,
        workflow_run_service=workflow_run_service,
        memory_service=memory_service,
        human_review_service=human_review_service,
        tool_executor=tool_executor,
        llm_service=llm_service,
        tracer=tracer,
    )

    return RuntimeStack(
        settings=current_settings,
        resources=resources,
        agent_registry=agent_registry,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        llm_service=llm_service,
        orchestrator=orchestrator,
        runtime_record_service=runtime_record_service,
        workflow_run_service=workflow_run_service,
        memory_service=memory_service,
        human_review_service=human_review_service,
        tracer=tracer,
    )
