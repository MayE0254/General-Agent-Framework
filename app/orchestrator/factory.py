from __future__ import annotations

from typing import Any

from app.agents.registry import AgentRegistry
from app.core import Settings
from app.llm import LLMService
from app.orchestrator.base import BaseOrchestrator
from app.orchestrator.langgraph_runner import LangGraphOrchestrator
from app.orchestrator.runner import SimpleOrchestrator
from app.services.memory_service import MemoryService
from app.services.runtime_record_service import RuntimeRecordService
from app.services.workflow_run_service import WorkflowRunService
from app.tools import ToolExecutor


def create_orchestrator(
    *,
    settings: Settings,
    registry: AgentRegistry,
    runtime_record_service: RuntimeRecordService | None = None,
    workflow_run_service: WorkflowRunService | None = None,
    memory_service: MemoryService | None = None,
    human_review_service: Any | None = None,
    tool_executor: ToolExecutor | None = None,
    llm_service: LLMService | None = None,
    tracer: Any | None = None,
) -> BaseOrchestrator:
    backend = settings.orchestrator.backend
    if backend == "simple":
        return SimpleOrchestrator(
            registry,
            runtime_record_service=runtime_record_service,
            workflow_run_service=workflow_run_service,
            memory_service=memory_service,
            human_review_service=human_review_service,
            tool_executor=tool_executor,
            llm_service=llm_service,
            tracer=tracer,
        )
    if backend == "langgraph":
        try:
            import langgraph.graph  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "LangGraph backend was requested but langgraph is not installed."
            ) from exc
        return LangGraphOrchestrator(
            registry,
            runtime_record_service=runtime_record_service,
            workflow_run_service=workflow_run_service,
            memory_service=memory_service,
            human_review_service=human_review_service,
            tool_executor=tool_executor,
            llm_service=llm_service,
            tracer=tracer,
            max_graph_steps=settings.orchestrator.max_graph_steps,
        )
    raise ValueError(f"Unsupported orchestrator backend: {backend}")
