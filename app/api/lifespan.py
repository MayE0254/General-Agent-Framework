from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.core import DependencyInitializationError, get_settings
from app.runtime_stack import build_runtime_stack
from app.services.worker_supervisor import start_worker_watchdog


@asynccontextmanager
async def application_lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    try:
        stack = build_runtime_stack(settings)
    except Exception as exc:  # pragma: no cover - defensive startup boundary
        raise DependencyInitializationError(
            "Application startup failed while preparing runtime resources.",
            details={"reason": str(exc)},
        ) from exc

    app.state.settings = stack.settings
    app.state.resources = stack.resources
    app.state.agent_registry = stack.agent_registry
    app.state.tool_registry = stack.tool_registry
    app.state.tool_executor = stack.tool_executor
    app.state.llm_service = stack.llm_service
    app.state.knowledge_retriever = stack.knowledge_retriever
    app.state.orchestrator = stack.orchestrator
    app.state.runtime_record_service = stack.runtime_record_service
    app.state.workflow_run_service = stack.workflow_run_service
    app.state.memory_service = stack.memory_service
    app.state.human_review_service = stack.human_review_service

    # Supervises the async layer: detects a queue that is stalled with no live
    # worker heartbeat and spawns a replacement (which is also what makes the
    # broker restore a message stuck unacked). Runs on the API loop so a
    # wedged worker cannot be the thing that reports on itself.
    app.state.worker_watchdog = start_worker_watchdog(stack.settings)

    try:
        yield
    finally:
        await app.state.worker_watchdog.stop()
        stack.close()
