"""Human-in-the-loop review service with an in-process SSE event bus."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from app.models.human_review import HumanReviewTask
from app.orchestrator.shared_state import set_review_context
from app.repositories.human_review_repository import HumanReviewRepository

if TYPE_CHECKING:
    from app.orchestrator.state import OrchestratorState
    from app.schemas import ReviewContinuationRequest

logger = logging.getLogger(__name__)


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ReviewEventBroker:
    """In-process fan-out of review lifecycle events to SSE subscribers.

    Queues are bounded; a slow consumer drops events rather than blocking
    publishers (polling the review list API remains the reliable path).
    """

    def __init__(self, *, max_queue_size: int = 100) -> None:
        self._queues: set[asyncio.Queue[dict[str, Any]]] = set()
        self._max_queue_size = max_queue_size

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=self._max_queue_size
        )
        self._queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._queues.discard(queue)

    async def publish(self, event: dict[str, Any]) -> None:
        for queue in list(self._queues):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # slow subscriber; drop and keep going
                logger.warning("Review event queue full, dropping event: %s", event)

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Yield events as an async iterator (drains into SSE frames)."""
        queue = self.subscribe()
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            self.unsubscribe(queue)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class HumanReviewService:
    """Creates, lists, and decides human-in-the-loop review tasks."""

    def __init__(self, repository: HumanReviewRepository) -> None:
        self._repository = repository
        self._broker = ReviewEventBroker()

    # ---- creation ----

    async def create_from_state(self, state: OrchestratorState) -> HumanReviewTask:
        final_result = state.final_result
        task = HumanReviewTask(
            id=str(uuid4()),
            request_id=state.request_id,
            session_id=state.session_id,
            user_id=state.user_id,
            status=ReviewStatus.PENDING.value,
            source_review_id=state.memory.get("human_review", {}).get(
                "source_review_id"
            ),
            agent_id=(
                final_result.agent_id if final_result is not None else state.current_agent_id or ""
            ),
            summary=final_result.summary if final_result is not None else "",
            payload={
                "workflow_id": state.workflow_id,
                "trace_id": state.trace_id,
                "project_key": state.project_key,
                "initial_agent_id": state.initial_agent_id,
                "current_agent_id": state.current_agent_id,
                "input_text": state.input_text,
                "structured_input": state.structured_input,
                "knowledge_refs": state.knowledge_refs,
                "workflow_status": state.status.value,
                "terminal_reason": (
                    state.terminal_reason.value
                    if state.terminal_reason is not None
                    else None
                ),
                "max_steps": state.max_steps,
                "execution_path": state.execution_path,
                "final_result": (
                    final_result.model_dump(mode="json")
                    if final_result is not None
                    else None
                ),
                "run_history": [
                    item.model_dump(mode="json") for item in state.run_history
                ],
                "errors": state.errors,
                "shared_state": state.shared_state,
                "memory": state.memory,
                "llm_profile": (
                    state.llm_profile.model_dump()
                    if state.llm_profile is not None
                    else None
                ),
            },
        )
        self._repository.add(task)
        self._repository.commit()
        await self._broker.publish(
            {
                "type": "review.created",
                "review_id": task.id,
                "request_id": task.request_id,
                "status": task.status,
                "created_at": task.created_at.isoformat()
                if task.created_at is not None
                else utc_now().isoformat(),
            }
        )
        return task

    # ---- queries ----

    def list_reviews(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        items, total = self._repository.list(
            status=status,
            limit=limit,
            offset=offset,
        )
        return {
            "items": items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "filters": {"status": status} if status else {},
        }

    def get_review(self, review_id: str) -> HumanReviewTask | None:
        return self._repository.get(review_id)

    def build_continuation_state(
        self,
        task: HumanReviewTask,
        *,
        continuation: ReviewContinuationRequest,
    ) -> OrchestratorState:
        from app.agents.base.schemas import LLMProfile
        from app.orchestrator import OrchestratorState

        snapshot = task.payload or {}
        source_structured_input = copy.deepcopy(
            snapshot.get("structured_input") or {}
        )
        source_structured_input.update(continuation.structured_input)

        source_shared_state = copy.deepcopy(snapshot.get("shared_state") or {})
        review_context = {
            "source_review_id": task.id,
            "source_request_id": task.request_id,
            "source_workflow_id": snapshot.get("workflow_id"),
            "source_trace_id": snapshot.get("trace_id"),
            "source_agent_id": task.agent_id,
            "source_workflow_status": snapshot.get("workflow_status"),
            "source_terminal_reason": snapshot.get("terminal_reason"),
            "approved_by": task.decided_by,
            "decision_note": task.decision_note,
            "resumed_agent_id": continuation.agent_id,
        }
        set_review_context(source_shared_state, review_context)

        source_memory = copy.deepcopy(snapshot.get("memory") or {})
        human_review_memory = source_memory.setdefault("human_review", {})
        previous_review_id = human_review_memory.pop("review_id", None)
        human_review_memory["source_review_id"] = task.id
        human_review_memory["source_request_id"] = task.request_id
        if previous_review_id is not None:
            human_review_memory["previous_review_id"] = previous_review_id

        profile_data = snapshot.get("llm_profile")
        return OrchestratorState(
            request_id=continuation.request_id or f"{task.id}-continuation",
            initial_agent_id=continuation.agent_id,
            session_id=task.session_id,
            user_id=task.user_id,
            workflow_id=snapshot.get("workflow_id"),
            trace_id=snapshot.get("trace_id"),
            project_key=snapshot.get("project_key") or "default",
            input_text=continuation.input_text,
            structured_input=source_structured_input,
            shared_state=source_shared_state,
            memory=source_memory,
            knowledge_refs=copy.deepcopy(snapshot.get("knowledge_refs") or []),
            max_steps=int(snapshot.get("max_steps") or 10),
            llm_profile=(
                LLMProfile.model_validate(profile_data)
                if profile_data is not None
                else None
            ),
        )

    # ---- decisions ----

    def approve(
        self,
        review_id: str,
        *,
        decided_by: str | None,
        note: str = "",
    ) -> HumanReviewTask:
        return self._decide(
            review_id,
            status=ReviewStatus.APPROVED,
            decided_by=decided_by,
            note=note,
        )

    def reject(
        self,
        review_id: str,
        *,
        decided_by: str | None,
        note: str = "",
    ) -> HumanReviewTask:
        return self._decide(
            review_id,
            status=ReviewStatus.REJECTED,
            decided_by=decided_by,
            note=note,
        )

    def record_continuation(
        self,
        task: HumanReviewTask,
        result_state: OrchestratorState,
    ) -> HumanReviewTask:
        """Persist the approve -> continuation run linkage on the review row."""
        updated = self._repository.update_continuation(
            task.id,
            continuation_request_id=result_state.request_id,
            continuation_status=result_state.status.value,
            continuation_terminal_reason=(
                result_state.terminal_reason.value
                if result_state.terminal_reason is not None
                else None
            ),
            continuation_run_at=utc_now(),
        )
        assert updated is not None
        self._repository.commit()
        return updated

    def _decide(
        self,
        review_id: str,
        *,
        status: ReviewStatus,
        decided_by: str | None,
        note: str,
    ) -> HumanReviewTask:
        task = self._repository.get(review_id)
        if task is None:
            raise KeyError(f"Review task '{review_id}' was not found.")
        if task.status != ReviewStatus.PENDING.value:
            raise ValueError(
                f"Review task '{review_id}' is already in status '{task.status}'."
            )
        task = self._repository.update_decision(
            review_id,
            status=status.value,
            decided_by=decided_by,
            decision_note=note,
            decided_at=utc_now(),
        )
        assert task is not None
        self._repository.commit()

        async def _notify() -> None:
            await self._broker.publish(
                {
                    "type": "review.decided",
                    "review_id": task.id,
                    "request_id": task.request_id,
                    "status": task.status,
                    "decided_at": (
                        task.decided_at.isoformat()
                        if task.decided_at is not None
                        else utc_now().isoformat()
                    ),
                }
            )

        # Fire-and-forget so the HTTP decision request returns immediately.
        # Sync endpoints run inside a threadpool where no event loop is
        # attached to the current thread, so fall back to a fresh loop.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_notify())
        else:
            asyncio.create_task(_notify())
        return task

    # ---- event bus passthroughs ----

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        return self._broker.subscribe()

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._broker.unsubscribe(queue)

    async def stream_events(self) -> AsyncIterator[dict[str, Any]]:
        async for event in self._broker.stream():
            yield event

    # ---- lifecycle ----

    def close(self) -> None:
        self._repository.close()


def event_to_sse_frame(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
