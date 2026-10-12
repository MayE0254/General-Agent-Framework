"""Auto-approve overdue human review tasks.

Pending reviews that nobody answered within ``AUTO_APPROVE_TIMEOUT_MINUTES``
are approved by the system and the workflow continues with the same defaults
the runtime console would use (resume at the review's own agent). This keeps
unattended runs (e.g. nightly scheduled articles) moving instead of blocking
forever on an unanswered approval dialog.

Scheduled by the framework beat entry in ``app.tasks.celery_app``.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta, timezone
from typing import Any

from app.tasks.celery_app import celery_app
from app.tasks.workflow_tasks import _get_runtime
from app.services.human_review_service import utc_now

# Reviews pending longer than this are auto-approved. Adjust to taste.
AUTO_APPROVE_TIMEOUT_MINUTES = 30
# Safety guard: never touch reviews older than this — those are stale
# artifacts from forgotten/aborted runs, and resuming them could re-run
# publishes for content nobody expects anymore.
AUTO_APPROVE_MAX_AGE_HOURS = 24
AUTO_APPROVE_DECIDED_BY = "system:auto_approve"


def _overdue_pending_reviews(stack: Any, cutoff: Any, stale_before: Any) -> list[Any]:
    """Pending review rows created between ``stale_before`` and ``cutoff``.

    Anything older than ``stale_before`` is a stale artifact from a forgotten
    run and is left untouched (resuming it could re-publish old content).
    """
    service = stack.human_review_service
    if service is None:
        return []
    items = (service.list_reviews(status="pending", limit=200) or {}).get(
        "items"
    ) or []
    overdue: list[Any] = []
    for item in items:
        created_at = item.created_at
        if created_at is None:
            continue
        if created_at.tzinfo is None:
            # Defensive: treat naive DB datetimes as UTC (see utc_now()).
            created_at = created_at.replace(tzinfo=timezone.utc)
        if stale_before < created_at <= cutoff:
            overdue.append(item)
    return overdue


@celery_app.task(
    name="review.auto_approve_overdue",
    max_retries=0,
    acks_late=True,
)
def auto_approve_overdue_reviews() -> dict[str, Any]:
    """Approve pending reviews older than the timeout and resume workflows."""
    from app.schemas import ReviewContinuationRequest

    stack = _get_runtime()
    service = stack.human_review_service
    if service is None:
        return {"approved": 0, "skipped": "database unavailable"}

    cutoff = utc_now() - timedelta(minutes=AUTO_APPROVE_TIMEOUT_MINUTES)
    stale_before = utc_now() - timedelta(hours=AUTO_APPROVE_MAX_AGE_HOURS)
    overdue = _overdue_pending_reviews(stack, cutoff, stale_before)
    results: list[dict[str, Any]] = []
    for item in overdue:
        entry: dict[str, Any] = {
            "review_id": item.id,
            "request_id": item.request_id,
        }
        try:
            task = service.approve(
                item.id,
                decided_by=AUTO_APPROVE_DECIDED_BY,
                note=(
                    f"超过 {AUTO_APPROVE_TIMEOUT_MINUTES} 分钟未响应，"
                    "系统自动批准。"
                ),
            )
            payload = task.payload or {}
            continuation = ReviewContinuationRequest(
                agent_id=task.agent_id,
                input_text=payload.get("input_text") or "(超时自动批准续跑)",
                structured_input=payload.get("structured_input") or {},
            )
            state = service.build_continuation_state(
                task,
                continuation=continuation,
            )
            result_state = asyncio.run(stack.orchestrator.run(state))
            task = service.record_continuation(task, result_state)
            entry.update(
                status="approved_and_resumed",
                continuation_request_id=result_state.request_id,
                continuation_status=result_state.status.value,
            )
        except Exception as exc:  # noqa: BLE001 - keep processing the rest
            entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
        results.append(entry)
    return {
        "timeout_minutes": AUTO_APPROVE_TIMEOUT_MINUTES,
        "approved": len(results),
        "items": results,
    }
