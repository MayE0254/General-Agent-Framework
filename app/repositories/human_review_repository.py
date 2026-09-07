from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.human_review import HumanReviewTask


class HumanReviewRepository:
    """Persistence for human-in-the-loop review tasks."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, task: HumanReviewTask) -> HumanReviewTask:
        self._session.add(task)
        self._session.flush()
        return task

    def get(self, review_id: str) -> HumanReviewTask | None:
        statement = select(HumanReviewTask).where(HumanReviewTask.id == review_id)
        return self._session.scalar(statement)

    def get_by_request_id(self, request_id: str) -> HumanReviewTask | None:
        statement = select(HumanReviewTask).where(
            HumanReviewTask.request_id == request_id
        )
        return self._session.scalar(statement)

    def list(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[HumanReviewTask], int]:
        filters = []
        if status:
            filters.append(HumanReviewTask.status == status)
        total = int(
            self._session.scalar(
                select(func.count())
                .select_from(HumanReviewTask)
                .where(*filters)
            )
            or 0
        )
        statement = (
            select(HumanReviewTask)
            .where(*filters)
            .order_by(HumanReviewTask.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self._session.scalars(statement)), total

    def update_decision(
        self,
        review_id: str,
        *,
        status: str,
        decided_by: str | None,
        decision_note: str,
        decided_at: object,
    ) -> HumanReviewTask | None:
        task = self.get(review_id)
        if task is None:
            return None
        task.status = status
        task.decided_by = decided_by
        task.decision_note = decision_note
        task.decided_at = decided_at
        self._session.flush()
        return task

    def update_continuation(
        self,
        review_id: str,
        *,
        continuation_request_id: str,
        continuation_status: str,
        continuation_terminal_reason: str | None,
        continuation_run_at: object,
    ) -> HumanReviewTask | None:
        """Link a review task to the request that continued its workflow."""
        task = self.get(review_id)
        if task is None:
            return None
        task.continuation_request_id = continuation_request_id
        task.continuation_status = continuation_status
        task.continuation_terminal_reason = continuation_terminal_reason
        task.continuation_run_at = continuation_run_at
        self._session.flush()
        return task

    def commit(self) -> None:
        self._session.commit()

    def close(self) -> None:
        self._session.close()
