from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import WorkflowRunRecord


class WorkflowRunRepository:
    """Repository for workflow execution records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, record: WorkflowRunRecord) -> WorkflowRunRecord:
        self._session.add(record)
        self._session.flush()
        return record

    def get_by_request_id(self, request_id: str) -> WorkflowRunRecord | None:
        statement = select(WorkflowRunRecord).where(
            WorkflowRunRecord.request_id == request_id
        )
        return self._session.scalar(statement)

    def list_recent(self, *, limit: int = 20) -> list[WorkflowRunRecord]:
        statement = (
            select(WorkflowRunRecord)
            .order_by(WorkflowRunRecord.created_at.desc())
            .limit(limit)
        )
        return list(self._session.scalars(statement))

    def list_workflow_runs(
        self,
        *,
        request_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[WorkflowRunRecord], int]:
        conditions = []
        if request_id is not None:
            conditions.append(WorkflowRunRecord.request_id == request_id)
        if status is not None:
            conditions.append(WorkflowRunRecord.status == status)

        statement = select(WorkflowRunRecord)
        count_statement = select(func.count()).select_from(WorkflowRunRecord)
        if conditions:
            statement = statement.where(*conditions)
            count_statement = count_statement.where(*conditions)

        statement = (
            statement.order_by(WorkflowRunRecord.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        total = int(self._session.scalar(count_statement) or 0)
        return list(self._session.scalars(statement)), total

    def commit(self) -> None:
        self._session.commit()
