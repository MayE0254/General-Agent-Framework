from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import AsyncWorkflowSubmissionRecord


class AsyncWorkflowSubmissionRepository:
    """Repository for persisted async workflow queue submissions."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(
        self, record: AsyncWorkflowSubmissionRecord
    ) -> AsyncWorkflowSubmissionRecord:
        self._session.add(record)
        self._session.flush()
        return record

    def get_by_task_id(
        self, task_id: str
    ) -> AsyncWorkflowSubmissionRecord | None:
        statement = select(AsyncWorkflowSubmissionRecord).where(
            AsyncWorkflowSubmissionRecord.task_id == task_id
        )
        return self._session.scalar(statement)

    def get_by_request_id(
        self, request_id: str
    ) -> AsyncWorkflowSubmissionRecord | None:
        statement = select(AsyncWorkflowSubmissionRecord).where(
            AsyncWorkflowSubmissionRecord.request_id == request_id
        )
        return self._session.scalar(statement)

    def list_submissions(
        self,
        *,
        status: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[AsyncWorkflowSubmissionRecord], int]:
        conditions = []
        if status is not None:
            conditions.append(AsyncWorkflowSubmissionRecord.status == status)
        if request_id is not None:
            conditions.append(AsyncWorkflowSubmissionRecord.request_id == request_id)
        if session_id is not None:
            conditions.append(AsyncWorkflowSubmissionRecord.session_id == session_id)
        if user_id is not None:
            conditions.append(AsyncWorkflowSubmissionRecord.user_id == user_id)

        statement = select(AsyncWorkflowSubmissionRecord)
        count_statement = select(func.count()).select_from(
            AsyncWorkflowSubmissionRecord
        )
        if conditions:
            statement = statement.where(*conditions)
            count_statement = count_statement.where(*conditions)

        statement = (
            statement.order_by(AsyncWorkflowSubmissionRecord.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        total = int(self._session.scalar(count_statement) or 0)
        return list(self._session.scalars(statement)), total

    def commit(self) -> None:
        self._session.commit()

    def close(self) -> None:
        self._session.close()
