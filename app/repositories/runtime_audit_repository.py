from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import (
    AgentRunLog,
    LLMCallLog,
    SessionRecord,
    TaskRunRecord,
    ToolCallLog,
)


class RuntimeAuditRepository:
    """Repository for platform-level runtime audit records."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_session(self, record: SessionRecord) -> SessionRecord:
        # Same session id can span multiple requests: update the existing
        # row instead of colliding on the unique session_id constraint.
        existing = self._session.scalar(
            select(SessionRecord).where(
                SessionRecord.session_id == record.session_id
            )
        )
        if existing is not None:
            existing.status = record.status
            existing.last_request_id = record.last_request_id
            existing.metadata_payload = record.metadata_payload
            self._session.flush()
            return existing
        self._session.add(record)
        self._session.flush()
        return record

    def add_task_run(self, record: TaskRunRecord) -> TaskRunRecord:
        self._session.add(record)
        self._session.flush()
        return record

    def add_agent_run_logs(self, records: list[AgentRunLog]) -> list[AgentRunLog]:
        self._session.add_all(records)
        self._session.flush()
        return records

    def add_tool_call_logs(self, records: list[ToolCallLog]) -> list[ToolCallLog]:
        self._session.add_all(records)
        self._session.flush()
        return records

    def get_task_run_by_request_id(self, request_id: str) -> TaskRunRecord | None:
        statement = select(TaskRunRecord).where(TaskRunRecord.request_id == request_id)
        return self._session.scalar(statement)

    def get_session_by_session_id(self, session_id: str) -> SessionRecord | None:
        statement = select(SessionRecord).where(
            SessionRecord.session_id == session_id
        )
        return self._session.scalar(statement)

    def list_sessions(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[SessionRecord], int]:
        conditions = []
        if session_id is not None:
            conditions.append(SessionRecord.session_id == session_id)
        if user_id is not None:
            conditions.append(SessionRecord.user_id == user_id)
        if status is not None:
            conditions.append(SessionRecord.status == status)

        statement = select(SessionRecord)
        count_statement = select(func.count()).select_from(SessionRecord)
        if conditions:
            statement = statement.where(*conditions)
            count_statement = count_statement.where(*conditions)

        statement = statement.order_by(
            SessionRecord.created_at.desc()
        ).offset(offset).limit(limit)
        total = int(self._session.scalar(count_statement) or 0)
        return list(self._session.scalars(statement)), total

    def list_task_runs_by_session_id(
        self, session_id: str, *, limit: int = 50
    ) -> list[TaskRunRecord]:
        statement = (
            select(TaskRunRecord)
            .where(TaskRunRecord.session_id == session_id)
            .order_by(TaskRunRecord.created_at.desc())
            .limit(limit)
        )
        return list(self._session.scalars(statement))

    def list_task_runs(
        self,
        *,
        status: str | None = None,
        session_id: str | None = None,
        workflow_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[TaskRunRecord], int]:
        conditions = []
        if status is not None:
            conditions.append(TaskRunRecord.status == status)
        if session_id is not None:
            conditions.append(TaskRunRecord.session_id == session_id)
        if workflow_id is not None:
            conditions.append(TaskRunRecord.workflow_id == workflow_id)

        statement = select(TaskRunRecord)
        count_statement = select(func.count()).select_from(TaskRunRecord)
        if conditions:
            statement = statement.where(*conditions)
            count_statement = count_statement.where(*conditions)

        statement = (
            statement.order_by(TaskRunRecord.created_at.desc()).offset(offset).limit(limit)
        )
        total = int(self._session.scalar(count_statement) or 0)
        return list(self._session.scalars(statement)), total

    def list_agent_runs_by_request_id(self, request_id: str) -> list[AgentRunLog]:
        statement = (
            select(AgentRunLog)
            .where(AgentRunLog.request_id == request_id)
            .order_by(AgentRunLog.step_index.asc())
        )
        return list(self._session.scalars(statement))

    def list_agent_runs(
        self,
        *,
        request_id: str | None = None,
        agent_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[AgentRunLog], int]:
        conditions = []
        if request_id is not None:
            conditions.append(AgentRunLog.request_id == request_id)
        if agent_id is not None:
            conditions.append(AgentRunLog.agent_id == agent_id)
        if status is not None:
            conditions.append(AgentRunLog.status == status)

        statement = select(AgentRunLog)
        count_statement = select(func.count()).select_from(AgentRunLog)
        if conditions:
            statement = statement.where(*conditions)
            count_statement = count_statement.where(*conditions)

        statement = statement.order_by(
            AgentRunLog.created_at.desc(),
            AgentRunLog.step_index.asc(),
        ).offset(offset).limit(limit)
        total = int(self._session.scalar(count_statement) or 0)
        return list(self._session.scalars(statement)), total

    def list_tool_calls_by_request_id(self, request_id: str) -> list[ToolCallLog]:
        statement = select(ToolCallLog).where(ToolCallLog.request_id == request_id)
        return list(self._session.scalars(statement))

    def list_tool_calls(
        self,
        *,
        request_id: str | None = None,
        agent_id: str | None = None,
        tool_name: str | None = None,
        status: str | None = None,
        error_category: str | None = None,
        retryable: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[ToolCallLog], int]:
        conditions = []
        if request_id is not None:
            conditions.append(ToolCallLog.request_id == request_id)
        if agent_id is not None:
            conditions.append(ToolCallLog.agent_id == agent_id)
        if tool_name is not None:
            conditions.append(ToolCallLog.tool_name == tool_name)
        if status is not None:
            conditions.append(ToolCallLog.status == status)

        statement = select(ToolCallLog)
        if conditions:
            statement = statement.where(*conditions)
        statement = statement.order_by(ToolCallLog.created_at.desc())
        items = list(self._session.scalars(statement))

        if error_category is not None:
            items = [
                item
                for item in items
                if item.metrics.get("error_category") == error_category
            ]
        if retryable is not None:
            items = [
                item
                for item in items
                if bool(item.metrics.get("retryable", False)) == retryable
            ]

        total = len(items)
        return items[offset : offset + limit], total

    def add_llm_call_logs(self, records: list[LLMCallLog]) -> list[LLMCallLog]:
        self._session.add_all(records)
        self._session.flush()
        return records

    def list_llm_call_logs_by_request_id(self, request_id: str) -> list[LLMCallLog]:
        statement = select(LLMCallLog).where(LLMCallLog.request_id == request_id)
        return list(self._session.scalars(statement))

    def list_llm_call_logs(
        self,
        *,
        request_id: str | None = None,
        agent_id: str | None = None,
        model: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[LLMCallLog], int]:
        conditions = []
        if request_id is not None:
            conditions.append(LLMCallLog.request_id == request_id)
        if agent_id is not None:
            conditions.append(LLMCallLog.agent_id == agent_id)
        if model is not None:
            conditions.append(LLMCallLog.model == model)
        if status is not None:
            conditions.append(LLMCallLog.status == status)

        statement = select(LLMCallLog)
        count_statement = select(func.count()).select_from(LLMCallLog)
        if conditions:
            statement = statement.where(*conditions)
            count_statement = count_statement.where(*conditions)

        statement = statement.order_by(
            LLMCallLog.created_at.desc()
        ).offset(offset).limit(limit)
        total = int(self._session.scalar(count_statement) or 0)
        return list(self._session.scalars(statement)), total

    def commit(self) -> None:
        self._session.commit()

    def close(self) -> None:
        self._session.close()
