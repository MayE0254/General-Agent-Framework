from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import case, delete, func, select
from sqlalchemy.orm import Session

from app.models import (
    AgentRunLog,
    LLMCallLog,
    SessionMessageRecord,
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
            # Merge rather than replace: a session row is shared by every
            # writer (workflow runs and chat), so one writer must not drop
            # the keys another one owns (e.g. the chat title).
            existing.metadata_payload = {
                **(existing.metadata_payload or {}),
                **(record.metadata_payload or {}),
            }
            self._session.flush()
            return existing
        self._session.add(record)
        self._session.flush()
        return record

    def update_session_metadata(
        self, session_id: str, patch: dict[str, object]
    ) -> SessionRecord | None:
        """Merge ``patch`` into a session's metadata without touching the rest."""
        record = self.get_session_by_session_id(session_id)
        if record is None:
            return None
        record.metadata_payload = {**(record.metadata_payload or {}), **patch}
        self._session.flush()
        return record

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its transcript.

        Runtime audit rows (task runs, agent runs, LLM calls) are immutable
        audit records and are intentionally left in place; they keep the
        session id as a plain string with no foreign key.
        """
        self._session.execute(
            delete(SessionMessageRecord).where(
                SessionMessageRecord.session_id == session_id
            )
        )
        result = self._session.execute(
            delete(SessionRecord).where(SessionRecord.session_id == session_id)
        )
        self._session.flush()
        return result.rowcount > 0

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
        user_id: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[LLMCallLog], int]:
        conditions = self._llm_usage_conditions(
            user_id=user_id,
            model=model,
            created_from=created_from,
            created_to=created_to,
        )
        if request_id is not None:
            conditions.append(LLMCallLog.request_id == request_id)
        if agent_id is not None:
            conditions.append(LLMCallLog.agent_id == agent_id)
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

    @staticmethod
    def _llm_usage_conditions(
        *,
        user_id: str | None,
        model: str | None,
        created_from: datetime | None,
        created_to: datetime | None,
    ) -> list[Any]:
        conditions: list[Any] = []
        if user_id is not None:
            conditions.append(LLMCallLog.user_id == user_id)
        if model is not None:
            conditions.append(LLMCallLog.model == model)
        if created_from is not None:
            conditions.append(LLMCallLog.created_at >= created_from)
        if created_to is not None:
            conditions.append(LLMCallLog.created_at <= created_to)
        return conditions

    def aggregate_llm_usage(
        self,
        *,
        user_id: str | None = None,
        model: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        by_model_limit: int = 100,
        by_user_limit: int = 200,
        by_user_model_limit: int = 500,
    ) -> dict[str, Any]:
        """Aggregate token usage per model and per user.

        ``by_user_model`` is the cross of the other two: it answers "which
        models did this user spend tokens on", which neither flat grouping can
        express on its own.
        """
        conditions = self._llm_usage_conditions(
            user_id=user_id,
            model=model,
            created_from=created_from,
            created_to=created_to,
        )
        return {
            # Aggregates without GROUP BY always yield exactly one row, so
            # totals are never missing even when no call matched.
            "totals": self._aggregate_llm_usage_rows(conditions=conditions)[0],
            "by_model": self._aggregate_llm_usage_rows(
                conditions=conditions,
                group_columns=[LLMCallLog.model],
            )[:by_model_limit],
            "by_user": self._aggregate_llm_usage_rows(
                conditions=conditions,
                group_columns=[LLMCallLog.user_id],
            )[:by_user_limit],
            "by_user_model": self._aggregate_llm_usage_rows(
                conditions=conditions,
                group_columns=[LLMCallLog.user_id, LLMCallLog.model],
            )[:by_user_model_limit],
        }

    def _aggregate_llm_usage_rows(
        self,
        *,
        conditions: list[Any],
        group_columns: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        group_columns = group_columns or []
        columns: list[Any] = [
            func.coalesce(func.sum(LLMCallLog.prompt_tokens), 0).label(
                "prompt_tokens"
            ),
            func.coalesce(func.sum(LLMCallLog.completion_tokens), 0).label(
                "completion_tokens"
            ),
            func.coalesce(func.sum(LLMCallLog.total_tokens), 0).label(
                "total_tokens"
            ),
            func.coalesce(func.sum(LLMCallLog.cost_usd), 0.0).label("cost_usd"),
            func.count(LLMCallLog.id).label("call_count"),
            func.coalesce(
                func.sum(
                    case((LLMCallLog.status == "success", 0), else_=1)
                ),
                0,
            ).label("failed_call_count"),
        ]
        statement = select(*group_columns, *columns)
        if conditions:
            statement = statement.where(*conditions)
        if group_columns:
            statement = statement.group_by(*group_columns).order_by(
                func.sum(LLMCallLog.total_tokens).desc()
            )
        rows = self._session.execute(statement).all()

        results: list[dict[str, Any]] = []
        for row in rows:
            mapping = row._mapping
            item: dict[str, Any] = {}
            if len(group_columns) == 1:
                item["key"] = mapping[group_columns[0].key]
            elif len(group_columns) > 1:
                item["user_id"] = mapping[group_columns[0].key]
                item["model"] = mapping[group_columns[1].key]
            item.update(
                {
                    "prompt_tokens": int(mapping["prompt_tokens"] or 0),
                    "completion_tokens": int(mapping["completion_tokens"] or 0),
                    "total_tokens": int(mapping["total_tokens"] or 0),
                    "cost_usd": round(float(mapping["cost_usd"] or 0.0), 6),
                    "call_count": int(mapping["call_count"] or 0),
                    "failed_call_count": int(mapping["failed_call_count"] or 0),
                }
            )
            results.append(item)
        return results

    def commit(self) -> None:
        self._session.commit()

    def close(self) -> None:
        self._session.close()
