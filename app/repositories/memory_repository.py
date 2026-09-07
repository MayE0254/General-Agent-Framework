from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.models import MemoryEntryRecord, SessionMessageRecord


class MemoryRepository:
    """Repository for session messages and durable memory entries."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ---- session messages ----

    def add_session_messages(
        self, records: list[SessionMessageRecord]
    ) -> list[SessionMessageRecord]:
        self._session.add_all(records)
        self._session.flush()
        return records

    def list_session_messages(
        self, session_id: str, *, limit: int = 50
    ) -> list[SessionMessageRecord]:
        # Return the most recent `limit` messages, ordered oldest-first so
        # consumers can append the current turn naturally. Request_id is a
        # tiebreaker so messages written in the same timestamp tick group by
        # request (newest request first, step ascending after the reverse).
        statement = (
            select(SessionMessageRecord)
            .where(SessionMessageRecord.session_id == session_id)
            .order_by(
                SessionMessageRecord.created_at.desc(),
                SessionMessageRecord.request_id.desc(),
                SessionMessageRecord.step_index.desc(),
            )
            .limit(limit)
        )
        rows = list(self._session.scalars(statement))
        rows.reverse()
        return rows

    def count_session_messages(self, session_id: str) -> int:
        statement = (
            select(func.count())
            .select_from(SessionMessageRecord)
            .where(SessionMessageRecord.session_id == session_id)
        )
        return int(self._session.scalar(statement) or 0)

    # ---- memory entries ----

    def add_memory(self, record: MemoryEntryRecord) -> MemoryEntryRecord:
        self._session.add(record)
        self._session.flush()
        return record

    def get_memory(self, memory_id: str) -> MemoryEntryRecord | None:
        statement = select(MemoryEntryRecord).where(
            MemoryEntryRecord.id == memory_id
        )
        return self._session.scalar(statement)

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        importance: int | None = None,
        tags: list[str] | None = None,
        metadata_payload: dict[str, object] | None = None,
    ) -> MemoryEntryRecord | None:
        record = self.get_memory(memory_id)
        if record is None:
            return None
        if content is not None:
            record.content = content
        if importance is not None:
            record.importance = importance
        if tags is not None:
            record.tags = tags
        if metadata_payload is not None:
            record.metadata_payload = metadata_payload
        self._session.flush()
        return record

    def delete_memory(self, memory_id: str) -> bool:
        statement = delete(MemoryEntryRecord).where(
            MemoryEntryRecord.id == memory_id
        )
        result = self._session.execute(statement)
        self._session.flush()
        return result.rowcount > 0

    def list_memories(
        self,
        *,
        scope: str | None = None,
        scope_key: str | None = None,
        memory_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[MemoryEntryRecord], int]:
        conditions = []
        if scope is not None:
            conditions.append(MemoryEntryRecord.scope == scope)
        if scope_key is not None:
            conditions.append(MemoryEntryRecord.scope_key == scope_key)
        if memory_type is not None:
            conditions.append(MemoryEntryRecord.memory_type == memory_type)

        statement = select(MemoryEntryRecord)
        count_statement = select(func.count()).select_from(MemoryEntryRecord)
        if conditions:
            statement = statement.where(*conditions)
            count_statement = count_statement.where(*conditions)

        statement = statement.order_by(
            MemoryEntryRecord.importance.desc(),
            MemoryEntryRecord.created_at.desc(),
        ).offset(offset).limit(limit)
        total = int(self._session.scalar(count_statement) or 0)
        return list(self._session.scalars(statement)), total

    def recall_memories(
        self,
        *,
        scope: str,
        scope_key: str,
        memory_type: str | None = None,
        limit: int = 10,
    ) -> list[MemoryEntryRecord]:
        conditions = [
            MemoryEntryRecord.scope == scope,
            MemoryEntryRecord.scope_key == scope_key,
        ]
        if memory_type is not None:
            conditions.append(MemoryEntryRecord.memory_type == memory_type)
        statement = (
            select(MemoryEntryRecord)
            .where(*conditions)
            .order_by(
                MemoryEntryRecord.importance.desc(),
                MemoryEntryRecord.created_at.desc(),
            )
            .limit(limit)
        )
        return list(self._session.scalars(statement))

    def commit(self) -> None:
        self._session.commit()

    def close(self) -> None:
        self._session.close()
