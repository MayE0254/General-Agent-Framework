from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.models import ModelPoolEntry


class ModelPoolRepository:
    """Repository for the console-curated model pool."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_entry(self, record: ModelPoolEntry) -> ModelPoolEntry:
        self._session.add(record)
        self._session.flush()
        return record

    def get_entry(self, entry_id: str) -> ModelPoolEntry | None:
        return self._session.scalar(
            select(ModelPoolEntry).where(ModelPoolEntry.id == entry_id)
        )

    def get_entry_by_name(self, name: str) -> ModelPoolEntry | None:
        return self._session.scalar(
            select(ModelPoolEntry).where(ModelPoolEntry.name == name)
        )

    def get_entry_by_model(self, model: str) -> ModelPoolEntry | None:
        """Oldest entry serving ``model``.

        Used to price a call whose resolved model is only known as a string.
        """
        statement = (
            select(ModelPoolEntry)
            .where(ModelPoolEntry.model == model)
            .order_by(
                ModelPoolEntry.sort_order.asc(),
                ModelPoolEntry.created_at.asc(),
            )
            .limit(1)
        )
        return self._session.scalar(statement)

    def get_default_entry(self) -> ModelPoolEntry | None:
        statement = (
            select(ModelPoolEntry)
            .where(
                ModelPoolEntry.enabled.is_(True),
                ModelPoolEntry.is_default.is_(True),
            )
            .order_by(ModelPoolEntry.sort_order.asc())
            .limit(1)
        )
        return self._session.scalar(statement)

    def list_entries(
        self,
        *,
        enabled_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[ModelPoolEntry], int]:
        conditions = []
        if enabled_only:
            conditions.append(ModelPoolEntry.enabled.is_(True))

        statement = select(ModelPoolEntry)
        count_statement = select(func.count()).select_from(ModelPoolEntry)
        if conditions:
            statement = statement.where(*conditions)
            count_statement = count_statement.where(*conditions)

        statement = (
            statement.order_by(
                ModelPoolEntry.sort_order.asc(),
                ModelPoolEntry.name.asc(),
            )
            .offset(offset)
            .limit(limit)
        )
        total = int(self._session.scalar(count_statement) or 0)
        return list(self._session.scalars(statement)), total

    def clear_default(self, *, except_id: str | None = None) -> None:
        """Demote other defaults so at most one entry stays default."""
        statement = select(ModelPoolEntry).where(ModelPoolEntry.is_default.is_(True))
        for record in self._session.scalars(statement):
            if record.id != except_id:
                record.is_default = False
        self._session.flush()

    def delete_entry(self, entry_id: str) -> bool:
        result = self._session.execute(
            delete(ModelPoolEntry).where(ModelPoolEntry.id == entry_id)
        )
        self._session.flush()
        return result.rowcount > 0

    def commit(self) -> None:
        self._session.commit()

    def close(self) -> None:
        self._session.close()
