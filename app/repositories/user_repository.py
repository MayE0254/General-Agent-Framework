from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.user import UserRecord


class UserRepository:
    """Persistence for console accounts."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, user_id: str) -> UserRecord | None:
        statement = select(UserRecord).where(UserRecord.id == user_id)
        return self._session.scalar(statement)

    def get_by_username(self, username: str) -> UserRecord | None:
        statement = select(UserRecord).where(UserRecord.username == username)
        return self._session.scalar(statement)

    def list(self) -> list[UserRecord]:
        statement = select(UserRecord).order_by(UserRecord.created_at.asc())
        return list(self._session.scalars(statement))

    def count(self) -> int:
        return int(
            self._session.scalar(select(func.count()).select_from(UserRecord)) or 0
        )

    def add(self, user: UserRecord) -> UserRecord:
        self._session.add(user)
        self._session.flush()
        return user

    def delete(self, user: UserRecord) -> None:
        self._session.delete(user)
        self._session.flush()

    def commit(self) -> None:
        self._session.commit()

    def close(self) -> None:
        self._session.close()
