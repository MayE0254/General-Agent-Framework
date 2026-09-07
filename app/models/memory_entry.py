from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class MemoryEntryRecord(TimestampMixin, Base):
    """Durable long-term memory: user preferences, project facts, etc."""

    __tablename__ = "memory_entries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    memory_type: Mapped[str] = mapped_column(
        String(32), nullable=False, index=True
    )  # preference | fact | rule
    scope: Mapped[str] = mapped_column(
        String(16), nullable=False, index=True
    )  # user | project | global
    scope_key: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True
    )  # user_id / project key
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(
        String(16), default="manual", nullable=False
    )  # user | llm | manual
    importance: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    metadata_payload: Mapped[dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )
