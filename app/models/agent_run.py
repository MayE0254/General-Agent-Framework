from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class AgentRunLog(TimestampMixin, Base):
    """Persistent audit log for every agent execution step."""

    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    workflow_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    next_agent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    started_at_runtime: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at_runtime: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    output: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    errors: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
