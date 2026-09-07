from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class WorkflowRunRecord(TimestampMixin, Base):
    """Example persistent record for a workflow execution."""

    __tablename__ = "workflow_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    workflow_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    initial_agent_id: Mapped[str] = mapped_column(String(64), nullable=False)
    current_agent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    execution_path: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    final_agent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    errors: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    metadata_payload: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
