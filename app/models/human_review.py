from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class HumanReviewTask(TimestampMixin, Base):
    """A human-in-the-loop decision point raised by a workflow run.

    One row per request that reached ``needs_human_review``. The payload
    carries a snapshot of the workflow state so approvers can inspect the
    run without re-querying live tables.
    """

    __tablename__ = "human_review_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str] = mapped_column(
        String(128),
        unique=True,
        nullable=False,
    )
    session_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        index=True,
        default="pending",
    )  # pending | approved | rejected | expired
    agent_id: Mapped[str] = mapped_column(String(64), nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )
    decided_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    decision_note: Mapped[str] = mapped_column(Text, default="", nullable=False)
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    # Chained-review linkage: set at creation when this task was raised by a
    # continuation run that itself ended in needs_human_review.
    source_review_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )
    # Continuation linkage: written back after approve + orchestrator rerun
    # so the review row knows which request continued the workflow.
    continuation_request_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )
    continuation_status: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )
    continuation_terminal_reason: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    continuation_run_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
