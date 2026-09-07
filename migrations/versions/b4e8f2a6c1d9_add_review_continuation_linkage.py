"""add review continuation linkage

Revision ID: b4e8f2a6c1d9
Revises: 7ab3c9d4e5f6
Create Date: 2026-09-02 16:20:00.000000+00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "b4e8f2a6c1d9"
down_revision = "7ab3c9d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "human_review_tasks",
        sa.Column("source_review_id", sa.String(length=36), nullable=True),
    )
    op.add_column(
        "human_review_tasks",
        sa.Column("continuation_request_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "human_review_tasks",
        sa.Column("continuation_status", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "human_review_tasks",
        sa.Column("continuation_terminal_reason", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "human_review_tasks",
        sa.Column("continuation_run_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        op.f("ix_human_review_tasks_source_review_id"),
        "human_review_tasks",
        ["source_review_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_human_review_tasks_continuation_request_id"),
        "human_review_tasks",
        ["continuation_request_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_human_review_tasks_continuation_request_id"),
        table_name="human_review_tasks",
    )
    op.drop_index(
        op.f("ix_human_review_tasks_source_review_id"),
        table_name="human_review_tasks",
    )
    op.drop_column("human_review_tasks", "continuation_run_at")
    op.drop_column("human_review_tasks", "continuation_terminal_reason")
    op.drop_column("human_review_tasks", "continuation_status")
    op.drop_column("human_review_tasks", "continuation_request_id")
    op.drop_column("human_review_tasks", "source_review_id")
