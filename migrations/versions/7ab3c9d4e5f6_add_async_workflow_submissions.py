"""add async workflow submissions

Revision ID: 7ab3c9d4e5f6
Revises: d821ffd3116d
Create Date: 2026-08-28 03:45:00.000000+00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "7ab3c9d4e5f6"
down_revision = "d821ffd3116d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "async_workflow_submissions",
        sa.Column("task_id", sa.String(length=36), nullable=False),
        sa.Column("request_id", sa.String(length=128), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=True),
        sa.Column("user_id", sa.String(length=128), nullable=True),
        sa.Column("initial_agent_id", sa.String(length=64), nullable=False),
        sa.Column("task_name", sa.String(length=128), nullable=False),
        sa.Column("queue_name", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("result_payload", sa.JSON(), nullable=True),
        sa.Column("traceback", sa.Text(), nullable=True),
        sa.Column("metadata_payload", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("task_id"),
        sa.UniqueConstraint("request_id"),
    )
    op.create_index(
        op.f("ix_async_workflow_submissions_status"),
        "async_workflow_submissions",
        ["status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_async_workflow_submissions_session_id"),
        "async_workflow_submissions",
        ["session_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_async_workflow_submissions_user_id"),
        "async_workflow_submissions",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_async_workflow_submissions_user_id"),
        table_name="async_workflow_submissions",
    )
    op.drop_index(
        op.f("ix_async_workflow_submissions_session_id"),
        table_name="async_workflow_submissions",
    )
    op.drop_index(
        op.f("ix_async_workflow_submissions_status"),
        table_name="async_workflow_submissions",
    )
    op.drop_table("async_workflow_submissions")
