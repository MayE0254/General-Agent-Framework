"""add model pool table and llm call user attribution

Revision ID: d4f1a8c2e6b9
Revises: c7d9e1f3a5b7
Create Date: 2026-09-16 18:40:00.000000+00:00
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "d4f1a8c2e6b9"
down_revision = "c7d9e1f3a5b7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_pool_entries",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("provider_key", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("description", sa.String(length=512), nullable=False, server_default=""),
        sa.Column("temperature", sa.Float(), nullable=False, server_default="0.2"),
        sa.Column("max_tokens", sa.Integer(), nullable=True),
        sa.Column("input_per_million", sa.Float(), nullable=False, server_default="0"),
        sa.Column("output_per_million", sa.Float(), nullable=False, server_default="0"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("sort_order", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_model_pool_entries_name"),
        "model_pool_entries",
        ["name"],
        unique=True,
    )

    # Per-call user attribution: the usage report groups llm_call_logs by
    # user, which previously could only be inferred by joining request_id
    # back to workflow_runs.metadata_payload.
    op.add_column(
        "llm_call_logs",
        sa.Column("user_id", sa.String(length=128), nullable=True),
    )
    op.create_index(
        op.f("ix_llm_call_logs_user_id"),
        "llm_call_logs",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_llm_call_logs_user_id"), table_name="llm_call_logs")
    op.drop_column("llm_call_logs", "user_id")
    op.drop_index(op.f("ix_model_pool_entries_name"), table_name="model_pool_entries")
    op.drop_table("model_pool_entries")
