"""add knowledge_gaps telemetry table

Revision ID: c4e9f27b6a3d
Revises: ad2507c9a1e9
Create Date: 2026-09-24

Knowledge-gap telemetry: knowledge-intent turns whose retrieval produced
no usable context. Industry-standard KB curation input (Zendesk
knowledge-gap reporting, 阿里小蜜 知识缺口挖掘). Written best-effort;
telemetry only.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4e9f27b6a3d"
down_revision: str | None = "ad2507c9a1e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_gaps",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("normalized_query", sa.String(length=200), nullable=False),
        sa.Column("intent", sa.String(length=50), nullable=False),
        sa.Column("top_score", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_knowledge_gaps_normalized_query"),
        "knowledge_gaps",
        ["normalized_query"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_knowledge_gaps_normalized_query"), table_name="knowledge_gaps")
    op.drop_table("knowledge_gaps")
