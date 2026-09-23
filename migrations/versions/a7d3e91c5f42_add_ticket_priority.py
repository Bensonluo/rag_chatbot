"""add handoff ticket queue priority

Revision ID: a7d3e91c5f42
Revises: c4e9f27b6a3d
Create Date: 2026-09-24

Priority routing for the human-agent queue: emotion and high-value
refund escalations (priority 1) are served before explicit requests
(priority 2), FIFO within each tier. server_default backfills
existing rows to the normal tier.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7d3e91c5f42"
down_revision: str | None = "c4e9f27b6a3d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "handoff_tickets",
        sa.Column(
            "priority",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("2"),
        ),
    )


def downgrade() -> None:
    op.drop_column("handoff_tickets", "priority")
