"""add handoff ticket claimed_at

Revision ID: b8f4c2d90a17
Revises: a7d3e91c5f42
Create Date: 2026-09-24

AHT observability (GB/T 47746—2026 时效要求): claimed_at is the
boundary between queue wait and agent handling. Pickup duration
(created→claimed) and handle duration (claimed→resolved) derive from
it. Nullable — open tickets have no claim yet; existing claimed rows
backfill to NULL (their pickup time is unrecoverable, so they simply
do not sample into the averages).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b8f4c2d90a17"
down_revision: str | None = "a7d3e91c5f42"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "handoff_tickets",
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("handoff_tickets", "claimed_at")
