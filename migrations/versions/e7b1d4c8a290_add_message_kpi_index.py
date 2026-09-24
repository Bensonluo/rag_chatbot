"""add messages(role, created_at) index for the one-shot KPI query

Revision ID: e7b1d4c8a290
Revises: b8f4c2d90a17
Create Date: 2026-09-25

The session-level one-shot aggregate (TicketRepository.get_one_shot_stats)
filters ``role == ASSISTANT AND created_at >= since`` over messages — the
system's biggest table at 50-60M daily visits — and the Prometheus bridge
reruns it on a 300s timer per replica. Without an index each run is a
sequential scan: the KPI observability layer itself becomes the database
hotspot. This composite index covers the exact predicate.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e7b1d4c8a290"
down_revision: str | None = "b8f4c2d90a17"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

def upgrade() -> None:
    # if_not_exists: replicas booting concurrently each run migrations;
    # CREATE INDEX IF NOT EXISTS keeps racing boots convergent without
    # one failing the deploy.
    op.create_index(
        "ix_messages_role_created_at",
        "messages",
        ["role", "created_at"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_messages_role_created_at", table_name="messages", if_exists=True)
