"""add chat_sessions.resolved_at + reopened_count (session resolution state)

Revision ID: c5f2a8e3d17b
Revises: e7b1d4c8a290
Create Date: 2026-10-03

Review 2026-09-26 #11: containment (no ticket, no downvote) is the
absence of failure signals, not a verified resolution. These columns
carry the positive half: resolved_at is stamped when the user's own
message confirms the problem is solved (ChatMessagePersister hooks
``is_resolution_confirmation``), and reopened_count increments when a
follow-up message arrives on a resolved session — the "same problem
short-term reopen" the review asked to make measurable. Every existing
row is open and never-reopened, which is exactly what the defaults say.
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "c5f2a8e3d17b"
down_revision: str | None = "e7b1d4c8a290"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chat_sessions",
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "chat_sessions",
        sa.Column("reopened_count", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("chat_sessions", "reopened_count")
    op.drop_column("chat_sessions", "resolved_at")
