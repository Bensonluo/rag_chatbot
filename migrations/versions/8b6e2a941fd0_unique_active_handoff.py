"""Allow at most one active human handoff per chat session.

Revision ID: 8b6e2a941fd0
Revises: f9a2c7d6b410
"""

import sqlalchemy as sa
from alembic import op

revision = "8b6e2a941fd0"
down_revision = "f9a2c7d6b410"
branch_labels = None
depends_on = None

INDEX_NAME = "uq_handoff_tickets_active_session"
ACTIVE_PREDICATE = "status IN ('open', 'claimed')"


def upgrade() -> None:
    if not op.get_context().as_sql:
        connection = op.get_bind()
        # Never silently resolve or delete existing customer requests to make
        # the constraint fit. Operators must reconcile these tickets first.
        duplicates = connection.execute(
            sa.text(
                "SELECT session_id, COUNT(*) AS active_count FROM handoff_tickets "
                f"WHERE {ACTIVE_PREDICATE} GROUP BY session_id "
                "HAVING COUNT(*) > 1 ORDER BY session_id LIMIT 10"
            )
        ).all()
        if duplicates:
            sample = ", ".join(
                f"{row.session_id} ({row.active_count} active)" for row in duplicates
            )
            raise RuntimeError(
                "Cannot enforce unique active handoff tickets: duplicate open/claimed "
                f"tickets exist for session(s): {sample}. "
                "Reconcile these requests with the support team before retrying the migration; "
                "no ticket data has been changed."
            )
        indexes = sa.inspect(connection).get_indexes("handoff_tickets")
        if any(index["name"] == INDEX_NAME for index in indexes):
            # A fresh ORM-created database already has this constraint.
            return

    op.create_index(
        INDEX_NAME,
        "handoff_tickets",
        ["session_id"],
        unique=True,
        postgresql_where=sa.text(ACTIVE_PREDICATE),
        sqlite_where=sa.text(ACTIVE_PREDICATE),
    )


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="handoff_tickets")
