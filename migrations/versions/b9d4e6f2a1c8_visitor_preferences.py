"""Visitor preference memory: live table plus append-only audit log.

Revision ID: b9d4e6f2a1c8
Revises: b7f3d2c9a4e1

Plain additive migration (two new tables, one FK to ``guest_visitors`` with
CASCADE): safe on both SQLite and PostgreSQL, no rebuilds, no data moves.
``pref_value`` is JSONB on PostgreSQL, generic JSON elsewhere, matching the
ORM variant in ``app.models.database.visitor_preference``.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "b9d4e6f2a1c8"
down_revision = "b7f3d2c9a4e1"
branch_labels = None
depends_on = None

JsonValue = sa.JSON(none_as_null=True).with_variant(
    postgresql.JSONB(none_as_null=True), "postgresql"
)


def upgrade() -> None:
    op.create_table(
        "visitor_preferences",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("visitor_id", sa.String(36), nullable=False),
        sa.Column("pref_key", sa.String(64), nullable=False),
        sa.Column("pref_value", JsonValue, nullable=False),
        sa.Column("source", sa.String(16), nullable=False, server_default="inferred"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(
            ["visitor_id"],
            ["guest_visitors.id"],
            name="fk_visitor_preferences_visitor_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("visitor_id", "pref_key", name="uq_visitor_preferences_visitor_key"),
        sa.CheckConstraint(
            "source IN ('inferred', 'confirmed')", name="ck_visitor_preferences_source"
        ),
        sa.CheckConstraint("length(pref_key) >= 1", name="ck_visitor_preferences_key_nonempty"),
    )
    op.create_index(
        "ix_visitor_preferences_visitor_id", "visitor_preferences", ["visitor_id"], unique=False
    )

    op.create_table(
        "visitor_preference_audit",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("visitor_id", sa.String(36), nullable=False),
        sa.Column("pref_key", sa.String(64), nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("old_value", JsonValue, nullable=True),
        sa.Column("new_value", JsonValue, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            "action IN ('save', 'forget')", name="ck_visitor_preference_audit_action"
        ),
    )
    op.create_index(
        "ix_visitor_preference_audit_visitor_id",
        "visitor_preference_audit",
        ["visitor_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_visitor_preference_audit_visitor_id", table_name="visitor_preference_audit")
    op.drop_table("visitor_preference_audit")
    op.drop_index("ix_visitor_preferences_visitor_id", table_name="visitor_preferences")
    op.drop_table("visitor_preferences")
