"""Retain execution identity independently of dialogue checkpoints.

Revision ID: b42f9c8e7310
Revises: a31e8d7f6290
"""

import sqlalchemy as sa
from alembic import op

revision = "b42f9c8e7310"
down_revision = "a31e8d7f6290"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = None if op.get_context().as_sql else sa.inspect(op.get_bind())
    if inspector is not None and inspector.has_table("action_executions"):
        return
    op.create_table(
        "action_executions",
        sa.Column("action_id", sa.String(36), primary_key=True),
        sa.Column("scope_session_id", sa.Integer(), nullable=False),
        sa.Column(
            "live_session_id",
            sa.Integer(),
            sa.ForeignKey("chat_sessions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("scope", sa.String(96), nullable=False),
        sa.Column("tool_name", sa.String(255), nullable=False),
        sa.Column("arguments_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("result", sa.JSON(none_as_null=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            "status IN ('in_progress', 'completed', 'unknown')", name="ck_action_execution_status"
        ),
        sa.CheckConstraint(
            "scope_session_id > 0 AND scope_session_id <= 2147483647",
            name="ck_action_execution_session_scope",
        ),
        sa.CheckConstraint("user_id >= 0", name="ck_action_execution_user_scope"),
        sa.CheckConstraint(
            "(status = 'completed' AND result IS NOT NULL) OR "
            "(status != 'completed' AND result IS NULL)",
            name="ck_action_execution_result",
        ),
    )
    op.create_index(
        "ix_action_executions_scope_session_id", "action_executions", ["scope_session_id"]
    )


def downgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name != "postgresql":
            raise RuntimeError("Action ledger downgrade requires online data checks for SQLite")
        op.execute("LOCK TABLE action_executions IN ACCESS EXCLUSIVE MODE")
        op.execute(
            "DO $$ BEGIN IF EXISTS (SELECT 1 FROM action_executions) THEN "
            "RAISE EXCEPTION 'Action execution data prevents lossless downgrade'; "
            "END IF; END $$"
        )
    else:
        db = op.get_bind()
        if context.dialect.name == "postgresql":
            op.execute("LOCK TABLE action_executions IN ACCESS EXCLUSIVE MODE")
        elif context.dialect.name == "sqlite":
            # Acquire the database write gate before checking retained rows.
            op.execute("UPDATE action_executions SET status = status WHERE 0")
        else:
            raise RuntimeError("Lossless action ledger downgrade requires PostgreSQL or SQLite")
        if db.execute(sa.text("SELECT 1 FROM action_executions LIMIT 1")).first() is not None:
            raise RuntimeError("Action execution data prevents lossless downgrade")
    op.drop_index("ix_action_executions_scope_session_id", table_name="action_executions")
    op.drop_table("action_executions")
