"""Add an independent human conversation channel and automation generation.

Revision ID: a31e8d7f6290
Revises: d4c9f1a82e70
"""

import sqlalchemy as sa
from alembic import op

revision = "a31e8d7f6290"
down_revision = "d4c9f1a82e70"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = None if op.get_context().as_sql else sa.inspect(op.get_bind())
    columns = (
        {column["name"] for column in inspector.get_columns("chat_sessions")}
        if inspector
        else set()
    )
    if "automation_epoch" not in columns:
        op.add_column(
            "chat_sessions",
            sa.Column("automation_epoch", sa.Integer(), nullable=False, server_default="0"),
        )
    if inspector is None or not inspector.has_table("handoff_messages"):
        op.create_table(
            "handoff_messages",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "session_id",
                sa.Integer(),
                sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "ticket_id",
                sa.Integer(),
                sa.ForeignKey("handoff_tickets.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("author_user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("author_kind", sa.String(16), nullable=False),
            sa.Column("client_message_id", sa.String(36), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("resolves_ticket", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.UniqueConstraint(
                "session_id",
                "author_user_id",
                "client_message_id",
                name="uq_handoff_message_client",
            ),
            sa.CheckConstraint(
                "author_kind IN ('customer', 'staff')", name="ck_handoff_message_author_kind"
            ),
        )
        op.create_index(
            "ix_handoff_messages_session_id_id", "handoff_messages", ["session_id", "id"]
        )


def downgrade() -> None:
    # Older application binaries also bypass private-session ownership in
    # Demo mode. Do not allow a schema rollback while any private rows remain.
    if op.get_context().as_sql:
        condition = "EXISTS (SELECT 1 FROM handoff_messages) OR EXISTS (SELECT 1 FROM chat_sessions WHERE automation_epoch > 0 OR session_metadata ->> 'access_mode' = 'private')"
        if op.get_context().dialect.name == "postgresql":
            op.execute("LOCK TABLE handoff_messages, chat_sessions IN ACCESS EXCLUSIVE MODE")
            op.execute(
                f"DO $$ BEGIN IF {condition} THEN RAISE EXCEPTION 'Human conversation data prevents lossless downgrade'; END IF; END $$"
            )
        elif op.get_context().dialect.name == "sqlite":
            raise RuntimeError("SQLite human-channel downgrade requires online data checks")
        else:
            raise RuntimeError("Lossless human-channel downgrade requires PostgreSQL or SQLite")
    else:
        db = op.get_bind()
        if op.get_context().dialect.name == "postgresql":
            op.execute("LOCK TABLE handoff_messages, chat_sessions IN ACCESS EXCLUSIVE MODE")
        elif op.get_context().dialect.name == "sqlite":
            # Enter a real write transaction before checking retained data.
            op.execute("UPDATE chat_sessions SET automation_epoch = automation_epoch WHERE 0")
        human = db.execute(sa.text("SELECT 1 FROM handoff_messages LIMIT 1")).first()
        generation = db.execute(
            sa.text("SELECT 1 FROM chat_sessions WHERE automation_epoch > 0 LIMIT 1")
        ).first()
        private = None
        if "session_metadata" in {c["name"] for c in sa.inspect(db).get_columns("chat_sessions")}:
            sessions = sa.table("chat_sessions", sa.column("session_metadata", sa.JSON))
            private = db.execute(
                sa.select(sa.literal(1))
                .where(sessions.c.session_metadata["access_mode"].as_string() == "private")
                .limit(1)
            ).first()
        if human is not None or generation is not None or private is not None:
            raise RuntimeError("Human conversation data prevents lossless downgrade")
    op.drop_index("ix_handoff_messages_session_id_id", table_name="handoff_messages")
    op.drop_table("handoff_messages")
    op.drop_column("chat_sessions", "automation_epoch")
