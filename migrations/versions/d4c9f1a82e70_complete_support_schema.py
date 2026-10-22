"""Migrate durable user memory and knowledge-gap resolution storage.

Revision ID: d4c9f1a82e70
Revises: 8b6e2a941fd0
"""

import sqlalchemy as sa
from alembic import op

revision = "d4c9f1a82e70"
down_revision = "8b6e2a941fd0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # These tables previously existed only in ORM-created databases. Adopt
    # them without replacing customer memory or curator resolution records.
    inspector = None if op.get_context().as_sql else sa.inspect(op.get_bind())
    if inspector is not None:
        priority = next(
            column
            for column in inspector.get_columns("handoff_tickets")
            if column["name"] == "priority"
        )
        if priority["default"] is None:
            # Historical create_all databases had only a Python-side default.
            # The versioned chain already supplied 2 in a7d3e91c5f42. Repair
            # adoption without changing the priority of any existing ticket.
            if op.get_context().dialect.name == "sqlite":
                with op.batch_alter_table("handoff_tickets") as batch:
                    batch.alter_column(
                        "priority",
                        existing_type=sa.Integer(),
                        existing_nullable=False,
                        server_default="2",
                    )
            else:
                op.alter_column(
                    "handoff_tickets",
                    "priority",
                    existing_type=sa.Integer(),
                    existing_nullable=False,
                    server_default="2",
                )
    if inspector is None or not inspector.has_table("user_facts"):
        op.create_table(
            "user_facts",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("fact", sa.Text(), nullable=False),
            sa.Column("category", sa.String(32), nullable=False),
            sa.Column("source_session_id", sa.String(64), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
        )
    if inspector is None or not any(
        index["name"] == "ix_user_facts_user_id" for index in inspector.get_indexes("user_facts")
    ):
        op.create_index("ix_user_facts_user_id", "user_facts", ["user_id"])

    if inspector is None or not inspector.has_table("knowledge_gap_resolutions"):
        op.create_table(
            "knowledge_gap_resolutions",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("normalized_query", sa.String(200), nullable=False),
            sa.Column(
                "resolved_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column("resolved_by", sa.Integer(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
                nullable=False,
            ),
        )
    if inspector is None or not any(
        index["name"] == "ix_knowledge_gap_resolutions_normalized_query"
        for index in inspector.get_indexes("knowledge_gap_resolutions")
    ):
        op.create_index(
            "ix_knowledge_gap_resolutions_normalized_query",
            "knowledge_gap_resolutions",
            ["normalized_query"],
            unique=True,
        )

    if op.get_context().dialect.supports_comments:
        op.alter_column(
            "chat_sessions",
            "resolved_at",
            existing_type=sa.DateTime(timezone=True),
            existing_nullable=True,
            comment="Last user-confirmed resolution (review #11); None = open/reopened",
        )
        op.alter_column(
            "chat_sessions",
            "reopened_count",
            existing_type=sa.Integer(),
            existing_nullable=False,
            comment="Times a resolved session was reopened by a follow-up message",
        )


def downgrade() -> None:
    if op.get_context().dialect.supports_comments:
        op.alter_column(
            "chat_sessions",
            "resolved_at",
            existing_type=sa.DateTime(timezone=True),
            existing_nullable=True,
            comment=None,
        )
        op.alter_column(
            "chat_sessions",
            "reopened_count",
            existing_type=sa.Integer(),
            existing_nullable=False,
            comment=None,
        )
    op.drop_table("knowledge_gap_resolutions")
    op.drop_table("user_facts")
