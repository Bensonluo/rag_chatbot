"""Give each conversation incarnation a server-generated identity.

Revision ID: c53a7d9e8421
Revises: b42f9c8e7310

SQLite maintenance requires stopped application writers and a backup. Its
parent-table rebuild commits preceding migrations, then uses an explicit
write transaction with foreign keys restored on either success or failure.
"""

from collections.abc import Callable
from uuid import UUID, uuid4

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine import Connection

revision = "c53a7d9e8421"
down_revision = "b42f9c8e7310"
branch_labels = None
depends_on = None

_UNIQUE = "uq_chat_sessions_session_key"
_SQLITE_UUID = (
    "lower(hex(randomblob(4))) || '-' || lower(hex(randomblob(2))) || '-4' || "
    "substr(lower(hex(randomblob(2))), 2, 3) || '-' || "
    "substr('89ab', (random() & 3) + 1, 1) || "
    "substr(lower(hex(randomblob(2))), 2, 3) || '-' || lower(hex(randomblob(6)))"
)


def _offline_sessions() -> sa.Table:
    """Frozen b42 table shape after the additive nullable-key operation."""
    metadata = sa.MetaData()
    sa.Table("users", metadata, sa.Column("id", sa.Integer(), primary_key=True))
    return sa.Table(
        "chat_sessions",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("memory_type", sa.String(50), nullable=False),
        sa.Column("context_window", sa.Integer(), nullable=False),
        sa.Column("session_metadata", sa.JSON(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reopened_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("automation_epoch", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("session_key", sa.String(36), nullable=True),
    )


def _sqlite_rebuild(operation: Callable[[], None]) -> None:
    context = op.get_context()
    with context.autocommit_block():
        if context.as_sql:
            op.execute("PRAGMA foreign_keys=OFF")
            op.execute("BEGIN IMMEDIATE")
            operation()
            op.execute("PRAGMA foreign_key_check")
            op.execute("COMMIT")
            op.execute("PRAGMA foreign_keys=ON")
            return
        db = op.get_bind()
        db.exec_driver_sql("PRAGMA foreign_keys=OFF")
        try:
            db.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                operation()
                if db.exec_driver_sql("PRAGMA foreign_key_check").first() is not None:
                    raise RuntimeError("Session incarnation migration found foreign-key violations")
                db.exec_driver_sql("COMMIT")
            except BaseException:
                db.exec_driver_sql("ROLLBACK")
                raise
        finally:
            db.exec_driver_sql("PRAGMA foreign_keys=ON")


def _validate_keys(db: Connection) -> None:
    for row in db.execute(
        sa.text("SELECT session_key FROM chat_sessions WHERE session_key IS NOT NULL")
    ):
        value: object = row[0]
        try:
            valid = isinstance(value, str) and str(UUID(value)) == value
        except ValueError:
            valid = False
        if not valid:
            raise RuntimeError("Existing session key is invalid; explicit repair is required")
    duplicate = db.execute(
        sa.text(
            "SELECT 1 FROM chat_sessions WHERE session_key IS NOT NULL GROUP BY session_key HAVING COUNT(*) > 1 LIMIT 1"
        )
    ).first()
    if duplicate is not None:
        raise RuntimeError("Existing session keys are duplicated; explicit repair is required")


def upgrade() -> None:
    context = op.get_context()
    if context.dialect.name not in {"postgresql", "sqlite"}:
        raise RuntimeError("Session incarnation migration requires PostgreSQL or SQLite")
    if context.as_sql:

        def offline_upgrade() -> None:
            op.add_column("chat_sessions", sa.Column("session_key", sa.String(36), nullable=True))
            expression = (
                "gen_random_uuid()::text" if context.dialect.name == "postgresql" else _SQLITE_UUID
            )
            op.execute(
                f"UPDATE chat_sessions SET session_key = {expression} WHERE session_key IS NULL"
            )
            if context.dialect.name == "sqlite":
                with op.batch_alter_table(
                    "chat_sessions", copy_from=_offline_sessions(), recreate="always"
                ) as batch:
                    batch.alter_column("session_key", existing_type=sa.String(36), nullable=False)
                    batch.create_unique_constraint(_UNIQUE, ["session_key"])
            else:
                op.alter_column(
                    "chat_sessions", "session_key", existing_type=sa.String(36), nullable=False
                )
                op.create_unique_constraint(_UNIQUE, "chat_sessions", ["session_key"])

        if context.dialect.name == "sqlite":
            _sqlite_rebuild(offline_upgrade)
        else:
            offline_upgrade()
        return

    db = op.get_bind()
    inspector = sa.inspect(db)
    columns = {column["name"]: column for column in inspector.get_columns("chat_sessions")}
    unique_exists = any(
        constraint["name"] == _UNIQUE
        for constraint in inspector.get_unique_constraints("chat_sessions")
    )
    complete = "session_key" in columns and not columns["session_key"]["nullable"] and unique_exists
    if complete:
        _validate_keys(db)
        return

    def online_upgrade() -> None:
        if "session_key" not in columns:
            op.add_column("chat_sessions", sa.Column("session_key", sa.String(36), nullable=True))
        _validate_keys(db)
        if context.dialect.name == "postgresql":
            db.execute(
                sa.text(
                    "UPDATE chat_sessions SET session_key=gen_random_uuid()::text WHERE session_key IS NULL"
                )
            )
            op.alter_column(
                "chat_sessions", "session_key", existing_type=sa.String(36), nullable=False
            )
            if not unique_exists:
                op.create_unique_constraint(_UNIQUE, "chat_sessions", ["session_key"])
        else:
            for row in db.execute(
                sa.text("SELECT id FROM chat_sessions WHERE session_key IS NULL")
            ):
                db.execute(
                    sa.text(
                        "UPDATE chat_sessions SET session_key=:key WHERE id=:id AND session_key IS NULL"
                    ),
                    {"key": str(uuid4()), "id": row[0]},
                )
            with op.batch_alter_table("chat_sessions", recreate="always") as batch:
                batch.alter_column("session_key", existing_type=sa.String(36), nullable=False)
                if not unique_exists:
                    batch.create_unique_constraint(_UNIQUE, ["session_key"])

    if context.dialect.name == "sqlite":
        _sqlite_rebuild(online_upgrade)
    else:
        online_upgrade()


def downgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name != "postgresql":
            raise RuntimeError("SQLite session incarnation downgrade requires online data checks")
        op.execute("LOCK TABLE chat_sessions, handoff_messages IN ACCESS EXCLUSIVE MODE")
        op.execute(
            "DO $$ BEGIN IF EXISTS (SELECT 1 FROM chat_sessions WHERE session_metadata ->> 'access_mode' = 'private') OR EXISTS (SELECT 1 FROM handoff_messages) THEN RAISE EXCEPTION 'Private conversation data prevents session identity downgrade'; END IF; END $$"
        )
        op.drop_constraint(_UNIQUE, "chat_sessions", type_="unique")
        op.drop_column("chat_sessions", "session_key")
        return
    if context.dialect.name not in {"postgresql", "sqlite"}:
        raise RuntimeError("Session incarnation migration requires PostgreSQL or SQLite")
    db = op.get_bind()

    def online_downgrade() -> None:
        inspector = sa.inspect(db)
        human_table = inspector.has_table("handoff_messages")
        if context.dialect.name == "postgresql":
            targets = "chat_sessions, handoff_messages" if human_table else "chat_sessions"
            op.execute(f"LOCK TABLE {targets} IN ACCESS EXCLUSIVE MODE")
        sessions = sa.table("chat_sessions", sa.column("session_metadata", sa.JSON))
        private = db.execute(
            sa.select(sa.literal(1))
            .select_from(sessions)
            .where(sessions.c.session_metadata["access_mode"].as_string() == "private")
            .limit(1)
        ).first()
        human = (
            db.execute(sa.text("SELECT 1 FROM handoff_messages LIMIT 1")).first()
            if human_table
            else None
        )
        if private is not None or human is not None:
            raise RuntimeError("Private conversation data prevents session identity downgrade")
        if context.dialect.name == "sqlite":
            with op.batch_alter_table("chat_sessions", recreate="always") as batch:
                batch.drop_constraint(_UNIQUE, type_="unique")
                batch.drop_column("session_key")
        else:
            op.drop_constraint(_UNIQUE, "chat_sessions", type_="unique")
            op.drop_column("chat_sessions", "session_key")

    if context.dialect.name == "sqlite":
        _sqlite_rebuild(online_downgrade)
    else:
        online_downgrade()
