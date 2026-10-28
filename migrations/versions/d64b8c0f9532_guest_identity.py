"""Anonymous browser principals without fabricated account credentials.

Revision ID: d64b8c0f9532
Revises: c53a7d9e8421

SQLite requires stopped writers and a backup: its users-parent rebuild runs
in an explicit FK maintenance transaction after committing prior revisions.
"""

from collections.abc import Callable

import sqlalchemy as sa
from alembic import op

revision = "d64b8c0f9532"
down_revision = "c53a7d9e8421"
branch_labels = None
depends_on = None

_CHECK = "ck_users_identity_kind"
_IDENTITY = (
    "(is_guest = true AND email IS NULL AND hashed_password IS NULL AND is_admin = false) "
    "OR (is_guest = false AND email IS NOT NULL AND hashed_password IS NOT NULL)"
)


def _offline_users() -> sa.Table:
    """Frozen c53 shape after adding is_guest, before relaxing credentials."""
    return sa.Table(
        "users",
        sa.MetaData(),
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("is_admin", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("is_guest", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def _sqlite_maintenance(operation: Callable[[], None]) -> None:
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
                    raise RuntimeError("Guest identity migration found foreign-key violations")
                db.exec_driver_sql("COMMIT")
            except BaseException:
                db.exec_driver_sql("ROLLBACK")
                raise
        finally:
            db.exec_driver_sql("PRAGMA foreign_keys=ON")


def _create_visitors() -> None:
    op.create_table(
        "guest_visitors",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("current_session_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["current_session_id"], ["chat_sessions.id"], ondelete="SET NULL"),
        sa.UniqueConstraint("user_id", name="uq_guest_visitors_user_id"),
        sa.UniqueConstraint("token_hash", name="uq_guest_visitors_token_hash"),
        sa.CheckConstraint("length(token_hash) = 64", name="ck_guest_visitors_token_hash_length"),
    )


def upgrade() -> None:
    context = op.get_context()
    dialect = context.dialect.name
    if dialect not in {"sqlite", "postgresql"}:
        raise RuntimeError("Guest identity migration requires PostgreSQL or SQLite")
    if context.as_sql:

        def offline_upgrade() -> None:
            op.add_column(
                "users",
                sa.Column("is_guest", sa.Boolean(), nullable=False, server_default=sa.false()),
            )
            if dialect == "sqlite":
                with op.batch_alter_table(
                    "users", recreate="always", copy_from=_offline_users()
                ) as batch:
                    batch.alter_column("email", existing_type=sa.String(255), nullable=True)
                    batch.alter_column(
                        "hashed_password", existing_type=sa.String(255), nullable=True
                    )
                    batch.create_check_constraint(_CHECK, _IDENTITY)
            else:
                op.alter_column("users", "email", existing_type=sa.String(255), nullable=True)
                op.alter_column(
                    "users", "hashed_password", existing_type=sa.String(255), nullable=True
                )
                op.create_check_constraint(_CHECK, "users", _IDENTITY)
            _create_visitors()

        if dialect == "sqlite":
            _sqlite_maintenance(offline_upgrade)
        else:
            offline_upgrade()
        return

    db = op.get_bind()
    inspector = sa.inspect(db)
    columns = {column["name"]: column for column in inspector.get_columns("users")}
    has_check = any(item["name"] == _CHECK for item in inspector.get_check_constraints("users"))
    visitor_exists = inspector.has_table("guest_visitors")

    def online_upgrade() -> None:
        if "is_guest" not in columns:
            op.add_column(
                "users",
                sa.Column("is_guest", sa.Boolean(), nullable=False, server_default=sa.false()),
            )
        invalid = db.execute(
            sa.text(f"SELECT id FROM users WHERE is_guest IS NULL OR NOT ({_IDENTITY}) LIMIT 1")
        ).first()
        if invalid is not None:
            raise RuntimeError("Existing identity is invalid; explicit repair is required")
        needs_rebuild = (
            not has_check
            or not columns["email"]["nullable"]
            or not columns["hashed_password"]["nullable"]
        )
        if needs_rebuild:
            if dialect == "sqlite":
                with op.batch_alter_table("users", recreate="always") as batch:
                    batch.alter_column("email", existing_type=sa.String(255), nullable=True)
                    batch.alter_column(
                        "hashed_password", existing_type=sa.String(255), nullable=True
                    )
                    if not has_check:
                        batch.create_check_constraint(_CHECK, _IDENTITY)
            else:
                op.alter_column("users", "email", existing_type=sa.String(255), nullable=True)
                op.alter_column(
                    "users", "hashed_password", existing_type=sa.String(255), nullable=True
                )
                if not has_check:
                    op.create_check_constraint(_CHECK, "users", _IDENTITY)
        if not visitor_exists:
            _create_visitors()

    if dialect == "sqlite":
        _sqlite_maintenance(online_upgrade)
    else:
        online_upgrade()


def downgrade() -> None:
    context = op.get_context()
    dialect = context.dialect.name
    if dialect not in {"sqlite", "postgresql"}:
        raise RuntimeError("Guest identity migration requires PostgreSQL or SQLite")
    if context.as_sql:
        if dialect != "postgresql":
            raise RuntimeError("SQLite guest downgrade requires online data checks")
        op.execute("LOCK TABLE users, guest_visitors IN ACCESS EXCLUSIVE MODE")
        op.execute(
            "DO $$ BEGIN IF EXISTS (SELECT 1 FROM users WHERE is_guest) OR EXISTS "
            "(SELECT 1 FROM guest_visitors) THEN RAISE EXCEPTION "
            "'Guest identity data prevents downgrade'; END IF; END $$"
        )
        op.drop_table("guest_visitors")
        op.drop_constraint(_CHECK, "users", type_="check")
        op.alter_column("users", "email", existing_type=sa.String(255), nullable=False)
        op.alter_column("users", "hashed_password", existing_type=sa.String(255), nullable=False)
        op.drop_column("users", "is_guest")
        return

    db = op.get_bind()
    visitor_exists = sa.inspect(db).has_table("guest_visitors")

    def online_downgrade() -> None:
        if dialect == "postgresql":
            targets = "users, guest_visitors" if visitor_exists else "users"
            op.execute(f"LOCK TABLE {targets} IN ACCESS EXCLUSIVE MODE")
        guest = db.execute(sa.text("SELECT 1 FROM users WHERE is_guest LIMIT 1")).first()
        visitor = (
            db.execute(sa.text("SELECT 1 FROM guest_visitors LIMIT 1")).first()
            if visitor_exists
            else None
        )
        if guest is not None or visitor is not None:
            raise RuntimeError("Guest identity data prevents downgrade")
        if visitor_exists:
            op.drop_table("guest_visitors")
        if dialect == "sqlite":
            with op.batch_alter_table("users", recreate="always") as batch:
                batch.drop_constraint(_CHECK, type_="check")
                batch.alter_column("email", existing_type=sa.String(255), nullable=False)
                batch.alter_column("hashed_password", existing_type=sa.String(255), nullable=False)
                batch.drop_column("is_guest")
        else:
            op.drop_constraint(_CHECK, "users", type_="check")
            op.alter_column("users", "email", existing_type=sa.String(255), nullable=False)
            op.alter_column(
                "users", "hashed_password", existing_type=sa.String(255), nullable=False
            )
            op.drop_column("users", "is_guest")

    if dialect == "sqlite":
        _sqlite_maintenance(online_downgrade)
    else:
        online_downgrade()
