"""Retain purchase orchestration and independent mock-provider facts.

Revision ID: e75c9d1064a3
Revises: d64b8c0f9532
"""

from typing import Any

import sqlalchemy as sa
from alembic import op

revision = "e75c9d1064a3"
down_revision = "d64b8c0f9532"
branch_labels = None
depends_on = None

TABLES = (
    "commerce_purchases",
    "commerce_operations",
    "commerce_confirmations",
    "commerce_actions",
    "commerce_mock_objects",
    "commerce_mock_receipts",
)


def _timestamps() -> list[sa.Column[Any]]:
    return [
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    ]


def upgrade() -> None:
    inspector = None if op.get_context().as_sql else sa.inspect(op.get_bind())

    def create(name: str, columns: list[Any], indexes: tuple[str, ...] = ()) -> None:
        if inspector is not None and inspector.has_table(name):
            return
        op.create_table(name, *columns, *_timestamps())
        for column in indexes:
            op.create_index(f"ix_{name}_{column}", name, [column])

    create(
        "commerce_purchases",
        [
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "owner_user_id",
                sa.Integer(),
                sa.ForeignKey("users.id", ondelete="RESTRICT"),
                nullable=False,
            ),
            sa.Column("active_slot", sa.String(96), nullable=True),
            sa.Column("mode", sa.String(16), nullable=False),
            sa.Column("merchant_id", sa.String(64), nullable=False),
            sa.Column("currency", sa.String(3), nullable=False),
            sa.Column("stage", sa.String(32), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("generation", sa.Integer(), nullable=False),
            sa.Column("sku_id", sa.String(64), nullable=True),
            sa.Column("quantity", sa.Integer(), nullable=False),
            sa.Column("cart_id", sa.String(64), nullable=True),
            sa.Column("cart_version", sa.Integer(), nullable=True),
            sa.Column("cart_snapshot", sa.JSON(none_as_null=True), nullable=True),
            sa.Column("quote_snapshot", sa.JSON(none_as_null=True), nullable=True),
            sa.Column("confirmation_id", sa.String(36), nullable=True),
            sa.Column("order_id", sa.String(64), nullable=True),
            sa.Column("payment_id", sa.String(64), nullable=True),
            sa.Column("payment_status", sa.String(32), nullable=True),
            sa.Column("active_operation_id", sa.String(36), nullable=True),
            sa.Column("error", sa.String(255), nullable=True),
            sa.UniqueConstraint("active_slot", name="uq_commerce_active_slot"),
            sa.CheckConstraint(
                "version >= 0 AND generation >= 0", name="ck_commerce_purchase_versions"
            ),
            sa.CheckConstraint(
                "quantity >= 0 AND quantity <= 10", name="ck_commerce_purchase_quantity"
            ),
        ],
        ("owner_user_id",),
    )
    create(
        "commerce_operations",
        [
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "purchase_id",
                sa.String(36),
                sa.ForeignKey("commerce_purchases.id", ondelete="RESTRICT"),
                nullable=False,
            ),
            sa.Column("kind", sa.String(32), nullable=False),
            sa.Column("generation", sa.Integer(), nullable=False),
            sa.Column("status", sa.String(16), nullable=False),
            sa.Column("request_hash", sa.String(64), nullable=False),
            sa.Column("request", sa.JSON(), nullable=False),
            sa.Column("result", sa.JSON(none_as_null=True), nullable=True),
            sa.Column("error", sa.String(255), nullable=True),
            sa.UniqueConstraint(
                "purchase_id", "kind", "generation", name="uq_commerce_operation_generation"
            ),
            sa.CheckConstraint(
                "status IN ('reserved','unknown','completed','failed')",
                name="ck_commerce_operation_status",
            ),
        ],
        ("purchase_id",),
    )
    create(
        "commerce_confirmations",
        [
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "purchase_id",
                sa.String(36),
                sa.ForeignKey("commerce_purchases.id", ondelete="RESTRICT"),
                nullable=False,
            ),
            sa.Column("purchase_version", sa.Integer(), nullable=False),
            sa.Column("generation", sa.Integer(), nullable=False),
            sa.Column("snapshot", sa.JSON(), nullable=False),
            sa.Column("snapshot_hash", sa.String(64), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("operation_id", sa.String(36), nullable=True),
        ],
        ("purchase_id",),
    )
    create(
        "commerce_actions",
        [
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "purchase_id",
                sa.String(36),
                sa.ForeignKey("commerce_purchases.id", ondelete="RESTRICT"),
                nullable=False,
            ),
            sa.Column("kind", sa.String(32), nullable=False),
            sa.Column("expected_version", sa.Integer(), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        ],
        ("purchase_id",),
    )
    create(
        "commerce_mock_objects",
        [
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("kind", sa.String(16), nullable=False),
            sa.Column("owner_user_id", sa.Integer(), nullable=False),
            sa.Column("parent_id", sa.String(64), nullable=True),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.UniqueConstraint("kind", "parent_id", name="uq_commerce_mock_parent"),
        ],
        ("kind",),
    )
    create(
        "commerce_mock_receipts",
        [
            sa.Column("request_key", sa.String(96), primary_key=True),
            sa.Column("owner_user_id", sa.Integer(), nullable=False),
            sa.Column("kind", sa.String(32), nullable=False),
            sa.Column("request_hash", sa.String(64), nullable=False),
            sa.Column("status", sa.String(16), nullable=False),
            sa.Column("result", sa.JSON(), nullable=False),
        ],
    )


def downgrade() -> None:
    context = op.get_context()
    if context.as_sql:
        if context.dialect.name != "postgresql":
            raise RuntimeError("Commerce downgrade requires online retained-data checks for SQLite")
        op.execute("LOCK TABLE " + ", ".join(TABLES) + " IN ACCESS EXCLUSIVE MODE")
        condition = " OR ".join(f"EXISTS (SELECT 1 FROM {name})" for name in TABLES)
        op.execute(
            "DO $$ BEGIN IF "
            + condition
            + " THEN RAISE EXCEPTION 'Commerce data prevents lossless downgrade'; END IF; END $$"
        )
    else:
        db = op.get_bind()
        if context.dialect.name == "postgresql":
            op.execute("LOCK TABLE " + ", ".join(TABLES) + " IN ACCESS EXCLUSIVE MODE")
        elif context.dialect.name == "sqlite":
            op.execute("UPDATE commerce_purchases SET version=version WHERE 0")
        else:
            raise RuntimeError("Commerce downgrade supports PostgreSQL or SQLite only")
        if any(
            db.execute(sa.text(f"SELECT 1 FROM {name} LIMIT 1")).first() is not None
            for name in TABLES
        ):
            raise RuntimeError("Commerce data prevents lossless downgrade")
    for table in reversed(TABLES):
        op.drop_table(table)
