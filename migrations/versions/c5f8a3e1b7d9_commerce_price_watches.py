"""Commerce price watches: one pre-authorized watch rule per visitor and SKU.

Revision ID: c5f8a3e1b7d9
Revises: b9d4e6f2a1c8

Plain additive migration (one new table, one FK to ``guest_visitors`` with
CASCADE): safe on both SQLite and PostgreSQL, no rebuilds, no data moves.
Matches the ORM in ``app.models.database.commerce_watch``.
"""

import sqlalchemy as sa
from alembic import op

revision = "c5f8a3e1b7d9"
down_revision = "b9d4e6f2a1c8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "commerce_price_watches",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("visitor_id", sa.String(36), nullable=False),
        sa.Column("sku_id", sa.String(64), nullable=False),
        sa.Column("sku_name", sa.String(128), nullable=False),
        sa.Column("target_price_minor", sa.Integer(), nullable=False),
        sa.Column("condition", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("triggered_price_minor", sa.Integer(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(
            ["visitor_id"],
            ["guest_visitors.id"],
            name="fk_commerce_price_watches_visitor_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("visitor_id", "sku_id", name="uq_commerce_price_watches_visitor_sku"),
        sa.CheckConstraint(
            "status IN ('active', 'triggered', 'asked', 'cancelled')",
            name="ck_commerce_price_watches_status",
        ),
        sa.CheckConstraint(
            "condition IN ('at_or_below')",
            name="ck_commerce_price_watches_condition",
        ),
        sa.CheckConstraint("target_price_minor >= 1", name="ck_commerce_price_watches_target"),
    )
    op.create_index(
        "ix_commerce_price_watches_visitor_id",
        "commerce_price_watches",
        ["visitor_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_commerce_price_watches_visitor_id", table_name="commerce_price_watches")
    op.drop_table("commerce_price_watches")
