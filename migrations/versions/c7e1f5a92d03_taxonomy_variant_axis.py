"""add variant axis columns to commerce taxonomy categories

Revision ID: c7e1f5a92d03
Revises: f3a8b2c91d04
Create Date: 2026-11-01

The category catalog expanded from 25 office-electronics categories to a
151-category general marketplace. Each generated category declares its own
variant axis (颜色 for electronics, 尺码 for apparel, 规格 for fresh food,
...). The axis is registry-internal metadata driving SKU addressing in the
provider; the public catalog wire stays 4-field. Existing rows backfill to
the historical default axis (颜色 with the four standard colors), matching
the pre-migration behavior for the 25 original categories.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7e1f5a92d03"
down_revision: str | None = "f3a8b2c91d04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DEFAULT_VARIANT_VALUES = '["黑色", "白色", "银色", "灰色"]'


def upgrade() -> None:
    op.add_column(
        "commerce_taxonomy_categories",
        sa.Column("variant_key", sa.String(length=32), nullable=False, server_default="颜色"),
    )
    op.add_column(
        "commerce_taxonomy_categories",
        sa.Column(
            "variant_values",
            sa.JSON(),
            nullable=False,
            server_default=sa.text(f"'{_DEFAULT_VARIANT_VALUES}'"),
        ),
    )


def downgrade() -> None:
    op.drop_column("commerce_taxonomy_categories", "variant_values")
    op.drop_column("commerce_taxonomy_categories", "variant_key")
