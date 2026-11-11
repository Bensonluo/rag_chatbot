"""convert commerce taxonomy variant_values to JSONB on PostgreSQL

Revision ID: a3f7c9d2e5b1
Revises: c5f8a3e1b7d9
Create Date: 2026-11-12

c7e1f5a92d03 created ``commerce_taxonomy_categories.variant_values`` as
plain ``json`` on PostgreSQL. The ``json`` type has no equality operator,
which crashes Alembic's ``command.check`` server-default probe (CI runs it
against a real PostgreSQL 16 service container):

    SELECT '["黑色", ...]'::json = '["黑色", ...]'  -- operator does not
                                                      -- exist: json = unknown

JSONB has an equality operator, so the probe evaluates instead of erroring.
The column is registry-internal metadata (variant axis per category), never
compared in application queries, so the conversion is a pure type change:
``USING variant_values::text::jsonb`` re-parses the stored text exactly.

Databases created after c7e1f5a92d03 was amended already get JSONB from the
variant type; this revision converges pre-existing deployments (the demo
server) and is a no-op rewrite on fresh PostgreSQL databases. SQLite keeps
generic JSON (TEXT storage) on both paths, so this revision does nothing
there — SQLite cannot ALTER COLUMN without a table rebuild.
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a3f7c9d2e5b1"
down_revision: str | None = "c5f8a3e1b7d9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.alter_column(
        "commerce_taxonomy_categories",
        "variant_values",
        type_=postgresql.JSONB(),
        postgresql_using="variant_values::text::jsonb",
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.alter_column(
        "commerce_taxonomy_categories",
        "variant_values",
        type_=postgresql.JSON(),
        postgresql_using="variant_values::text::json",
    )
