"""Drop the never-written PurchaseSession.quantity check constraint.

Revision ID: b7f3d2c9a4e1
Revises: c7e1f5a92d03

``commerce_purchases.quantity`` is a legacy single-SKU column: since
multi-line carts landed, the service layer never writes it (rows keep the
column default 0), so the ``0..10`` CHECK only rejects honest manual SQL on a
dead column. The model definition drops the constraint in the same change;
this migration removes it from existing databases. SQLite cannot DROP a
table-level CHECK natively, so its rebuild runs inside the same stopped-
writers foreign-key maintenance window the guest-identity migration uses.
Offline (``--sql``) mode is supported for PostgreSQL only; generating SQLite
rebuild SQL without a live schema is refused loudly instead of guessed.
"""

from collections.abc import Callable

import sqlalchemy as sa
from alembic import op

revision = "b7f3d2c9a4e1"
down_revision = "c7e1f5a92d03"
branch_labels = None
depends_on = None

_TABLE = "commerce_purchases"
_CHECK = "ck_commerce_purchase_quantity"
_RULE = "quantity >= 0 AND quantity <= 10"


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
                    raise RuntimeError("purchase quantity migration found foreign-key violations")
                db.exec_driver_sql("COMMIT")
            except BaseException:
                db.exec_driver_sql("ROLLBACK")
                raise
        finally:
            db.exec_driver_sql("PRAGMA foreign_keys=ON")


def _has_check(db: sa.engine.Connection) -> bool:
    return any(item["name"] == _CHECK for item in sa.inspect(db).get_check_constraints(_TABLE))


def upgrade() -> None:
    context = op.get_context()
    dialect = context.dialect.name
    if dialect not in {"sqlite", "postgresql"}:
        raise RuntimeError("Purchase quantity migration requires PostgreSQL or SQLite")

    if context.as_sql:
        if dialect != "postgresql":
            raise RuntimeError("Offline purchase quantity migration requires PostgreSQL")
        op.drop_constraint(_CHECK, _TABLE, type_="check")
        return

    db = op.get_bind()
    if not _has_check(db):
        # Fresh create_all schema or replayed migration: nothing to drop.
        return

    def drop_check() -> None:
        with op.batch_alter_table(_TABLE, recreate="always") as batch:
            batch.drop_constraint(_CHECK, type_="check")

    if dialect == "sqlite":
        _sqlite_maintenance(drop_check)
    else:
        op.drop_constraint(_CHECK, _TABLE, type_="check")


def downgrade() -> None:
    context = op.get_context()
    dialect = context.dialect.name
    if dialect not in {"sqlite", "postgresql"}:
        raise RuntimeError("Purchase quantity migration requires PostgreSQL or SQLite")

    if context.as_sql:
        if dialect != "postgresql":
            raise RuntimeError("Offline purchase quantity downgrade requires PostgreSQL")
        op.create_check_constraint(_CHECK, _TABLE, _RULE)
        return

    db = op.get_bind()
    if _has_check(db):
        return

    def create_check() -> None:
        with op.batch_alter_table(_TABLE, recreate="always") as batch:
            batch.create_check_constraint(_CHECK, _RULE)

    if dialect == "sqlite":
        _sqlite_maintenance(create_check)
    else:
        op.create_check_constraint(_CHECK, _TABLE, _RULE)
