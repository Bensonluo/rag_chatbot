"""The dead PurchaseSession.quantity CHECK is dropped, data-retentively.

``commerce_purchases.quantity`` is never written by the service layer; the
``0..10`` CHECK (``ck_commerce_purchase_quantity``) only existed to fail
honest manual SQL on that dead column. Migration ``b7f3d2c9a4e1`` removes it
from existing databases while keeping every purchase row, restores it on
downgrade, and refuses to guess offline SQLite rebuild SQL.
"""

import importlib.util
import re
from io import StringIO
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from app.config.settings import settings
from app.models.database.base import Base
from app.models.database.commerce import PurchaseSession

ROOT = Path(__file__).resolve().parents[4]
_HEAD = "b7f3d2c9a4e1"
_PARENT = "c7e1f5a92d03"
_CHECK = "ck_commerce_purchase_quantity"


@pytest.fixture
def quantity_migration():
    path = ROOT / "migrations/versions/b7f3d2c9a4e1_drop_dead_purchase_quantity_check.py"
    spec = importlib.util.spec_from_file_location("purchase_quantity_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def engine_at_parent(tmp_path, monkeypatch):
    """A live SQLite database migrated to the pre-change head (constraint on)."""

    url = f"sqlite:///{tmp_path / 'purchase-quantity.sqlite3'}"
    monkeypatch.setattr(settings, "DATABASE_URL", url)
    command.upgrade(Config(str(ROOT / "alembic.ini")), _PARENT)
    engine = create_engine(url)
    try:
        with engine.begin() as db:
            db.execute(
                text(
                    "INSERT INTO users (id,email,hashed_password,is_guest,is_active,is_admin) "
                    "VALUES (1,'quantity@migration.test','hash',0,1,0)"
                )
            )
            db.execute(
                text(
                    "INSERT INTO commerce_purchases "
                    "(id,owner_user_id,mode,merchant_id,currency,stage,version,generation,quantity) "
                    "VALUES (:id,1,'mock','demo-store','CNY','shopping',0,0,0)"
                ),
                {"id": str(uuid4())},
            )
        yield engine
    finally:
        engine.dispose()


def _check_names(db) -> set[str]:
    return {item["name"] for item in inspect(db).get_check_constraints("commerce_purchases")}


def test_upgrade_removes_dead_constraint_and_retains_rows(engine_at_parent):
    with engine_at_parent.connect() as db:
        assert _CHECK in _check_names(db)
        purchases_before = db.execute(
            text("SELECT id, stage, quantity FROM commerce_purchases")
        ).all()
    command.upgrade(Config(str(ROOT / "alembic.ini")), _HEAD)
    with engine_at_parent.connect() as db:
        assert _CHECK not in _check_names(db)
        assert "ck_commerce_purchase_versions" in _check_names(db)
        assert (
            db.execute(text("SELECT id, stage, quantity FROM commerce_purchases")).all()
            == purchases_before
        )
        assert db.execute(text("SELECT version_num FROM alembic_version")).scalar() == _HEAD
        # The dead column no longer rejects values the service never writes.
        db.execute(
            text(
                "INSERT INTO commerce_purchases "
                "(id,owner_user_id,mode,merchant_id,currency,stage,version,generation,quantity) "
                "VALUES (:id,1,'mock','demo-store','CNY','shopping',0,0,99)"
            ),
            {"id": str(uuid4())},
        )
        db.commit()
        assert db.execute(text("SELECT count(*) FROM commerce_purchases")).scalar() == 2


def test_downgrade_restores_constraint_and_replay_is_idempotent(engine_at_parent):
    config = Config(str(ROOT / "alembic.ini"))
    command.upgrade(config, _HEAD)
    with engine_at_parent.connect() as db:
        assert _CHECK not in _check_names(db)
    command.downgrade(config, _PARENT)
    with engine_at_parent.connect() as db:
        assert _CHECK in _check_names(db)
        with pytest.raises(IntegrityError):
            db.execute(
                text(
                    "INSERT INTO commerce_purchases "
                    "(id,owner_user_id,mode,merchant_id,currency,stage,version,generation,quantity) "
                    "VALUES (:id,1,'mock','demo-store','CNY','shopping',0,0,11)"
                ),
                {"id": str(uuid4())},
            )
            db.commit()
        db.rollback()
    command.upgrade(config, _HEAD)
    with engine_at_parent.connect() as db:
        assert _CHECK not in _check_names(db)


def test_replayed_module_upgrade_is_a_noop(quantity_migration, engine_at_parent, monkeypatch):
    command.upgrade(Config(str(ROOT / "alembic.ini")), _HEAD)
    with engine_at_parent.connect() as db:
        context = MigrationContext.configure(db)
        monkeypatch.setattr(quantity_migration, "op", Operations(context))
        quantity_migration.upgrade()  # constraint already absent: no-op, no error
        db.commit()
        assert _CHECK not in _check_names(db)


@pytest.mark.parametrize("direction", ["upgrade", "downgrade"])
def test_offline_postgresql_sql_drops_and_restores_the_check(
    quantity_migration, monkeypatch, direction
):
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output}
    )
    monkeypatch.setattr(quantity_migration, "op", Operations(context))
    getattr(quantity_migration, direction)()
    sql = output.getvalue()
    assert _CHECK in sql
    assert "commerce_purchases" in sql


def test_offline_sqlite_rebuild_is_refused_loudly(quantity_migration, monkeypatch):
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="sqlite", opts={"as_sql": True, "output_buffer": output}
    )
    monkeypatch.setattr(quantity_migration, "op", Operations(context))
    with pytest.raises(RuntimeError, match="requires PostgreSQL"):
        quantity_migration.upgrade()
    assert output.getvalue() == ""


def test_current_model_matches_migrated_schema(quantity_migration):
    assert re.fullmatch(r"[0-9a-f]{12}", _HEAD)
    assert str(PurchaseSession.__table__) == "commerce_purchases"
    engine = create_engine("sqlite:///:memory:")
    try:
        Base.metadata.create_all(engine)
        with engine.connect() as db:
            # The ORM no longer defines the dead constraint anywhere.
            assert _CHECK not in _check_names(db)
            assert "ck_commerce_purchase_versions" in _check_names(db)
    finally:
        engine.dispose()
