"""Commerce price-watch table: upgrade creates it, downgrade drops it cleanly."""

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, event, inspect, text


def _load_migration() -> Any:
    path = (
        Path(__file__).resolve().parents[4]
        / "migrations/versions/c5f8a3e1b7d9_commerce_price_watches.py"
    )
    spec = importlib.util.spec_from_file_location("price_watch_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_parents(db) -> str:
    """users + guest_visitors rows so the FK has a live target."""

    db.execute(
        text(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, email VARCHAR(255), "
            "hashed_password VARCHAR(255), is_active BOOLEAN, is_admin BOOLEAN, "
            "created_at DATETIME DEFAULT CURRENT_TIMESTAMP, "
            "updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
    )
    db.execute(text("INSERT INTO users (id) VALUES (1)"))
    db.execute(
        text(
            "CREATE TABLE guest_visitors (id VARCHAR(36) PRIMARY KEY, user_id INTEGER, "
            "token_hash VARCHAR(64), expires_at DATETIME, current_session_id VARCHAR(36), "
            "created_at DATETIME DEFAULT CURRENT_TIMESTAMP, "
            "updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
        )
    )
    visitor_id = str(uuid4())
    db.execute(
        text(
            "INSERT INTO guest_visitors (id, user_id, token_hash, expires_at) "
            "VALUES (:id, 1, :hash, :expires)"
        ),
        {
            "id": visitor_id,
            "hash": "a" * 64,
            "expires": datetime(2026, 12, 1, tzinfo=UTC).isoformat(),
        },
    )
    return visitor_id


def test_upgrade_creates_watch_table_and_downgrade_drops_it(monkeypatch):
    migration = _load_migration()
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def fk(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    with engine.begin() as db:
        visitor_id = _seed_parents(db)
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(db)))
        migration.upgrade()
        assert migration.down_revision == "b9d4e6f2a1c8"

        assert "commerce_price_watches" in inspect(db).get_table_names()
        fks = inspect(db).get_foreign_keys("commerce_price_watches")
        assert fks and fks[0]["referred_table"] == "guest_visitors"

        db.execute(
            text(
                "INSERT INTO commerce_price_watches "
                "(id, visitor_id, sku_id, sku_name, target_price_minor, condition, "
                "status, expires_at) "
                "VALUES (:id, :vid, 'ANC-02', '轻音降噪耳机 · 白色', 25000, 'at_or_below', "
                "'active', :expires)"
            ),
            {
                "id": str(uuid4()),
                "vid": visitor_id,
                "expires": datetime(2026, 12, 1, tzinfo=UTC).isoformat(),
            },
        )
        assert db.scalar(text("SELECT COUNT(*) FROM commerce_price_watches")) == 1
        status = db.scalar(text("SELECT status FROM commerce_price_watches"))
        assert status == "active"
        # The DB-level status CHECK is the real guarantee; defaults live in
        # the ORM/service layer, which always writes status explicitly.
        invalid_status_rejected = False
        try:
            db.execute(
                text(
                    "INSERT INTO commerce_price_watches "
                    "(id, visitor_id, sku_id, sku_name, target_price_minor, condition, "
                    "status, expires_at) "
                    "VALUES (:id, :vid, 'KEY-02', 'k', 15000, 'at_or_below', 'bogus', :expires)"
                ),
                {
                    "id": str(uuid4()),
                    "vid": visitor_id,
                    "expires": datetime(2026, 12, 1, tzinfo=UTC).isoformat(),
                },
            )
        except Exception:
            invalid_status_rejected = True
        assert invalid_status_rejected

        # Unique (visitor_id, sku_id): one rule per visitor and SKU.
        duplicate_failed = False
        try:
            db.execute(
                text(
                    "INSERT INTO commerce_price_watches "
                    "(id, visitor_id, sku_id, sku_name, target_price_minor, expires_at) "
                    "VALUES (:id, :vid, 'ANC-02', 'dup', 1, :expires)"
                ),
                {
                    "id": str(uuid4()),
                    "vid": visitor_id,
                    "expires": datetime(2026, 12, 1, tzinfo=UTC).isoformat(),
                },
            )
        except Exception:
            duplicate_failed = True
        assert duplicate_failed

        migration.downgrade()
        assert "commerce_price_watches" not in inspect(db).get_table_names()
        assert db.scalar(text("SELECT COUNT(*) FROM guest_visitors")) == 1
    engine.dispose()


def test_triggered_columns_roundtrip_through_the_migration_shape(monkeypatch):
    """triggered_at / triggered_price_minor store the trigger receipt."""

    migration = _load_migration()
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as db:
        visitor_id = _seed_parents(db)
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(db)))
        migration.upgrade()
        triggered_at = datetime(2026, 11, 11, 10, 0, 0, tzinfo=UTC)
        db.execute(
            text(
                "INSERT INTO commerce_price_watches "
                "(id, visitor_id, sku_id, sku_name, target_price_minor, condition, "
                "status, triggered_at, triggered_price_minor, expires_at) "
                "VALUES (:id, :vid, 'ANC-02', '轻音降噪耳机 · 白色', 30000, 'at_or_below', "
                "'triggered', :triggered_at, 28000, :expires)"
            ),
            {
                "id": str(uuid4()),
                "vid": visitor_id,
                "triggered_at": triggered_at.isoformat(),
                "expires": triggered_at + timedelta(days=30),
            },
        )
        stored = db.scalar(text("SELECT triggered_at FROM commerce_price_watches"))
        assert datetime.fromisoformat(stored) == triggered_at
        assert db.scalar(text("SELECT triggered_price_minor FROM commerce_price_watches")) == 28000
    engine.dispose()
