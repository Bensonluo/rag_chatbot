"""Visitor preference tables: upgrade creates them, downgrade drops them cleanly."""

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
        / "migrations/versions/b9d4e6f2a1c8_visitor_preferences.py"
    )
    spec = importlib.util.spec_from_file_location("visitor_preference_migration", path)
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


def test_upgrade_creates_tables_and_downgrade_drops_them(monkeypatch):
    migration = _load_migration()
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def fk(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    with engine.begin() as db:
        visitor_id = _seed_parents(db)
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(db)))
        migration.upgrade()
        assert migration.down_revision == "b7f3d2c9a4e1"

        tables = set(inspect(db).get_table_names())
        assert {"visitor_preferences", "visitor_preference_audit"}.issubset(tables)

        fks = inspect(db).get_foreign_keys("visitor_preferences")
        assert fks and fks[0]["referred_table"] == "guest_visitors"

        db.execute(
            text(
                "INSERT INTO visitor_preferences (id, visitor_id, pref_key, pref_value, source) "
                "VALUES (:id, :vid, 'display_preference', :value, 'confirmed')"
            ),
            {"id": str(uuid4()), "vid": visitor_id, "value": '{"size": "27英寸"}'},
        )
        db.execute(
            text(
                "INSERT INTO visitor_preference_audit (visitor_id, pref_key, action, old_value, new_value) "
                "VALUES (:vid, 'display_preference', 'save', NULL, :value)"
            ),
            {"vid": visitor_id, "value": '{"size": "27英寸"}'},
        )
        assert db.scalar(text("SELECT COUNT(*) FROM visitor_preferences")) == 1
        assert db.scalar(text("SELECT COUNT(*) FROM visitor_preference_audit")) == 1

        # Unique (visitor_id, pref_key): a second row for the same pair must fail.
        duplicate_failed = False
        try:
            db.execute(
                text(
                    "INSERT INTO visitor_preferences (id, visitor_id, pref_key, pref_value, source) "
                    "VALUES (:id, :vid, 'display_preference', 'null', 'inferred')"
                ),
                {"id": str(uuid4()), "vid": visitor_id},
            )
        except Exception:
            duplicate_failed = True
        assert duplicate_failed

        migration.downgrade()
        remaining = set(inspect(db).get_table_names())
        assert "visitor_preferences" not in remaining
        assert "visitor_preference_audit" not in remaining
        # Parent rows survive the downgrade untouched.
        assert db.scalar(text("SELECT COUNT(*) FROM guest_visitors")) == 1
    engine.dispose()


def test_expiry_column_roundtrips_through_the_migration_shape(monkeypatch):
    """expires_at stores timezone-aware ISO values usable by load_active's filter."""

    migration = _load_migration()
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as db:
        visitor_id = _seed_parents(db)
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(db)))
        migration.upgrade()
        expires = datetime(2026, 11, 11, 10, 0, 0, tzinfo=UTC)
        db.execute(
            text(
                "INSERT INTO visitor_preferences (id, visitor_id, pref_key, pref_value, source, expires_at) "
                "VALUES (:id, :vid, 'budget_hint', '{}', 'inferred', :expires)"
            ),
            {"id": str(uuid4()), "vid": visitor_id, "expires": expires.isoformat()},
        )
        stored = db.scalar(text("SELECT expires_at FROM visitor_preferences"))
        assert datetime.fromisoformat(stored) == expires
        assert datetime.fromisoformat(stored) > expires - timedelta(seconds=1)
    engine.dispose()
