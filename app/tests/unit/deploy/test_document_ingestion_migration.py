"""Durable ingestion must upgrade both versioned and legacy databases."""

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, Mock

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text

ROOT = Path(__file__).resolve().parents[4]


def load(path: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_additive_migration_preserves_legacy_documents_and_is_replayable(monkeypatch):
    migration = load(
        "migrations/versions/f9a2c7d6b410_document_ingestion_jobs.py", "document_migration"
    )
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE documents (id INTEGER PRIMARY KEY, external_doc_id VARCHAR(255) UNIQUE, title VARCHAR(500))"
            )
        )
        conn.execute(text("INSERT INTO documents VALUES (1, 'policy', 'Refunds')"))
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(conn)))
        migration.upgrade()
        migration.upgrade()  # An ORM-created legacy database can already have these additions.
        assert "document_versions" in inspect(conn).get_table_names()
        assert "active_version_id" in {c["name"] for c in inspect(conn).get_columns("documents")}
        assert conn.execute(text("SELECT title FROM documents")).scalar() == "Refunds"
        assert {"source_content", "normalized_text", "revision", "status", "pipeline_config"} <= {
            c["name"] for c in inspect(conn).get_columns("document_versions")
        }
    engine.dispose()


async def test_legacy_boot_applies_new_migration_after_adopting_baseline(monkeypatch):
    runner = load("scripts/init_db.py", "document_migration_runner")
    events = []
    monkeypatch.setattr(runner, "verify_database", AsyncMock(return_value=True))
    monkeypatch.setattr(runner, "_database_state", AsyncMock(return_value="legacy"))
    stamp = Mock(side_effect=lambda: events.append("adopt"))
    upgrade = Mock(side_effect=lambda: events.append("upgrade"))
    monkeypatch.setattr(runner, "_stamp_legacy_schema", stamp)
    monkeypatch.setattr(runner, "_run_migrations", upgrade)
    await runner.main()
    assert events == ["adopt", "upgrade"]


def test_legacy_stamp_does_not_mark_new_ingestion_tables_as_applied(monkeypatch):
    runner = load("scripts/init_db.py", "document_migration_stamp")
    stamp = Mock()
    monkeypatch.setattr(runner.command, "stamp", stamp)
    runner._stamp_legacy_schema()
    assert stamp.call_args.args[1] == "c5f2a8e3d17b"
