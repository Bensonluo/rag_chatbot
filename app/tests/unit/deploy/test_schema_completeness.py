"""Versioned databases must contain every runtime model and survive rollback."""

import importlib.util
import subprocess
import sys
from io import StringIO
from pathlib import Path
from types import ModuleType

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import MetaData, create_engine, inspect, text

from app.models.database.base import Base
from app.models.database.knowledge_gap_resolution import KnowledgeGapResolution
from app.models.database.user_fact import UserFactRecord

ROOT = Path(__file__).resolve().parents[4]
MIGRATION = "migrations/versions/d4c9f1a82e70_complete_support_schema.py"


def load_migration(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("schema_completeness_migration", ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fresh_environment_registers_all_models_and_sqlite_round_trip(tmp_path: Path) -> None:
    # A fresh interpreter prevents pytest's application imports from hiding a
    # missing env.py model import (which can propose dropping a live table).
    script = """
import importlib
import pkgutil
import sys
from alembic import command
from alembic.config import Config
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect
from app.config.settings import settings
from app.models.database.base import Base
import app.models.database

settings.DATABASE_URL = sys.argv[1]
config = Config('alembic.ini')
command.upgrade(config, 'head')
registered = set(Base.metadata.tables)
for module in pkgutil.iter_modules(app.models.database.__path__):
    importlib.import_module(f'app.models.database.{module.name}')
assert registered == set(Base.metadata.tables), (registered, set(Base.metadata.tables))
engine = create_engine(settings.DATABASE_URL)
try:
    with engine.connect() as conn:
        assert compare_metadata(MigrationContext.configure(conn, opts={"compare_server_default": True}), Base.metadata) == []
    command.downgrade(config, 'base')
    with engine.connect() as conn:
        assert set(inspect(conn).get_table_names()) <= {'alembic_version'}
    command.upgrade(config, 'head')
    with engine.connect() as conn:
        assert compare_metadata(MigrationContext.configure(conn, opts={"compare_server_default": True}), Base.metadata) == []
finally:
    engine.dispose()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, f"sqlite:///{tmp_path / 'migrations.db'}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("missing_priority_default", [False, True])
def test_adopts_orm_tables_preserving_rows_and_repairs_missing_indexes(
    monkeypatch, missing_priority_default
) -> None:
    migration = load_migration(MIGRATION)
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as conn:
            legacy_metadata = MetaData()
            for table in Base.metadata.sorted_tables:
                table.to_metadata(legacy_metadata)
            if missing_priority_default:
                legacy_metadata.tables["handoff_tickets"].c.priority.server_default = None
            legacy_metadata.create_all(conn)
            conn.execute(
                legacy_metadata.tables["users"]
                .insert()
                .values(id=1, email="migration@example.test", hashed_password="test-only")
            )
            conn.execute(legacy_metadata.tables["chat_sessions"].insert().values(id=1, user_id=1))
            conn.execute(
                legacy_metadata.tables["handoff_tickets"]
                .insert()
                .values(session_id=1, priority=1, reason="emotion", summary="Retain this request")
            )
            conn.execute(
                Base.metadata.tables[UserFactRecord.__tablename__]
                .insert()
                .values(
                    user_id=7, fact="Prefers email", category="preference", source_session_id="12"
                )
            )
            conn.execute(
                Base.metadata.tables[KnowledgeGapResolution.__tablename__]
                .insert()
                .values(normalized_query="refund policy", resolved_by=99)
            )
            conn.execute(text("DROP INDEX ix_user_facts_user_id"))
            conn.execute(text("DROP INDEX ix_knowledge_gap_resolutions_normalized_query"))
            before_facts = conn.execute(text("SELECT * FROM user_facts")).all()
            before_tickets = conn.execute(text("SELECT * FROM handoff_tickets")).all()
            before_resolutions = conn.execute(text("SELECT * FROM knowledge_gap_resolutions")).all()
            monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(conn)))

            migration.upgrade()
            migration.upgrade()

            assert conn.execute(text("SELECT * FROM user_facts")).all() == before_facts
            assert conn.execute(text("SELECT * FROM handoff_tickets")).all() == before_tickets
            assert (
                conn.execute(text("SELECT * FROM knowledge_gap_resolutions")).all()
                == before_resolutions
            )
            assert (
                compare_metadata(
                    MigrationContext.configure(conn, opts={"compare_server_default": True}),
                    Base.metadata,
                )
                == []
            )
            assert any(
                index["unique"] for index in inspect(conn).get_indexes("knowledge_gap_resolutions")
            )
    finally:
        engine.dispose()


@pytest.mark.parametrize("dialect", ["postgresql", "sqlite"])
def test_offline_schema_sql_and_enum_cleanup_are_dialect_compatible(monkeypatch, dialect) -> None:
    migration = load_migration(MIGRATION)
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect, opts={"as_sql": True, "output_buffer": output}
    )
    monkeypatch.setattr(migration, "op", Operations(context))
    migration.upgrade()
    migration.downgrade()
    sql = output.getvalue()
    assert "CREATE TABLE user_facts" in sql
    assert "CREATE TABLE knowledge_gap_resolutions" in sql
    assert "CREATE UNIQUE INDEX ix_knowledge_gap_resolutions_normalized_query" in sql
    assert "DROP TABLE knowledge_gap_resolutions" in sql
    assert ("COMMENT ON COLUMN chat_sessions.resolved_at" in sql) == (dialect == "postgresql")

    initial = load_migration("migrations/versions/ad9a30054d1c_initial_schema.py")
    monkeypatch.setattr(initial, "op", Operations(context))
    initial.downgrade()
    sql = output.getvalue()
    if dialect == "postgresql":
        assert sql.index("DROP TABLE messages") < sql.index("DROP TYPE IF EXISTS messagerole")
        assert "DROP TYPE IF EXISTS messagerole RESTRICT" in sql
        assert "DROP TYPE IF EXISTS messagestatus RESTRICT" in sql
        assert "CASCADE" not in sql
    else:
        assert "DROP TYPE" not in sql
