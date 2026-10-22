"""Active handoff uniqueness upgrades preserve all customer requests."""

import importlib.util
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from types import ModuleType

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Connection, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from app.models.database.base import Base
from app.models.database.ticket import HandoffTicket

INDEX_NAME = "uq_handoff_tickets_active_session"


@pytest.fixture
def migration() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[4]
        / "migrations/versions/8b6e2a941fd0_unique_active_handoff.py"
    )
    spec = importlib.util.spec_from_file_location("handoff_uniqueness_migration", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def connection(migration: ModuleType, monkeypatch: pytest.MonkeyPatch) -> Iterator[Connection]:
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE handoff_tickets (id INTEGER PRIMARY KEY, "
                    "session_id INTEGER NOT NULL, status TEXT NOT NULL, summary TEXT NOT NULL)"
                )
            )
            monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(conn)))
            yield conn
    finally:
        engine.dispose()


def _insert(connection: Connection, session_id: int, status: str) -> None:
    connection.execute(
        text(
            "INSERT INTO handoff_tickets (session_id, status, summary) "
            "VALUES (:session_id, :status, 'Customer request must be retained')"
        ),
        {"session_id": session_id, "status": status},
    )


@pytest.mark.parametrize(
    "statuses", [("open", "open"), ("open", "claimed"), ("claimed", "claimed")]
)
def test_duplicates_stop_upgrade_without_modifying_tickets(connection, migration, statuses):
    for status in (*statuses, "resolved"):
        _insert(connection, 17, status)
    before = connection.execute(text("SELECT * FROM handoff_tickets ORDER BY id")).all()

    with pytest.raises(RuntimeError, match=r"duplicate open/claimed.*17 \(2 active\)") as exc:
        migration.upgrade()

    assert "no ticket data has been changed" in str(exc.value)
    assert connection.execute(text("SELECT * FROM handoff_tickets ORDER BY id")).all() == before
    assert INDEX_NAME not in {
        index["name"] for index in inspect(connection).get_indexes("handoff_tickets")
    }


@pytest.mark.parametrize("first", ["open", "claimed"])
@pytest.mark.parametrize("second", ["open", "claimed"])
def test_unique_index_rejects_every_active_status_pair(connection, migration, first, second):
    migration.upgrade()
    _insert(connection, 17, first)
    _insert(connection, 18, second)

    with pytest.raises(IntegrityError):
        _insert(connection, 17, second)


def test_history_and_claim_transition_remain_valid(connection, migration):
    _insert(connection, 17, "resolved")
    _insert(connection, 17, "resolved")
    _insert(connection, 17, "open")
    migration.upgrade()

    connection.execute(text("UPDATE handoff_tickets SET status = 'claimed' WHERE status = 'open'"))
    with pytest.raises(IntegrityError):
        connection.execute(text("UPDATE handoff_tickets SET status = 'open' WHERE id = 1"))
    connection.execute(
        text("UPDATE handoff_tickets SET status = 'resolved' WHERE status = 'claimed'")
    )
    _insert(connection, 17, "open")
    assert connection.execute(text("SELECT COUNT(*) FROM handoff_tickets")).scalar_one() == 4


def test_upgrade_replay_and_downgrade_preserve_rows(connection, migration):
    _insert(connection, 17, "resolved")
    _insert(connection, 17, "claimed")
    before = connection.execute(text("SELECT * FROM handoff_tickets ORDER BY id")).all()
    migration.upgrade()
    migration.upgrade()
    indexes = inspect(connection).get_indexes("handoff_tickets")
    index = next(index for index in indexes if index["name"] == INDEX_NAME)
    assert index["unique"]
    assert index["column_names"] == ["session_id"]

    migration.downgrade()
    assert INDEX_NAME not in {
        index["name"] for index in inspect(connection).get_indexes("handoff_tickets")
    }
    assert connection.execute(text("SELECT * FROM handoff_tickets ORDER BY id")).all() == before
    migration.upgrade()
    assert connection.execute(text("SELECT * FROM handoff_tickets ORDER BY id")).all() == before
    migration.downgrade()
    _insert(connection, 17, "open")


def test_adopts_index_already_created_by_orm(migration, monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as connection:
            Base.metadata.create_all(connection)
            table = Base.metadata.tables[HandoffTicket.__tablename__]
            connection.execute(table.insert().values(session_id=17, reason="explicit"))
            before = inspect(connection).get_indexes("handoff_tickets")
            assert INDEX_NAME in {index["name"] for index in before}
            monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))

            migration.upgrade()

            assert {
                index["name"] for index in inspect(connection).get_indexes("handoff_tickets")
            } == {index["name"] for index in before}
            assert (
                connection.execute(text("SELECT COUNT(*) FROM handoff_tickets")).scalar_one() == 1
            )
            with pytest.raises(IntegrityError):
                connection.execute(table.insert().values(session_id=17, reason="explicit"))
    finally:
        engine.dispose()


@pytest.mark.parametrize("dialect", ["postgresql", "sqlite"])
def test_offline_sql_has_unique_partial_index_and_nondestructive_downgrade(
    migration, monkeypatch, dialect
):
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect, opts={"as_sql": True, "output_buffer": output}
    )
    monkeypatch.setattr(migration, "op", Operations(context))

    migration.upgrade()
    migration.downgrade()

    sql = output.getvalue()
    assert f"CREATE UNIQUE INDEX {INDEX_NAME} ON handoff_tickets (session_id)" in sql
    assert "WHERE status IN ('open', 'claimed')" in sql
    assert f"DROP INDEX {INDEX_NAME}" in sql
    assert "DELETE" not in sql
    assert "UPDATE" not in sql
    assert "DROP TABLE" not in sql
