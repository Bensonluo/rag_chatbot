"""Execution history cannot disappear during conversation deletion or rollback."""

import importlib.util
from io import StringIO
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError, OperationalError

from app.models.database.action_execution import ActionExecution
from app.models.database.base import Base
from app.models.database.session import ChatSession  # noqa: F401
from app.models.database.user import User  # noqa: F401


@pytest.fixture
def action_migration():
    path = (
        Path(__file__).resolve().parents[4]
        / "migrations/versions/b42f9c8e7310_action_execution_ledger.py"
    )
    spec = importlib.util.spec_from_file_location("action_ledger_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _insert(db: Connection, *, status: str = "in_progress", action_id: str | None = None) -> str:
    action_id = action_id or str(uuid4())
    db.execute(
        text(
            "INSERT INTO action_executions (action_id,scope_session_id,live_session_id,user_id,scope,tool_name,arguments_hash,status,result) VALUES (:action,10,10,0,'10:automation:1','process_refund',:hash,:status,:result)"
        ),
        {
            "action": action_id,
            "hash": "a" * 64,
            "status": status,
            "result": '{"success":true,"data":{"refund_id":"retained"},"message":"OK"}'
            if status == "completed"
            else None,
        },
    )
    return action_id


def test_upgrade_replay_and_empty_downgrade_preserve_prior_session(action_migration, monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as db:
            db.execute(text("CREATE TABLE chat_sessions (id INTEGER PRIMARY KEY, title TEXT)"))
            db.execute(text("INSERT INTO chat_sessions VALUES (10,'Retained session')"))
            monkeypatch.setattr(action_migration, "op", Operations(MigrationContext.configure(db)))
            action_migration.upgrade()
            action_migration.upgrade()
            assert "action_executions" in inspect(db).get_table_names()
            assert action_migration.down_revision == "a31e8d7f6290"
            action_migration.downgrade()
            assert "action_executions" not in inspect(db).get_table_names()
            assert (
                db.execute(text("SELECT title FROM chat_sessions")).scalar() == "Retained session"
            )
    finally:
        engine.dispose()


@pytest.mark.parametrize("status", ["in_progress", "completed", "unknown"])
def test_every_retained_execution_prevents_destructive_downgrade(
    action_migration, monkeypatch, status
):
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as db:
            db.execute(text("CREATE TABLE chat_sessions (id INTEGER PRIMARY KEY)"))
            db.execute(text("INSERT INTO chat_sessions VALUES (10)"))
            monkeypatch.setattr(action_migration, "op", Operations(MigrationContext.configure(db)))
            action_migration.upgrade()
            action_id = _insert(db, status=status)
            with pytest.raises(RuntimeError, match="prevents lossless downgrade"):
                action_migration.downgrade()
            row = db.execute(text("SELECT action_id,status FROM action_executions")).one()
            assert row == (action_id, status)
    finally:
        engine.dispose()


def test_create_all_adoption_retains_primary_key_status_and_nullable_live_fk(
    action_migration, monkeypatch
):
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    try:
        with engine.begin() as db:
            Base.metadata.create_all(db)
            db.execute(
                text(
                    "INSERT INTO users (id,email,hashed_password,is_active,is_admin) VALUES (1,'action-migration@unit.invalid','!',1,0)"
                )
            )
            db.execute(
                text(
                    "INSERT INTO chat_sessions (id,user_id,title,memory_type,context_window,reopened_count,automation_epoch,session_key) VALUES (10,1,'Kept', 'sliding_window',10,0,0,:session_key)"
                ),
                {"session_key": str(uuid4())},
            )
            monkeypatch.setattr(action_migration, "op", Operations(MigrationContext.configure(db)))
            action_migration.upgrade()
            action_id = _insert(db, status="completed")
            with pytest.raises(IntegrityError):
                _insert(db, status="completed", action_id=action_id)
            with pytest.raises(IntegrityError):
                _insert(db, status="invalid")
            db.execute(text("DELETE FROM chat_sessions WHERE id=10"))
            row = db.execute(
                text("SELECT scope_session_id,live_session_id,result FROM action_executions")
            ).one()
            assert row[0] == 10 and row[1] is None and "retained" in row[2]
            assert "arguments_hash" in ActionExecution.__table__.columns
            assert "arguments" not in ActionExecution.__table__.columns
            with pytest.raises(RuntimeError, match="prevents lossless downgrade"):
                action_migration.downgrade()
    finally:
        engine.dispose()


@pytest.mark.parametrize("dialect", ["postgresql", "sqlite"])
def test_offline_upgrade_and_lossless_downgrade_guard(action_migration, monkeypatch, dialect):
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect, opts={"as_sql": True, "output_buffer": output}
    )
    monkeypatch.setattr(action_migration, "op", Operations(context))
    action_migration.upgrade()
    sql = output.getvalue()
    assert "PRIMARY KEY (action_id)" in sql
    assert "REFERENCES chat_sessions (id) ON DELETE SET NULL" in sql
    assert "'in_progress', 'completed', 'unknown'" in sql
    output.truncate(0)
    output.seek(0)
    if dialect == "sqlite":
        with pytest.raises(RuntimeError, match="requires online data checks"):
            action_migration.downgrade()
        assert "DROP" not in output.getvalue()
    else:
        action_migration.downgrade()
        sql = output.getvalue()
        assert (
            sql.index("LOCK TABLE action_executions")
            < sql.index("EXISTS (SELECT 1 FROM action_executions)")
            < sql.index("DROP TABLE action_executions")
        )
        assert "DELETE FROM" not in sql


def test_sqlite_downgrade_write_gate_blocks_a_racing_reservation(
    action_migration, monkeypatch, tmp_path
):
    engine = create_engine(f"sqlite:///{tmp_path}/rollback.sqlite3", connect_args={"timeout": 0.02})
    try:
        with engine.begin() as db:
            db.execute(text("CREATE TABLE chat_sessions (id INTEGER PRIMARY KEY)"))
            db.execute(text("INSERT INTO chat_sessions VALUES (10)"))
            monkeypatch.setattr(action_migration, "op", Operations(MigrationContext.configure(db)))
            action_migration.upgrade()
        raced = []

        def race(_conn, _cursor, statement, _parameters, _context, _many):
            if statement == "SELECT 1 FROM action_executions LIMIT 1":
                with engine.begin() as writer, pytest.raises(OperationalError, match="locked"):
                    _insert(writer)
                raced.append(True)

        event.listen(engine, "before_cursor_execute", race)
        with engine.begin() as db:
            monkeypatch.setattr(action_migration, "op", Operations(MigrationContext.configure(db)))
            action_migration.downgrade()
        assert raced == [True]
    finally:
        engine.dispose()
