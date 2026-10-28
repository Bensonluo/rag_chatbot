"""Human-channel migrations adopt existing schemas and refuse lossy rollback."""

import importlib.util
from io import StringIO
from pathlib import Path
from types import ModuleType

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from app.models.database.base import Base


@pytest.fixture
def human_migration() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[4]
        / "migrations/versions/a31e8d7f6290_human_conversation_channel.py"
    )
    spec = importlib.util.spec_from_file_location("human_channel_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_upgrade_replay_and_empty_downgrade_preserve_rows(human_migration, monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as db:
            db.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
            db.execute(
                text(
                    "CREATE TABLE chat_sessions (id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT)"
                )
            )
            db.execute(
                text("CREATE TABLE handoff_tickets (id INTEGER PRIMARY KEY, session_id INTEGER)")
            )
            db.execute(text("INSERT INTO users VALUES (1)"))
            db.execute(text("INSERT INTO chat_sessions VALUES (10,1,'Retained conversation')"))
            monkeypatch.setattr(human_migration, "op", Operations(MigrationContext.configure(db)))
            human_migration.upgrade()
            human_migration.upgrade()
            assert db.execute(text("SELECT automation_epoch FROM chat_sessions")).scalar() == 0
            assert "handoff_messages" in inspect(db).get_table_names()
            human_migration.downgrade()
            assert (
                db.execute(text("SELECT title FROM chat_sessions")).scalar()
                == "Retained conversation"
            )
            assert "automation_epoch" not in {
                column["name"] for column in inspect(db).get_columns("chat_sessions")
            }
    finally:
        engine.dispose()


def test_orm_adoption_retains_all_constraints_and_human_data(human_migration, monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as db:
            Base.metadata.create_all(db)
            monkeypatch.setattr(human_migration, "op", Operations(MigrationContext.configure(db)))
            human_migration.upgrade()
            db.execute(
                text(
                    "INSERT INTO users (id,email,hashed_password,is_active,is_admin) VALUES (1,'migration@unit.invalid','!',1,0)"
                )
            )
            db.execute(
                text(
                    "INSERT INTO chat_sessions (id,user_id,memory_type,context_window,title,reopened_count,automation_epoch) VALUES (10,1,'sliding_window',10,'Retained',0,0)"
                )
            )
            db.execute(
                text(
                    "INSERT INTO handoff_tickets (id,session_id,user_id,reason,summary,status) VALUES (1,10,1,'explicit','{}','open')"
                )
            )
            insert = text(
                "INSERT INTO handoff_messages (session_id,ticket_id,author_user_id,author_kind,client_message_id,content) VALUES (10,1,1,'customer','00000000-0000-0000-0000-000000000001','Retained human text')"
            )
            db.execute(insert)
            assert db.execute(text("SELECT resolves_ticket FROM handoff_messages")).scalar() == 0
            with pytest.raises(IntegrityError):
                db.execute(insert)
            with pytest.raises(RuntimeError, match="prevents lossless downgrade"):
                human_migration.downgrade()
            assert (
                db.execute(text("SELECT content FROM handoff_messages")).scalar()
                == "Retained human text"
            )
            assert "automation_epoch" in {
                column["name"] for column in inspect(db).get_columns("chat_sessions")
            }
    finally:
        engine.dispose()


def test_nonzero_generation_alone_prevents_destructive_downgrade(human_migration, monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as db:
            Base.metadata.create_all(db)
            db.execute(
                text(
                    "INSERT INTO chat_sessions (id,user_id,memory_type,context_window,title,reopened_count,automation_epoch) VALUES (10,1,'sliding_window',10,'Retained',0,2)"
                )
            )
            monkeypatch.setattr(human_migration, "op", Operations(MigrationContext.configure(db)))
            with pytest.raises(RuntimeError, match="prevents lossless downgrade"):
                human_migration.downgrade()
            assert db.execute(text("SELECT automation_epoch FROM chat_sessions")).scalar() == 2
    finally:
        engine.dispose()


def test_private_session_alone_prevents_rollback_to_public_demo(human_migration, monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as db:
            Base.metadata.create_all(db)
            db.execute(
                text(
                    "INSERT INTO chat_sessions (id,user_id,memory_type,context_window,title,reopened_count,automation_epoch,session_metadata) VALUES (10,1,'sliding_window',10,'Private',0,0,:metadata)"
                ),
                {"metadata": '{"access_mode":"private"}'},
            )
            monkeypatch.setattr(human_migration, "op", Operations(MigrationContext.configure(db)))
            with pytest.raises(RuntimeError, match="prevents lossless downgrade"):
                human_migration.downgrade()
            assert "handoff_messages" in inspect(db).get_table_names()
            assert db.execute(text("SELECT title FROM chat_sessions")).scalar() == "Private"
    finally:
        engine.dispose()


@pytest.mark.parametrize("dialect", ["postgresql", "sqlite"])
def test_offline_sql_has_foreign_keys_idempotency_and_guard_before_drop(
    human_migration, monkeypatch, dialect
):
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect, opts={"as_sql": True, "output_buffer": output}
    )
    monkeypatch.setattr(human_migration, "op", Operations(context))
    human_migration.upgrade()
    sql = output.getvalue()
    assert "automation_epoch" in sql and "DEFAULT" in sql
    assert "UNIQUE (session_id, author_user_id, client_message_id)" in sql
    assert "REFERENCES chat_sessions (id) ON DELETE CASCADE" in sql
    assert "REFERENCES handoff_tickets (id) ON DELETE CASCADE" in sql
    output.truncate(0)
    output.seek(0)
    if dialect == "sqlite":
        with pytest.raises(RuntimeError, match="requires online data checks"):
            human_migration.downgrade()
        assert "DROP" not in output.getvalue()
        return
    human_migration.downgrade()
    sql = output.getvalue()
    assert sql.index("EXISTS (SELECT 1 FROM handoff_messages)") < sql.index(
        "DROP TABLE handoff_messages"
    )
    assert "automation_epoch > 0" in sql
    assert "session_metadata ->> 'access_mode' = 'private'" in sql
    assert "DELETE FROM" not in sql
