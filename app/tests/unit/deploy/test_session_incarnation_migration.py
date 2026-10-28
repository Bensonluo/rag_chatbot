"""Conversation incarnations survive adoption and prevent numeric-ID aliasing."""

import importlib.util
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.models.database.base import Base
from app.models.database.session import ChatSession
from app.models.database.user import User

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def incarnation_migration():
    path = ROOT / "migrations/versions/c53a7d9e8421_session_incarnation.py"
    spec = importlib.util.spec_from_file_location("session_incarnation_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def incarnation_db(tmp_path, monkeypatch, incarnation_migration):
    url = f"sqlite:///{tmp_path}/incarnation.sqlite3"
    monkeypatch.setattr(settings, "DATABASE_URL", url)
    command.upgrade(Config(str(ROOT / "alembic.ini")), "b42f9c8e7310")
    engine = create_engine(url)

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    with engine.begin() as db:
        db.execute(
            text(
                "INSERT INTO users (id,email,hashed_password,is_active,is_admin) VALUES (1,'legacy@incarnation.test','!',1,0),(2,'second@incarnation.test','!',1,0)"
            )
        )
        db.execute(
            text(
                "INSERT INTO chat_sessions (id,user_id,title,memory_type,context_window,session_metadata) VALUES (10,1,'Private preserved','sliding_window',10,:private),(20,2,'Public preserved','sliding_window',10,NULL)"
            ),
            {"private": '{"access_mode":"private"}'},
        )
        db.execute(
            text(
                "INSERT INTO messages (id,session_id,role,content,status) VALUES (1,10,'USER','Retained AI turn','COMPLETED')"
            )
        )
        db.execute(
            text(
                "INSERT INTO handoff_tickets (id,session_id,user_id,reason,summary,status) VALUES (1,10,1,'explicit','Retained ticket','open')"
            )
        )
        db.execute(
            text(
                "INSERT INTO handoff_messages (id,session_id,ticket_id,author_user_id,author_kind,client_message_id,content) VALUES (1,10,1,1,'customer','00000000-0000-4000-8000-000000000001','Retained human turn')"
            )
        )
        db.execute(
            text(
                "INSERT INTO action_executions (action_id,scope_session_id,live_session_id,user_id,scope,tool_name,arguments_hash,status) VALUES ('00000000-0000-4000-8000-000000000002',10,10,1,'10','process_refund',:hash,'in_progress')"
            ),
            {"hash": "a" * 64},
        )
    try:
        yield incarnation_migration, engine
    finally:
        engine.dispose()


@contextmanager
def migration_connection(module, engine, monkeypatch):
    with engine.connect() as db:
        context = MigrationContext.configure(db)
        monkeypatch.setattr(module, "op", Operations(context))
        # Match Alembic's per-revision transaction ownership; SQLite may
        # deliberately leave it for its FK-safe maintenance transaction.
        with context.begin_transaction(_per_migration=True):
            yield db


def _snapshot(db: Connection) -> dict[str, Any]:
    return {
        name: db.execute(text(f"SELECT {columns} FROM {name} ORDER BY id")).all()
        for name, columns in {
            "chat_sessions": "id,user_id,title,session_metadata,created_at,updated_at",
            "messages": "id,session_id,content",
            "handoff_tickets": "id,session_id,summary",
            "handoff_messages": "id,session_id,ticket_id,content",
        }.items()
    } | {"ledger_live": db.execute(text("SELECT live_session_id FROM action_executions")).scalar()}


def test_recreated_numeric_session_gets_a_different_server_generated_key():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    try:
        Base.metadata.create_all(engine)
        with Session(engine, expire_on_commit=False) as db:
            db.add_all(
                [
                    User(id=1, email="old@incarnation.test", hashed_password="!"),
                    User(id=2, email="new@incarnation.test", hashed_password="!"),
                ]
            )
            db.flush()
            original = ChatSession(user_id=1)
            db.add(original)
            db.commit()
            numeric_id, original_key = original.id, original.session_key
            assert str(UUID(original_key)) == original_key
            db.delete(original)
            db.commit()
            replacement = ChatSession(user_id=2)
            db.add(replacement)
            db.commit()
            assert replacement.id == numeric_id
            assert str(UUID(replacement.session_key)) == replacement.session_key
            assert replacement.session_key != original_key
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "value",
    ["not-a-uuid", "00000000000040008000000000000001", "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"],
)
def test_explicit_internal_keys_must_be_canonical_uuid_strings(value):
    with pytest.raises(ValueError, match="canonical UUID"):
        ChatSession(user_id=1, session_key=value)


def test_server_reset_can_rotate_an_existing_key_without_changing_numeric_scope():
    engine = create_engine("sqlite:///:memory:")
    try:
        Base.metadata.create_all(engine)
        with Session(engine, expire_on_commit=False) as db:
            db.add(User(id=1, email="reset@incarnation.test", hashed_password="!"))
            db.flush()
            chat = ChatSession(user_id=1)
            db.add(chat)
            db.commit()
            numeric_id, before = chat.id, chat.session_key
            chat.automation_epoch += 1
            db.commit()
            assert chat.session_key == before
            chat.session_key = str(uuid4())
            db.commit()
            assert chat.id == numeric_id and chat.session_key != before
    finally:
        engine.dispose()


def test_sqlite_upgrade_preserves_all_fk_children_and_backfills_independent_keys(
    incarnation_db, monkeypatch
):
    module, engine = incarnation_db
    with migration_connection(module, engine, monkeypatch) as db:
        before = _snapshot(db)
        module.upgrade()
        first = db.execute(text("SELECT id,session_key FROM chat_sessions ORDER BY id")).all()
        assert len({key for _, key in first}) == 2
        assert all(str(UUID(key)) == key for _, key in first)
        assert _snapshot(db) == before
        assert db.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert db.exec_driver_sql("PRAGMA foreign_key_check").all() == []
        module.upgrade()
        assert (
            db.execute(text("SELECT id,session_key FROM chat_sessions ORDER BY id")).all() == first
        )
        column = next(
            c for c in inspect(db).get_columns("chat_sessions") if c["name"] == "session_key"
        )
        assert column["nullable"] is False and column["default"] is None
        assert any(
            c["name"] == "uq_chat_sessions_session_key"
            for c in inspect(db).get_unique_constraints("chat_sessions")
        )
        assert "ix_chat_sessions_user_id" in {
            i["name"] for i in inspect(db).get_indexes("chat_sessions")
        }
        with pytest.raises(IntegrityError):
            db.execute(
                text("UPDATE chat_sessions SET session_key=:key WHERE id=20"), {"key": first[0][1]}
            )


def test_sqlite_failed_rebuild_rolls_back_parent_children_and_restores_fk(
    incarnation_db, monkeypatch
):
    module, engine = incarnation_db

    def refuse_rebuild(_conn, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().startswith("CREATE TABLE _alembic_tmp_chat_sessions"):
            raise RuntimeError("synthetic rebuild refusal")

    with migration_connection(module, engine, monkeypatch) as db:
        before = _snapshot(db)
        event.listen(engine, "before_cursor_execute", refuse_rebuild)
        try:
            with pytest.raises(RuntimeError, match="synthetic rebuild refusal"):
                module.upgrade()
        finally:
            event.remove(engine, "before_cursor_execute", refuse_rebuild)
        assert _snapshot(db) == before
        assert "session_key" not in {
            column["name"] for column in inspect(db).get_columns("chat_sessions")
        }
        assert db.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert db.exec_driver_sql("PRAGMA foreign_key_check").all() == []


def test_orm_adoption_retains_a_previously_issued_key(incarnation_migration, monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    try:
        Base.metadata.create_all(engine)
        key = str(uuid4())
        with Session(engine) as db:
            db.add(User(id=1, email="adopt@incarnation.test", hashed_password="!"))
            db.flush()
            db.add(ChatSession(id=10, user_id=1, session_key=key))
            db.commit()
        with migration_connection(incarnation_migration, engine, monkeypatch) as db:
            incarnation_migration.upgrade()
            assert db.execute(text("SELECT session_key FROM chat_sessions")).scalar() == key
    finally:
        engine.dispose()


@pytest.mark.parametrize("invalid", ["malformed", "duplicate"])
def test_suspect_existing_keys_are_not_silently_replaced(incarnation_db, monkeypatch, invalid):
    module, engine = incarnation_db
    key = "suspect-existing-key" if invalid == "malformed" else str(uuid4())
    with engine.begin() as db:
        db.execute(text("ALTER TABLE chat_sessions ADD COLUMN session_key VARCHAR(36)"))
        db.execute(text("UPDATE chat_sessions SET session_key=:key"), {"key": key})
    with migration_connection(module, engine, monkeypatch) as db:
        with pytest.raises(RuntimeError, match="explicit repair"):
            module.upgrade()
        assert db.execute(text("SELECT session_key FROM chat_sessions")).scalars().all() == [
            key,
            key,
        ]
        assert db.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert db.exec_driver_sql("PRAGMA foreign_key_check").all() == []


@pytest.mark.parametrize("retained", ["private", "human"])
def test_private_or_human_rows_prevent_downgrade_and_restore_fk(
    incarnation_db, monkeypatch, retained
):
    module, engine = incarnation_db
    with migration_connection(module, engine, monkeypatch) as db:
        module.upgrade()
        keys = db.execute(text("SELECT session_key FROM chat_sessions ORDER BY id")).scalars().all()
        if retained == "private":
            db.execute(text("DELETE FROM handoff_messages"))
        else:
            db.execute(text("UPDATE chat_sessions SET session_metadata=NULL"))
        with pytest.raises(RuntimeError, match="prevents session identity downgrade"):
            module.downgrade()
        assert (
            db.execute(text("SELECT session_key FROM chat_sessions ORDER BY id")).scalars().all()
            == keys
        )
        assert db.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1


def test_public_only_downgrade_keeps_children_indexes_and_reenables_fk(incarnation_db, monkeypatch):
    module, engine = incarnation_db
    with migration_connection(module, engine, monkeypatch) as db:
        module.upgrade()
        db.execute(text("UPDATE chat_sessions SET session_metadata=NULL"))
        db.execute(text("DELETE FROM handoff_messages"))
        before = _snapshot(db)
        module.downgrade()
        assert _snapshot(db) == before
        assert "session_key" not in {c["name"] for c in inspect(db).get_columns("chat_sessions")}
        assert "ix_chat_sessions_user_id" in {
            i["name"] for i in inspect(db).get_indexes("chat_sessions")
        }
        assert db.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert db.exec_driver_sql("PRAGMA foreign_key_check").all() == []


def test_failed_downgrade_rebuild_keeps_key_children_and_fk_enabled(incarnation_db, monkeypatch):
    module, engine = incarnation_db

    def refuse_rebuild(_conn, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().startswith("CREATE TABLE _alembic_tmp_chat_sessions"):
            raise RuntimeError("synthetic downgrade refusal")

    with migration_connection(module, engine, monkeypatch) as db:
        module.upgrade()
        db.execute(text("UPDATE chat_sessions SET session_metadata=NULL"))
        db.execute(text("DELETE FROM handoff_messages"))
        before = _snapshot(db)
        keys = db.execute(text("SELECT session_key FROM chat_sessions ORDER BY id")).scalars().all()
        event.listen(engine, "before_cursor_execute", refuse_rebuild)
        try:
            with pytest.raises(RuntimeError, match="synthetic downgrade refusal"):
                module.downgrade()
        finally:
            event.remove(engine, "before_cursor_execute", refuse_rebuild)
        assert _snapshot(db) == before
        assert (
            db.execute(text("SELECT session_key FROM chat_sessions ORDER BY id")).scalars().all()
            == keys
        )
        assert db.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert db.exec_driver_sql("PRAGMA foreign_key_check").all() == []


@pytest.mark.parametrize("dialect", ["postgresql", "sqlite"])
def test_offline_upgrade_uses_per_row_keys_and_safe_downgrade_policy(
    incarnation_migration, monkeypatch, dialect
):
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect, opts={"as_sql": True, "output_buffer": output}
    )
    monkeypatch.setattr(incarnation_migration, "op", Operations(context))
    incarnation_migration.upgrade()
    sql = output.getvalue()
    assert "session_key" in sql and "UNIQUE (session_key)" in sql
    assert "NOT NULL" in sql
    if dialect == "postgresql":
        assert "gen_random_uuid()::text" in sql
    else:
        assert "randomblob" in sql
        assert sql.index("PRAGMA foreign_key_check") < sql.index("COMMIT")
        assert (
            sql.index("PRAGMA foreign_keys=OFF")
            < sql.index("DROP TABLE chat_sessions")
            < sql.index("PRAGMA foreign_keys=ON")
        )
    output.truncate(0)
    output.seek(0)
    if dialect == "sqlite":
        with pytest.raises(RuntimeError, match="requires online data checks"):
            incarnation_migration.downgrade()
        assert "DROP" not in output.getvalue()
    else:
        incarnation_migration.downgrade()
        sql = output.getvalue()
        assert (
            sql.index("LOCK TABLE")
            < sql.index("EXISTS (SELECT 1 FROM chat_sessions")
            < sql.index("DROP COLUMN session_key")
        )


def test_generated_offline_sqlite_upgrade_executes_without_cascading_children(
    incarnation_db, monkeypatch
):
    module, engine = incarnation_db
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="sqlite", opts={"as_sql": True, "output_buffer": output}
    )
    monkeypatch.setattr(module, "op", Operations(context))
    module.upgrade()
    with engine.connect() as db:
        before = _snapshot(db)
    raw = engine.raw_connection()
    try:
        raw.driver_connection.executescript(output.getvalue())
        assert raw.driver_connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        raw.close()
    with engine.connect() as db:
        assert _snapshot(db) == before
        keys = db.execute(text("SELECT session_key FROM chat_sessions")).scalars().all()
        assert len(set(keys)) == 2 and all(str(UUID(key)) == key for key in keys)
        assert db.exec_driver_sql("PRAGMA foreign_key_check").all() == []
