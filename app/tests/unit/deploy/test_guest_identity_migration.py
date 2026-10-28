"""User-parent maintenance preserves registered data and all FK children."""

import importlib.util
from contextlib import contextmanager
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.models.database.base import Base
from app.models.database.guest import GuestVisitor
from app.models.database.session import ChatSession
from app.models.database.user import User

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def guest_migration():
    path = ROOT / "migrations/versions/d64b8c0f9532_guest_identity.py"
    spec = importlib.util.spec_from_file_location("guest_identity_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def guest_migration_db(tmp_path, monkeypatch, guest_migration):
    url = f"sqlite:///{tmp_path}/guest-migration.sqlite3"
    monkeypatch.setattr(settings, "DATABASE_URL", url)
    command.upgrade(Config(str(ROOT / "alembic.ini")), "c53a7d9e8421")
    engine = create_engine(url)

    @event.listens_for(engine, "connect")
    def foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    with engine.begin() as db:
        db.execute(
            text(
                "INSERT INTO users (id,email,hashed_password,full_name,is_active,is_admin) VALUES (1,'registered@guest-migration.test','retained-hash','Retained customer',1,0),(2,'staff@guest-migration.test','retained-staff-hash',NULL,1,1)"
            )
        )
        db.execute(
            text(
                "INSERT INTO chat_sessions (id,user_id,title,memory_type,context_window,session_key,session_metadata) VALUES (10,1,'Retained private','sliding_window',10,:key,:metadata)"
            ),
            {"key": str(uuid4()), "metadata": '{"access_mode":"private","retained":true}'},
        )
        db.execute(
            text(
                "INSERT INTO messages (id,session_id,role,content,status) VALUES (1,10,'USER','Retained conversation','COMPLETED')"
            )
        )
        db.execute(
            text(
                "INSERT INTO handoff_tickets (id,session_id,user_id,reason,summary,status,assigned_to) VALUES (1,10,1,'explicit','Retained ticket','claimed',2)"
            )
        )
        db.execute(
            text(
                "INSERT INTO handoff_messages (id,session_id,ticket_id,author_user_id,author_kind,client_message_id,content) VALUES (1,10,1,2,'staff',:key,'Retained human answer')"
            ),
            {"key": str(uuid4())},
        )
        db.execute(
            text(
                "INSERT INTO action_executions (action_id,scope_session_id,live_session_id,user_id,scope,tool_name,arguments_hash,status) VALUES (:key,10,10,1,'10','process_refund',:hash,'in_progress')"
            ),
            {"key": str(uuid4()), "hash": "a" * 64},
        )
    try:
        yield guest_migration, engine
    finally:
        engine.dispose()


@contextmanager
def migration_connection(module, engine, monkeypatch):
    with engine.connect() as db:
        context = MigrationContext.configure(db)
        monkeypatch.setattr(module, "op", Operations(context))
        with context.begin_transaction(_per_migration=True):
            yield db


def snapshot(db: Connection) -> dict[str, Any]:
    return {
        name: db.execute(text(f"SELECT {columns} FROM {name} ORDER BY id")).all()
        for name, columns in {
            "users": "id,email,hashed_password,full_name,is_active,is_admin,created_at,updated_at",
            "chat_sessions": "id,user_id,title,session_key,session_metadata,created_at,updated_at",
            "messages": "id,session_id,content",
            "handoff_tickets": "id,session_id,user_id,assigned_to,summary",
            "handoff_messages": "id,session_id,author_user_id,content",
        }.items()
    } | {"ledger": db.execute(text("SELECT live_session_id,user_id FROM action_executions")).all()}


def test_upgrade_retains_registered_rows_children_indexes_and_is_replayable(
    guest_migration_db, monkeypatch
):
    module, engine = guest_migration_db
    with migration_connection(module, engine, monkeypatch) as db:
        before = snapshot(db)
        module.upgrade()
        assert snapshot(db) == before
        assert db.execute(text("SELECT is_guest FROM users ORDER BY id")).scalars().all() == [0, 0]
        module.upgrade()
        assert snapshot(db) == before
        columns = {c["name"]: c for c in inspect(db).get_columns("users")}
        assert columns["email"]["nullable"] and columns["hashed_password"]["nullable"]
        assert not columns["is_guest"]["nullable"]
        assert any(
            c["name"] == "ck_users_identity_kind"
            for c in inspect(db).get_check_constraints("users")
        )
        assert any(
            i["name"] == "ix_users_email" and i["unique"] for i in inspect(db).get_indexes("users")
        )
        assert inspect(db).has_table("guest_visitors")
        assert db.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert db.exec_driver_sql("PRAGMA foreign_key_check").all() == []


def test_upgrade_failure_rolls_back_user_rebuild_and_restores_foreign_keys(
    guest_migration_db, monkeypatch
):
    module, engine = guest_migration_db

    def refuse(_conn, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().startswith("CREATE TABLE _alembic_tmp_users"):
            raise RuntimeError("synthetic users rebuild refusal")

    with migration_connection(module, engine, monkeypatch) as db:
        before = snapshot(db)
        event.listen(engine, "before_cursor_execute", refuse)
        try:
            with pytest.raises(RuntimeError, match="synthetic"):
                module.upgrade()
        finally:
            event.remove(engine, "before_cursor_execute", refuse)
        assert snapshot(db) == before
        assert "is_guest" not in {c["name"] for c in inspect(db).get_columns("users")}
        assert not inspect(db).has_table("guest_visitors")
        assert db.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert db.exec_driver_sql("PRAGMA foreign_key_check").all() == []


@pytest.mark.parametrize("retained", ["user", "visitor"])
def test_downgrade_refuses_any_guest_data_without_changing_parent(
    guest_migration_db, monkeypatch, retained
):
    module, engine = guest_migration_db
    with migration_connection(module, engine, monkeypatch) as db:
        module.upgrade()
        if retained == "user":
            db.execute(
                text(
                    "INSERT INTO users (id,email,hashed_password,is_guest,is_active,is_admin) VALUES (3,NULL,NULL,1,1,0)"
                )
            )
        else:
            db.execute(
                text(
                    "INSERT INTO guest_visitors (id,user_id,token_hash,expires_at,current_session_id) VALUES (:id,1,:hash,'2099-01-01',10)"
                ),
                {"id": str(uuid4()), "hash": "a" * 64},
            )
        before = snapshot(db)
        with pytest.raises(RuntimeError, match="prevents downgrade"):
            module.downgrade()
        assert snapshot(db) == before
        assert inspect(db).has_table("guest_visitors")
        assert db.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1


def test_registered_only_downgrade_keeps_all_children_and_original_nullability(
    guest_migration_db, monkeypatch
):
    module, engine = guest_migration_db
    with migration_connection(module, engine, monkeypatch) as db:
        before = snapshot(db)
        module.upgrade()
        module.downgrade()
        assert snapshot(db) == before
        columns = {c["name"]: c for c in inspect(db).get_columns("users")}
        assert not columns["email"]["nullable"] and not columns["hashed_password"]["nullable"]
        assert "is_guest" not in columns
        assert not inspect(db).has_table("guest_visitors")
        assert db.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert db.exec_driver_sql("PRAGMA foreign_key_check").all() == []


def test_failed_downgrade_restores_guest_schema_and_registered_children(
    guest_migration_db, monkeypatch
):
    module, engine = guest_migration_db

    def refuse(_conn, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().startswith("CREATE TABLE _alembic_tmp_users"):
            raise RuntimeError("synthetic downgrade refusal")

    with migration_connection(module, engine, monkeypatch) as db:
        module.upgrade()
        before = snapshot(db)
        event.listen(engine, "before_cursor_execute", refuse)
        try:
            with pytest.raises(RuntimeError, match="synthetic"):
                module.downgrade()
        finally:
            event.remove(engine, "before_cursor_execute", refuse)
        assert snapshot(db) == before
        assert "is_guest" in {c["name"] for c in inspect(db).get_columns("users")}
        assert inspect(db).has_table("guest_visitors")
        assert db.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1


def test_current_session_fk_sets_null_and_user_delete_cascades_visitor(
    guest_migration_db, monkeypatch
):
    module, engine = guest_migration_db
    with migration_connection(module, engine, monkeypatch) as db:
        module.upgrade()
        db.execute(
            text(
                "INSERT INTO users (id,email,hashed_password,is_guest,is_active,is_admin) VALUES (3,NULL,NULL,1,1,0)"
            )
        )
        db.execute(
            text(
                "INSERT INTO chat_sessions (id,user_id,title,memory_type,context_window,session_key) VALUES (30,3,'Guest','sliding_window',10,:key)"
            ),
            {"key": str(uuid4())},
        )
        visitor_id = str(uuid4())
        db.execute(
            text(
                "INSERT INTO guest_visitors (id,user_id,token_hash,expires_at,current_session_id) VALUES (:id,3,:hash,'2099-01-01',30)"
            ),
            {"id": visitor_id, "hash": "b" * 64},
        )
        db.execute(text("DELETE FROM chat_sessions WHERE id=30"))
        assert (
            db.execute(
                text("SELECT current_session_id FROM guest_visitors WHERE id=:id"),
                {"id": visitor_id},
            ).scalar()
            is None
        )
        db.execute(text("DELETE FROM users WHERE id=3"))
        assert db.execute(text("SELECT count(*) FROM guest_visitors")).scalar() == 0
        assert db.execute(text("SELECT count(*) FROM users")).scalar() == 2
        assert db.exec_driver_sql("PRAGMA foreign_key_check").all() == []


@pytest.mark.parametrize("dialect", ["postgresql", "sqlite"])
def test_offline_upgrade_and_guarded_downgrade_sql(guest_migration, monkeypatch, dialect):
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name=dialect, opts={"as_sql": True, "output_buffer": output}
    )
    monkeypatch.setattr(guest_migration, "op", Operations(context))
    guest_migration.upgrade()
    sql = output.getvalue()
    assert "is_guest" in sql and "CREATE TABLE guest_visitors" in sql
    assert "ck_users_identity_kind" in sql and "ON DELETE SET NULL" in sql
    assert "uq_guest_visitors_user_id" in sql and "uq_guest_visitors_token_hash" in sql
    output.truncate(0)
    output.seek(0)
    if dialect == "postgresql":
        guest_migration.downgrade()
        sql = output.getvalue()
        assert (
            sql.index("LOCK TABLE")
            < sql.index("EXISTS (SELECT 1 FROM users")
            < sql.index("DROP TABLE guest_visitors")
        )
    else:
        with pytest.raises(RuntimeError, match="requires online"):
            guest_migration.downgrade()
        assert "DROP" not in output.getvalue()


def test_generated_offline_sqlite_upgrade_executes_without_losing_children(
    guest_migration_db, monkeypatch
):
    module, engine = guest_migration_db
    with engine.connect() as db:
        before = snapshot(db)
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="sqlite", opts={"as_sql": True, "output_buffer": output}
    )
    monkeypatch.setattr(module, "op", Operations(context))
    module.upgrade()
    raw = engine.raw_connection()
    try:
        raw.driver_connection.executescript(output.getvalue())
        assert raw.driver_connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        raw.close()
    with engine.connect() as db:
        assert snapshot(db) == before
        assert db.exec_driver_sql("PRAGMA foreign_key_check").all() == []


def test_orm_adoption_preserves_existing_guest_and_matches_model_metadata(
    guest_migration, monkeypatch
):
    engine = create_engine("sqlite:///:memory:")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as db:
            user = User(is_guest=True, email=None, hashed_password=None, is_admin=False)
            db.add(user)
            db.flush()
            chat = ChatSession(user_id=user.id, session_metadata={"access_mode": "private"})
            db.add(chat)
            db.flush()
            visitor = GuestVisitor(
                user_id=user.id,
                token_hash="c" * 64,
                expires_at=datetime(2099, 1, 1, tzinfo=UTC),
                current_session_id=chat.id,
            )
            db.add(visitor)
            db.commit()
        with migration_connection(guest_migration, engine, monkeypatch) as db:
            before = db.execute(text("SELECT * FROM guest_visitors")).all()
            guest_migration.upgrade()
            assert db.execute(text("SELECT * FROM guest_visitors")).all() == before
            assert (
                compare_metadata(
                    MigrationContext.configure(db, opts={"compare_server_default": True}),
                    Base.metadata,
                )
                == []
            )
    finally:
        engine.dispose()


def test_alembic_registered_data_upgrade_downgrade_reupgrade_matches_identity_models(
    guest_migration_db,
):
    _module, engine = guest_migration_db
    config = Config(str(ROOT / "alembic.ini"))
    with engine.connect() as db:
        before = snapshot(db)
    command.upgrade(config, "d64b8c0f9532")
    with engine.connect() as db:
        assert snapshot(db) == before
        assert (
            db.execute(text("SELECT version_num FROM alembic_version")).scalar() == "d64b8c0f9532"
        )

        def identity_tables(_obj, name, type_, _reflected, _compare_to):
            return type_ != "table" or name in {"users", "guest_visitors"}

        assert (
            compare_metadata(
                MigrationContext.configure(
                    db,
                    opts={"compare_server_default": True, "include_object": identity_tables},
                ),
                Base.metadata,
            )
            == []
        )
    command.downgrade(config, "c53a7d9e8421")
    with engine.connect() as db:
        assert snapshot(db) == before
        assert "is_guest" not in {c["name"] for c in inspect(db).get_columns("users")}
    command.upgrade(config, "d64b8c0f9532")
    with engine.connect() as db:
        assert snapshot(db) == before
        assert db.exec_driver_sql("PRAGMA foreign_key_check").all() == []
