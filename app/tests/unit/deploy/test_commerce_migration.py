"""Commerce data is retained across downgrade attempts; empty schema can roundtrip."""

import importlib.util
from io import StringIO
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, event, inspect, text


@pytest.fixture
def migration():
    path = (
        Path(__file__).resolve().parents[4]
        / "migrations/versions/e75c9d1064a3_commerce_framework.py"
    )
    spec = importlib.util.spec_from_file_location("commerce_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_empty_upgrade_replay_and_downgrade_preserve_parent_user(migration, monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as db:
        db.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
        db.execute(text("INSERT INTO users VALUES (1)"))
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(db)))
        migration.upgrade()
        migration.upgrade()
        assert set(migration.TABLES).issubset(inspect(db).get_table_names())
        assert migration.down_revision == "d64b8c0f9532"
        assert inspect(db).get_foreign_keys("commerce_mock_objects") == []
        assert inspect(db).get_foreign_keys("commerce_purchases")[0]["referred_table"] == "users"
        migration.downgrade()
        assert inspect(db).get_table_names() == ["users"]
        assert db.scalar(text("SELECT id FROM users")) == 1
    engine.dispose()


@pytest.mark.parametrize("retained", ["purchase", "operation", "receipt", "provider"])
def test_retained_domain_or_provider_data_blocks_destructive_downgrade(
    migration, monkeypatch, retained
):
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def fk(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    with engine.begin() as db:
        db.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
        db.execute(text("INSERT INTO users VALUES (1)"))
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(db)))
        migration.upgrade()
        pid = str(uuid4())
        if retained in {"purchase", "operation"}:
            db.execute(
                text(
                    "INSERT INTO commerce_purchases (id,owner_user_id,mode,merchant_id,currency,stage,version,generation,quantity) VALUES (:id,1,'mock','demo-store','CNY','shopping',0,0,0)"
                ),
                {"id": pid},
            )
            if retained == "operation":
                db.execute(
                    text(
                        "INSERT INTO commerce_operations (id,purchase_id,kind,generation,status,request_hash,request) VALUES (:id,:pid,'order',0,'unknown','hash','{}')"
                    ),
                    {"id": str(uuid4()), "pid": pid},
                )
        elif retained == "receipt":
            db.execute(
                text(
                    "INSERT INTO commerce_mock_receipts (request_key,owner_user_id,kind,request_hash,status,result) VALUES ('retained',1,'order','hash','completed','{}')"
                )
            )
        else:
            db.execute(
                text(
                    "INSERT INTO commerce_mock_objects (id,kind,owner_user_id,version,payload) VALUES ('retained','order',1,0,'{}')"
                )
            )
        with pytest.raises(RuntimeError, match="prevents lossless downgrade"):
            migration.downgrade()
        assert set(migration.TABLES).issubset(inspect(db).get_table_names())
    engine.dispose()


def test_offline_postgres_has_lock_and_all_data_guards_before_drop(migration, monkeypatch):
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output}
    )
    monkeypatch.setattr(migration, "op", Operations(context))
    migration.upgrade()
    migration.downgrade()
    sql = output.getvalue()
    assert "uq_commerce_active_slot" in sql and "uq_commerce_operation_generation" in sql
    assert (
        sql.index("ACCESS EXCLUSIVE MODE")
        < sql.index("prevents lossless downgrade")
        < sql.index("DROP TABLE")
    )
    assert all(f"EXISTS (SELECT 1 FROM {table})" in sql for table in migration.TABLES)


def test_offline_sqlite_upgrade_executes_and_unsafe_downgrade_is_refused(migration, monkeypatch):
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="sqlite", opts={"as_sql": True, "output_buffer": output}
    )
    monkeypatch.setattr(migration, "op", Operations(context))
    migration.upgrade()
    import sqlite3

    db = sqlite3.connect(":memory:")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
    db.executescript(output.getvalue())
    assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    with pytest.raises(RuntimeError, match="requires online"):
        migration.downgrade()
    db.close()
