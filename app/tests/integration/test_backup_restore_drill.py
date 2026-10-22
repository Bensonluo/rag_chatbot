"""Opt-in recovery drill using only a newly created PostgreSQL Compose project.

Run with RAG_RUN_RESTORE_DRILL=1. No configured application database is used.
The same backup/restore entry points used by operators must preserve migrated
schema, source bytes, historical citations, ticket state, and sequences.
"""

import gzip
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.postgres import PostgresSaver
from sqlalchemy import MetaData, create_engine, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.models.database.document import Document, DocumentVersion
from app.models.database.knowledge_gap import KnowledgeGapRecord
from app.models.database.knowledge_gap_resolution import KnowledgeGapResolution
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.models.database.user import User
from app.models.database.user_fact import UserFactRecord
from app.models.enums.message import MessageRole

pytestmark = pytest.mark.skipif(
    os.environ.get("RAG_RUN_RESTORE_DRILL") != "1",
    reason="requires explicit opt-in to create a disposable Docker Compose project",
)
ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "deployment" / "production" / "scripts"


def _run(
    args: list[str], env: dict[str, str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=120)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def _snapshot(engine: Engine) -> dict[str, list[dict[str, Any]]]:
    metadata = MetaData()
    metadata.reflect(bind=engine)
    with engine.connect() as connection:
        return {
            table.name: [
                dict(row)
                for row in connection.execute(select(table).order_by(*table.primary_key)).mappings()
            ]
            for table in metadata.sorted_tables
        }


def _schema(engine: Engine) -> dict[str, list[tuple[Any, ...]]]:
    """Compare restored column definitions, constraints and indexes, including
    tables owned by LangGraph rather than Alembic's application metadata.
    """
    queries = {
        "columns": """SELECT table_name,column_name,data_type,is_nullable,column_default,udt_name
            FROM information_schema.columns WHERE table_schema='public'
            ORDER BY table_name,ordinal_position""",
        # PostgreSQL reparses the varchar[] -> text[] predicate during restore,
        # moving casts between the array and its elements. Compare this index's
        # structure here and its actual membership/uniqueness behavior below.
        "indexes": """SELECT tablename,indexname,
            CASE WHEN indexname='uq_handoff_tickets_active_session'
                THEN split_part(indexdef, ' WHERE ', 1) ELSE indexdef END
            FROM pg_indexes
            WHERE schemaname='public' ORDER BY tablename,indexname""",
        "constraints": """SELECT r.relname,c.conname,pg_get_constraintdef(c.oid)
            FROM pg_constraint c JOIN pg_class r ON c.conrelid=r.oid
            JOIN pg_namespace n ON r.relnamespace=n.oid
            WHERE n.nspname='public' ORDER BY r.relname,c.conname""",
    }
    with engine.connect() as connection:
        return {
            name: [tuple(row) for row in connection.execute(text(query))]
            for name, query in queries.items()
        }


def _active_ticket_membership(engine: Engine) -> list[bool | None]:
    with engine.connect() as connection:
        predicate = connection.scalar(
            text("""SELECT pg_get_expr(indpred, indrelid)
            FROM pg_index WHERE indexrelid='uq_handoff_tickets_active_session'::regclass""")
        )
        assert isinstance(predicate, str)
        # Predicate comes only from the isolated fixture's PostgreSQL catalog.
        return [
            connection.scalar(
                text(f"SELECT {predicate} FROM (SELECT CAST(:status AS varchar) AS status) s"),
                {"status": status},
            )
            for status in ("open", "claimed", "resolved", "unknown", None)
        ]


def _seed_checkpoint(url: str) -> None:
    with PostgresSaver.from_conn_string(
        url.replace("postgresql+pg8000://", "postgresql://")
    ) as saver:
        saver.setup()
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {
            "pending_confirmation": {"tool": "refund", "amount": 42, "executed": False},
            "messages": [{"role": "user", "content": "请先确认退款"}],
        }
        checkpoint["channel_versions"] = {"pending_confirmation": 1, "messages": 1}
        config = saver.put(
            {"configurable": {"thread_id": "recovery-session", "checkpoint_ns": ""}},
            checkpoint,
            {"source": "input", "step": 0, "parents": {}},
            checkpoint["channel_versions"],
        )
        saver.put_writes(config, [("intent", "refund")], "recovery-task")


def _seed(engine: Engine) -> None:
    with Session(engine) as session:
        visitor = User(email="recovery@example.invalid", hashed_password="fixture-only")
        session.add(visitor)
        session.flush()
        chat = ChatSession(user_id=visitor.id, title="退款进度 / refund", reopened_count=2)
        session.add(chat)
        session.flush()
        document = Document(
            external_doc_id="recovery-policy",
            title="退款政策",
            source="policy.txt",
            doc_type="txt",
            active_version_id="old-policy-version",
            chunk_count=1,
        )
        session.add(document)
        session.flush()
        session.add_all(
            [
                DocumentVersion(
                    id="old-policy-version",
                    document_id=document.external_doc_id,
                    title=document.title,
                    filename="policy.txt",
                    file_type="txt",
                    content_hash="a" * 64,
                    revision=1,
                    source_content="旧版退款政策\n".encode(),
                    normalized_text="原路退款需要3至5个工作日。",
                    status="succeeded",
                    source_metadata={"language": "zh"},
                    pipeline_config={"chunk_size": 800},
                ),
                Message(session_id=chat.id, role=MessageRole.USER, content="我的退款到哪了？"),
                Message(
                    session_id=chat.id,
                    role=MessageRole.ASSISTANT,
                    content="已提交人工处理，参考当时政策。",
                    message_metadata={
                        "handoff_outcome": "queued",
                        "citations": [
                            {
                                "source_type": "document",
                                "document_id": document.external_doc_id,
                                "version_id": "old-policy-version",
                                "chunk_id": "old-chunk",
                                "excerpt": "原路退款需要3至5个工作日。",
                            }
                        ],
                    },
                ),
                HandoffTicket(
                    session_id=chat.id,
                    user_id=visitor.id,
                    reason="explicit",
                    status="claimed",
                    assigned_to=visitor.id,
                    priority=1,
                    summary='{"context":"pending refund"}',
                ),
                UserFactRecord(
                    user_id=visitor.id, fact="用户偏好中文", source_session_id=str(chat.id)
                ),
                KnowledgeGapRecord(query="节假日到账吗", normalized_query="节假日到账吗"),
                KnowledgeGapResolution(normalized_query="节假日到账吗", resolved_by=visitor.id),
            ]
        )
        session.commit()


def test_postgres_backup_restores_data_and_rejects_unsafe_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = "rag-recovery-" + uuid.uuid4().hex[:12]
    compose = tmp_path / "compose.yaml"
    compose.write_text("""services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: archive_owner
      POSTGRES_PASSWORD: disposable_fixture_password
      POSTGRES_DB: archive_source
    ports:
      - "127.0.0.1::5432"
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "archive_owner", "-d", "archive_source"]
      interval: 1s
      timeout: 3s
      retries: 30
""")
    # Explicit project/file/service prevent access to the repository's Compose
    # stack. Host database/role variables must not override container defaults.
    env = {
        **os.environ,
        "COMPOSE_FILE": str(compose),
        "COMPOSE_PROJECT_NAME": project,
        "PROJECT_DIRECTORY": str(tmp_path),
        "DB_SERVICE": "db",
        "BACKUP_DIR": str(tmp_path / "backups"),
    }
    for key in ("POSTGRES_USER", "POSTGRES_DB", "ENV_FILE", "RESTORE_DB"):
        env.pop(key, None)
    docker = ["docker", "compose", "-p", project, "-f", str(compose)]
    source: Engine | None = None
    restored: Engine | None = None
    failed: Engine | None = None
    try:
        _run([*docker, "up", "-d", "--wait", "--wait-timeout", "60"], env)
        address = _run([*docker, "port", "db", "5432"], env).stdout.strip()
        base_url = f"postgresql+pg8000://archive_owner:disposable_fixture_password@{address}/"
        monkeypatch.setattr(settings, "DATABASE_URL", base_url + "archive_source")
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
        command.check(Config(str(ROOT / "alembic.ini")))
        source = create_engine(base_url + "archive_source")
        _seed(source)
        _seed_checkpoint(base_url + "archive_source")
        before = _snapshot(source)
        schema_before = _schema(source)
        assert _active_ticket_membership(source) == [True, True, False, False, None]
        _run(["bash", str(SCRIPTS / "backup.sh"), "7"], env)
        archives = list((tmp_path / "backups").glob("db_backup_*.sql.gz"))
        assert len(archives) == 1
        archive = archives[0]
        assert archive.stat().st_mode & 0o777 == 0o600
        assert "COPY public.messages" in gzip.decompress(archive.read_bytes()).decode()

        restore = ["bash", str(SCRIPTS / "restore.sh"), str(archive), "--database"]
        _run([*restore, "recovered"], env)
        restored = create_engine(base_url + "recovered")
        assert _snapshot(restored) == before
        schema_after = _schema(restored)
        assert schema_after == schema_before, {
            key: {
                "missing": set(schema_before[key]) - set(schema_after[key]),
                "added": set(schema_after[key]) - set(schema_before[key]),
            }
            for key in schema_before
            if schema_before[key] != schema_after[key]
        }
        assert _active_ticket_membership(restored) == [True, True, False, False, None]
        with PostgresSaver.from_conn_string(
            (base_url + "recovered").replace("postgresql+pg8000://", "postgresql://")
        ) as saver:
            recovered_state = saver.get_tuple({"configurable": {"thread_id": "recovery-session"}})
            assert recovered_state is not None
            assert recovered_state.checkpoint["channel_values"]["pending_confirmation"] == {
                "tool": "refund",
                "amount": 42,
                "executed": False,
            }
            assert recovered_state.pending_writes == [("recovery-task", "intent", "refund")]

        # Neither an existing recovered database nor the original may be touched.
        assert _run([*restore, "recovered"], env, check=False).returncode != 0
        assert _run([*restore, "archive_source"], env, check=False).returncode != 0
        assert _snapshot(restored) == before
        assert _snapshot(source) == before

        # Restored sequences and the active-ticket uniqueness constraint still work.
        with Session(restored) as session:
            user = User(email="after-restore@example.invalid", hashed_password="fixture-only")
            session.add(user)
            session.commit()
            assert user.id > max(row["id"] for row in before["users"])
        with (
            pytest.raises(IntegrityError, match="uq_handoff_tickets_active_session"),
            restored.begin() as connection,
        ):
            connection.execute(
                text("""INSERT INTO handoff_tickets
                (session_id, user_id, reason, summary, status)
                VALUES (1, 1, 'explicit', '', 'open')""")
            )

        # An SQL error after a valid write must roll back the entire new database.
        bad_sql = tmp_path / "invalid.sql.gz"
        bad_sql.write_bytes(
            gzip.compress(b"CREATE TABLE partial(id int);\nSELECT missing_function();\n")
        )
        result = _run(
            ["bash", str(SCRIPTS / "restore.sh"), str(bad_sql), "--database", "failed_restore"],
            env,
            check=False,
        )
        assert result.returncode != 0
        failed = create_engine(base_url + "failed_restore")
        assert inspect(failed).get_table_names() == []

        # A corrupt gzip with executable SQL in its prefix must not create a DB.
        corrupt = tmp_path / "corrupt.sql.gz"
        corrupt.write_bytes(gzip.compress(b"CREATE TABLE partial(id int);\n")[:-5])
        result = _run(
            ["bash", str(SCRIPTS / "restore.sh"), str(corrupt), "--database", "corrupt_restore"],
            env,
            check=False,
        )
        assert result.returncode != 0
        with source.connect() as connection:
            assert (
                connection.scalar(
                    text("SELECT count(*) FROM pg_database WHERE datname='corrupt_restore'")
                )
                == 0
            )
        assert _snapshot(source) == before
    finally:
        for engine in (failed, restored, source):
            if engine is not None:
                engine.dispose()
        _run([*docker, "down", "--volumes", "--remove-orphans"], env)
