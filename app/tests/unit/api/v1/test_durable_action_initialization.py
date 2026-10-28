"""Local schema/readiness and factory binding for the durable action ledger.

Schema reads, graph construction, default registry and SQLite writes are
real. Provider construction is stopped or replaced locally; no external
model, Redis or default database is used.
"""

import socket
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.v1 import chat as chat_api
from app.config.settings import settings
from app.models.database.action_execution import ActionExecution
from app.models.database.base import Base
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.repositories.message_repository import MessageRepository
from app.repositories.session_repository import SessionRepository
from app.services.chat.coordination import SessionCoordinator
from app.services.chat.factory import ChatServiceFactory
from app.services.dialogue.action_ledger import ActionLedger
from app.services.dialogue.graph import build_dialogue_graph
from app.services.llm.base import LLMServiceBase


@pytest.fixture(autouse=True)
def no_external_io(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    attempts: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        attempts.append(repr(args))
        raise AssertionError("durable-action initialization tests forbid network IO")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        monkeypatch.setattr(socket, name, forbidden)
    yield
    assert attempts == []


@pytest.fixture
async def local_database(
    tmp_path: Path,
) -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/initialization.sqlite3")

    @event.listens_for(engine.sync_engine, "connect")
    def enable_foreign_keys(connection: Any, _record: Any) -> None:
        connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()


@pytest.mark.parametrize("schema_present", [False, True])
async def test_action_schema_checked_before_any_provider_is_constructed(
    local_database: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
    schema_present: bool,
) -> None:
    monkeypatch.setattr(chat_api, "_chat_service", None)
    async with local_database() as db:
        if not schema_present:
            await db.execute(text("DROP TABLE action_executions"))
            await db.commit()
        with (
            patch(
                "app.services.llm.LLMFactory.create_from_settings",
                side_effect=RuntimeError("isolated provider construction sentinel"),
            ) as provider,
            patch("app.services.embeddings.EmbeddingFactory.create_from_settings") as embeddings,
            patch("app.services.retrieval.RetrievalFactory.create_vector_client") as vector,
        ):
            if schema_present:
                # Positive control: these same real queries allow provider
                # construction when all three required tables are readable.
                with pytest.raises(RuntimeError, match="isolated provider construction sentinel"):
                    await chat_api.initialize_chat_service(db)
                provider.assert_called_once_with()
            else:
                with pytest.raises(OperationalError, match="no such table: action_executions"):
                    await chat_api.initialize_chat_service(db)
                provider.assert_not_called()
            embeddings.assert_not_called()
            vector.assert_not_called()
    assert chat_api._chat_service is None


async def test_production_factory_uses_coordinator_database_for_real_action_ledger(
    local_database: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    for setting in (
        "MCP_TOOLS_ENABLED",
        "CHAT_ANSWER_CACHE_ENABLED",
        "SEMANTIC_CACHE_ENABLED",
        "RETRIEVAL_CACHE_ENABLED",
        "AGENT_TOOLS_ENABLED",
        "AGENT_KNOWLEDGE_TOOL_ENABLED",
        "AGENT_ESCALATE_TOOL_ENABLED",
        "USER_FACT_RECALL_ENABLED",
        "FAQ_ENABLED",
    ):
        monkeypatch.setattr(settings, setting, False)
    async with local_database() as db:
        db.add(User(id=1, email="factory-action@evaluation.example.com", hashed_password="!"))
        await db.flush()
        db.add(ChatSession(id=11, user_id=1))
        await db.commit()
    coordinator = SessionCoordinator(local_database)
    provider = Mock(spec=LLMServiceBase, model="unused-local-provider")
    async with local_database() as db:
        with patch(
            "app.services.dialogue.graph.build_dialogue_graph", wraps=build_dialogue_graph
        ) as builder:
            service = ChatServiceFactory.create_with_defaults(
                llm_service=provider,
                message_repo=MessageRepository(db),
                session_repo=SessionRepository(db),
                memory_type="sliding_window",
                intent_type="rule_based",
                coordinator=coordinator,
            )
    assert service.graph is not None
    assert service.coordinator is coordinator
    builder.assert_called_once()
    registry = builder.call_args.kwargs["tool_registry"]
    assert isinstance(registry.action_ledger, ActionLedger)
    assert registry.action_ledger._session_maker is coordinator.sessions
    assert registry.durable_actions is True
    action_id = str(uuid4())
    # Exercise the produced registry against the explicit local maker;
    # graph/provider construction spies are not business outcome oracles.
    result = await registry.execute(
        "refund",
        {"order_id": "ORD1001", "reason": "damaged"},
        user_id=1,
        session_id=11,
        action_id=action_id,
        execution_scope="11",
    )
    assert result.execution_status == "completed"
    async with local_database() as db:
        row = await db.get(ActionExecution, action_id)
        assert row is not None and row.status == "completed"
        assert row.scope_session_id == row.live_session_id == 11
        assert row.user_id == 1 and row.scope == "11"
    provider.generate.assert_not_called()
