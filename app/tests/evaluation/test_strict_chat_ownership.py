"""Real JWT, graph and database checks for chat session ownership.

The owner leaves an order ID in the checkpoint through a real refund turn.
Foreign input must be rejected before it can read that state, persist a turn,
create a handoff, or create a checkpoint for a missing session.
"""

from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_db
from app.api.v1 import chat as chat_api
from app.config.settings import settings
from app.core.security import create_access_token
from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.models.database.user import User
from app.services.chat.chat_service import ChatService
from app.services.chat.persistence import ChatMessagePersister
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.tools import ToolRegistry
from app.services.handoff.service import HandoffService
from app.services.intent.rule_based import RuleBasedIntentDetector


@dataclass
class OwnershipAPI:
    client: AsyncClient
    service: ChatService
    maker: async_sessionmaker[AsyncSession]
    headers: dict[int, dict[str, str]]

    async def rows(self) -> tuple[list[tuple[int, str]], list[tuple[int, int]]]:
        async with self.maker() as session:
            messages = (
                await session.execute(
                    select(Message.session_id, Message.content).order_by(Message.id)
                )
            ).all()
            tickets = (
                await session.execute(
                    select(HandoffTicket.session_id, HandoffTicket.user_id).order_by(
                        HandoffTicket.id
                    )
                )
            ).all()
            return [(row[0], row[1]) for row in messages], [(row[0], row[1]) for row in tickets]

    async def state(self, session_id: int) -> dict[str, Any]:
        snapshot = await self.service.graph.aget_state(
            {"configurable": {"thread_id": str(session_id)}}
        )
        return deepcopy(snapshot.values)


@pytest.fixture
async def ownership_api(tmp_path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[OwnershipAPI]:
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "SECRET_KEY", "isolated-chat-ownership-signing-key-32bytes")
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/ownership.sqlite3")

    @event.listens_for(engine.sync_engine, "connect")
    def enable_foreign_keys(connection: Any, _record: Any) -> None:
        connection.execute("PRAGMA foreign_keys=ON")

    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(
                lambda conn: Base.metadata.create_all(
                    conn,
                    tables=[
                        Base.metadata.tables[name]
                        for name in ("users", "chat_sessions", "messages", "handoff_tickets")
                    ],
                )
            )
        async with maker() as session:
            session.add_all(
                User(
                    id=user_id,
                    email=f"user-{user_id}@ownership.invalid",
                    hashed_password="!",
                    is_admin=user_id == 3,
                )
                for user_id in (1, 2, 3)
            )
            await session.flush()
            session.add_all([ChatSession(id=10, user_id=1), ChatSession(id=20, user_id=2)])
            await session.commit()

        service = ChatService(
            graph=build_dialogue_graph(
                intent_detector=RuleBasedIntentDetector(),
                slot_filler=None,
                tool_registry=ToolRegistry(),
                retrieval_pipeline=None,
                llm_service=None,
                checkpointer=MemorySaver(),
                handoff_service=HandoffService(maker),
            ),
            persister=ChatMessagePersister(maker),
        )

        async def isolated_db() -> AsyncIterator[AsyncSession]:
            async with maker() as session:
                yield session

        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        app.dependency_overrides[get_db] = isolated_db
        app.dependency_overrides[chat_api.get_chat_service] = lambda: service
        headers = {
            user_id: {
                "Authorization": "Bearer "
                + create_access_token(
                    {"sub": f"user-{user_id}@ownership.invalid", "user_id": user_id}
                )
            }
            for user_id in (1, 2, 3)
        }
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://ownership.invalid"
        ) as client:
            owner = await client.post(
                "/api/v1/chat",
                headers=headers[1],
                json={"session_id": 10, "message": "退款 订单号：OWNER12345"},
            )
            assert owner.status_code == 200, owner.text
            assert owner.json()["metadata"]["filled_slots"] == {"order_id": "OWNER12345"}
            assert owner.json()["metadata"]["llm_calls"] == 0
            yield OwnershipAPI(client, service, maker, headers)
    finally:
        await engine.dispose()


@pytest.mark.parametrize("path", ["/api/v1/chat", "/api/v1/chat/stream"])
@pytest.mark.parametrize("actor", [2, 3], ids=["other_customer", "other_admin"])
@pytest.mark.parametrize("session_id", [10, 999], ids=["foreign", "missing"])
async def test_strict_input_cannot_read_or_change_foreign_or_missing_session(
    ownership_api: OwnershipAPI, path: str, actor: int, session_id: int
) -> None:
    api = ownership_api
    before_rows = await api.rows()
    before_owner_state = await api.state(10)
    response = await api.client.post(
        path,
        headers=api.headers[actor],
        json={"session_id": session_id, "message": "human agent"},
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}
    assert "text/event-stream" not in response.headers.get("content-type", "")
    assert "OWNER12345" not in response.text
    assert await api.rows() == before_rows
    assert await api.state(10) == before_owner_state
    assert await api.state(999) == {}


@pytest.mark.parametrize("path", ["/api/v1/chat", "/api/v1/chat/stream"])
async def test_owner_can_still_chat_and_create_its_own_handoff(
    ownership_api: OwnershipAPI, path: str
) -> None:
    api = ownership_api
    response = await api.client.post(
        path, headers=api.headers[1], json={"session_id": 10, "message": "human agent"}
    )
    assert response.status_code == 200, response.text
    if path.endswith("/stream"):
        assert "data: [DONE]" in response.text
    else:
        assert response.json()["metadata"]["handoff_outcome"] == "queued"
        assert response.json()["metadata"]["llm_calls"] == 0
    messages, tickets = await api.rows()
    assert len(messages) == 4
    assert all(session_id == 10 for session_id, _ in messages)
    assert tickets == [(10, 1)]
    assert (await api.state(10))["user_id"] == 1


@pytest.mark.parametrize("path", ["/api/v1/chat", "/api/v1/chat/stream"])
async def test_demo_retains_anonymous_body_identity_and_shared_session_behavior(
    ownership_api: OwnershipAPI, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    api = ownership_api
    response = await api.client.post(
        path, json={"session_id": 10, "user_id": 2, "message": "human agent"}
    )
    assert response.status_code == 200, response.text
    if path.endswith("/stream"):
        assert "data: [DONE]" in response.text
    assert (await api.state(10))["user_id"] == 2
    messages, tickets = await api.rows()
    assert len(messages) == 4
    assert tickets == [(10, 2)]
