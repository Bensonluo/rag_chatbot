"""Clearing history removes durable messages without pretending failed resets succeeded."""

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_current_user
from app.api.v1 import chat as chat_api
from app.config.settings import settings
from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.ticket import HandoffTicket
from app.models.database.user_fact import UserFactRecord
from app.models.enums.message import MessageRole
from app.services.chat.chat_service import ChatService
from app.services.chat.persistence import ChatMessagePersister


@pytest.fixture
async def clear_stack(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[Any, ...]]:
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    persister = ChatMessagePersister(session_maker=sessions)
    checkpoint = SimpleNamespace(adelete_thread=AsyncMock())
    service = ChatService(graph=SimpleNamespace(checkpointer=checkpoint), persister=persister)
    app = FastAPI()
    app.include_router(chat_api.router, prefix="/api/v1")
    app.dependency_overrides[chat_api.get_chat_service] = lambda: service
    app.dependency_overrides[get_current_user] = lambda: None
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            yield client, service, persister, sessions, checkpoint
    finally:
        await persister.drain()
        await engine.dispose()


async def test_delete_clears_durable_messages_and_summaries_only_for_target_session(clear_stack):
    client, _, persister, sessions, checkpoint = clear_stack
    for session_id in (1, 2):
        await persister.persist_turn(session_id, 0, "Question", "Answer")
    async with sessions() as db:
        db.add(Message(session_id=1, role=MessageRole.SYSTEM, content="Old summary"))
        db.add(UserFactRecord(user_id=7, fact="Prefers email", source_session_id="1"))
        db.add(HandoffTicket(session_id=1, reason="explicit", summary="Support request"))
        await db.commit()
    assert (await client.get("/api/v1/chat/history?session_id=1")).json()["messages"]

    response = await client.delete("/api/v1/chat/history?session_id=1")

    assert response.status_code == 204
    assert (await client.get("/api/v1/chat/history?session_id=1")).json()["messages"] == []
    async with sessions() as db:
        remaining = (await db.execute(select(Message))).scalars().all()
        assert len(remaining) == 2
        assert {row.session_id for row in remaining} == {2}
        assert (await db.execute(select(UserFactRecord))).scalar_one().fact == "Prefers email"
        assert (await db.execute(select(HandoffTicket))).scalar_one().status == "open"
    checkpoint.adelete_thread.assert_awaited_once_with("1")


@pytest.mark.parametrize("failure", ["database_commit", "checkpoint"])
async def test_failed_clear_returns_500_instead_of_false_success(clear_stack, monkeypatch, failure):
    client, _, persister, sessions, checkpoint = clear_stack
    await persister.persist_turn(1, 0, "Question", "Answer")
    if failure == "checkpoint":
        checkpoint.adelete_thread.side_effect = RuntimeError("checkpoint offline")
    with monkeypatch.context() as patches:
        if failure == "database_commit":
            patches.setattr(
                AsyncSession, "commit", AsyncMock(side_effect=RuntimeError("commit failed"))
            )
        response = await client.delete("/api/v1/chat/history?session_id=1")
    assert response.status_code == 500
    async with sessions() as db:
        assert len((await db.execute(select(Message))).scalars().all()) == 2


@pytest.mark.parametrize("background_kind", ["compression", "extraction"])
async def test_clear_drains_target_background_writes_without_waiting_for_other_sessions(
    clear_stack, background_kind
):
    client, _, persister, sessions, _ = clear_stack
    target_started = asyncio.Event()
    target_release = asyncio.Event()
    unrelated_started = asyncio.Event()
    unrelated_release = asyncio.Event()
    target_finished = asyncio.Event()

    async def compress(session_id: int) -> bool:
        if session_id == 1:
            target_started.set()
            await target_release.wait()
        else:
            unrelated_started.set()
            await unrelated_release.wait()
        async with sessions() as db:
            if background_kind == "compression":
                db.add(
                    Message(session_id=session_id, role=MessageRole.SYSTEM, content="Late summary")
                )
            else:
                db.add(
                    UserFactRecord(user_id=7, fact="Preference", source_session_id=str(session_id))
                )
            await db.commit()
        if session_id == 1:
            target_finished.set()
        return True

    if background_kind == "compression":
        persister._compressor = Mock(maybe_compress=compress)
    else:
        persister._fact_extractor = Mock(maybe_extract=compress)
    await persister.persist_turn(1, 0, "Question", "Answer")
    await persister.persist_turn(2, 0, "Other", "Other answer")
    await asyncio.wait_for(target_started.wait(), 1)
    await asyncio.wait_for(unrelated_started.wait(), 1)
    clearing = asyncio.create_task(client.delete("/api/v1/chat/history?session_id=1"))
    try:
        await asyncio.sleep(0)
        assert not clearing.done()
        target_release.set()
        response = await asyncio.wait_for(clearing, 1)
        assert response.status_code == 204
        assert target_finished.is_set()
        assert not unrelated_release.is_set()
        async with sessions() as db:
            rows = (
                (await db.execute(select(Message).where(Message.session_id == 1))).scalars().all()
            )
            assert rows == []
            if background_kind == "extraction":
                assert (
                    await db.execute(select(UserFactRecord))
                ).scalar_one().source_session_id == "1"
    finally:
        target_release.set()
        unrelated_release.set()
        await persister.drain()
        await clearing
