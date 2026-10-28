"""Completion identity comes from the committed assistant row, never history order."""

from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config.settings import settings
from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.models.enums.message import MessageRole, MessageStatus
from app.services.chat.chat_service import ChatService, CompletionEvent
from app.services.chat.persistence import ChatMessagePersister


class _Graph:
    async def ainvoke(self, state, config):
        queue = config["configurable"].get("stream_queue")
        if queue is not None:
            queue.put_nowait("same answer")
        return {"response": "same answer", "intent": "direct"}


@pytest.fixture
async def receipt_stack(monkeypatch):
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as db:
        db.add(User(id=1, email="receipt@unit.invalid", hashed_password="!"))
        await db.flush()
        db.add(ChatSession(id=1, user_id=1))
        await db.commit()
    try:
        yield engine, maker, ChatMessagePersister(maker)
    finally:
        await engine.dispose()


async def test_receipt_is_immutable_and_identifies_the_committed_assistant(receipt_stack):
    _, maker, persister = receipt_stack
    receipt = await persister.persist_turn(1, 1, "question", "answer")
    assert receipt is not None
    assert receipt.session_id == 1
    assert receipt.status == "completed"
    with pytest.raises(FrozenInstanceError):
        receipt.message_id = 999
    async with maker() as db:
        row = await db.get(Message, receipt.message_id)
        assert row is not None
        assert row.role == MessageRole.ASSISTANT
        assert row.content == "answer"
        assert row.status == MessageStatus.COMPLETED


async def test_assistant_insert_failure_rolls_back_the_whole_turn(receipt_stack):
    _, maker, persister = receipt_stack
    async with maker() as db:
        await db.execute(
            text("""
            CREATE TRIGGER reject_assistant BEFORE INSERT ON messages
            WHEN NEW.role = 'ASSISTANT'
            BEGIN SELECT RAISE(ABORT, 'synthetic assistant rejection'); END
        """)
        )
        await db.commit()
    receipt = await persister.persist_turn(1, 1, "question", "answer")
    assert receipt is None
    async with maker() as db:
        assert (await db.execute(select(Message))).scalars().all() == []


async def test_receipt_survives_expiring_sessions_without_detached_orm_access(receipt_stack):
    engine, _, _ = receipt_stack
    persister = ChatMessagePersister(async_sessionmaker(engine, expire_on_commit=True))
    receipt = await persister.persist_turn(1, 1, "question", "answer")
    assert receipt is not None
    assert receipt.message_id > 0
    assert receipt.status == "completed"


async def test_empty_answer_has_no_receipt_but_retains_user_turn(receipt_stack):
    _, maker, persister = receipt_stack
    assert await persister.persist_turn(1, 1, "question", "") is None
    async with maker() as db:
        assert [row.role for row in (await db.execute(select(Message))).scalars()] == [
            MessageRole.USER
        ]


async def test_commit_failure_returns_no_receipt_and_rolls_back_flushed_rows(receipt_stack):
    engine, maker, _ = receipt_stack

    class RejectedCommitSession(AsyncSession):
        async def commit(self):
            raise RuntimeError("synthetic commit rejection")

    persister = ChatMessagePersister(
        async_sessionmaker(engine, class_=RejectedCommitSession, expire_on_commit=False)
    )
    assert await persister.persist_turn(1, 1, "question", "answer") is None
    async with maker() as db:
        assert (await db.execute(select(Message))).scalars().all() == []


async def test_housekeeping_failure_cannot_revoke_committed_receipt(receipt_stack, monkeypatch):
    _, _, persister = receipt_stack

    def reject_scheduling(session_id):
        raise RuntimeError("synthetic scheduling failure")

    monkeypatch.setattr(persister, "_schedule_compression", reject_scheduling)
    receipt = await persister.persist_turn(1, 1, "question", "answer")
    assert receipt is not None
    assert receipt.status == "completed"


async def test_legacy_null_status_history_remains_readable_and_unknown(receipt_stack):
    _, maker, persister = receipt_stack
    await persister.persist_turn(1, 1, "question", "legacy answer")
    async with maker() as db:
        # A historical nullable schema is deliberately isolated from the
        # maintained NOT NULL model; enum deserialization accepts NULL.
        await db.execute(text("CREATE TABLE legacy_messages AS SELECT * FROM messages"))
        await db.execute(text("DROP TABLE messages"))
        await db.execute(text("ALTER TABLE legacy_messages RENAME TO messages"))
        await db.execute(text("UPDATE messages SET status = NULL WHERE role = 'ASSISTANT'"))
        await db.commit()
    history = await persister.get_history(1, raise_on_error=True)
    assert history[1].content == "legacy answer"
    assert history[1].message_id is not None
    assert history[1].status is None


async def test_rest_response_and_history_carry_actual_id_status_and_rating(receipt_stack):
    _, maker, persister = receipt_stack
    service = ChatService(graph=_Graph(), persister=persister)
    response = await service.process_message(1, "question", 1)
    assert response.message_id is not None
    assert response.status == "completed"
    async with maker() as db:
        row = await db.get(Message, response.message_id)
        row.user_rating = -1
        await db.commit()
    history = await service.get_chat_history(1)
    assert history[1].message_id == response.message_id
    assert history[1].status == "completed"
    assert history[1].user_rating == -1
    assert history[0].user_rating is None


async def test_opted_in_stream_emits_only_its_committed_completion(receipt_stack):
    _, maker, persister = receipt_stack
    service = ChatService(graph=_Graph(), persister=persister)
    chunks = [
        chunk
        async for chunk in service.process_message_stream(1, "question", 1, include_completion=True)
    ]
    assert chunks[0] == "same answer"
    assert len(chunks) == 2
    completion = chunks[1]
    assert isinstance(completion, CompletionEvent)
    assert completion.session_id == 1
    assert completion.status == "completed"
    async with maker() as db:
        assert (await db.get(Message, completion.message_id)).content == "same answer"


async def test_failed_persistence_keeps_received_answer_without_completion(receipt_stack):
    _, maker, persister = receipt_stack
    async with maker() as db:
        await db.execute(text("DROP TABLE messages"))
        await db.commit()
    service = ChatService(graph=_Graph(), persister=persister)
    chunks = [
        chunk
        async for chunk in service.process_message_stream(1, "question", 1, include_completion=True)
    ]
    assert chunks == ["same answer"]
    response = await service.process_message(1, "question", 1)
    assert response.content == "same answer"
    assert response.message_id is None
    assert response.status is None


async def test_legacy_persister_none_return_does_not_invent_completion():
    persister = AsyncMock()
    persister.persist_turn.return_value = None
    service = ChatService(graph=_Graph(), persister=persister)
    chunks = [
        chunk
        async for chunk in service.process_message_stream(1, "question", 1, include_completion=True)
    ]
    assert chunks == ["same answer"]
    response = await service.process_message(1, "question", 1)
    assert response.message_id is None
    assert response.status is None


async def test_legacy_content_stream_does_not_receive_new_metadata(receipt_stack):
    _, _, persister = receipt_stack
    service = ChatService(graph=_Graph(), persister=persister)
    assert [chunk async for chunk in service.process_message_stream(1, "question", 1)] == [
        "same answer"
    ]


async def test_memory_history_keeps_identity_unknown():
    memory = AsyncMock()
    memory.get_context.return_value = [{"role": "assistant", "content": "legacy answer"}]
    service = ChatService(graph=_Graph(), memory_strategy=memory)
    history = await service.get_chat_history(1)
    assert history[0].message_id is None
    assert history[0].status is None
    assert history[0].user_rating is None
