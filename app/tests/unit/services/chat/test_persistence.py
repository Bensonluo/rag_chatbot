"""Tests for chat turn persistence (request-scoped sessions)."""

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession  # noqa: F401 (registers table)
from app.models.enums.message import MessageRole
from app.services.chat.chat_service import ChatService
from app.services.chat.persistence import ChatMessagePersister


class _FakeGraph:
    """Minimal stand-in for a compiled LangGraph.

    ``chunks`` are pushed onto the request's stream queue during ainvoke,
    mimicking terminal nodes under token streaming.
    """

    def __init__(self, result: dict[str, Any] | None = None, chunks: list[str] | None = None):
        self._result = result or {}
        self._chunks = chunks or []

    async def ainvoke(self, state, config):
        queue = (config.get("configurable") or {}).get("stream_queue")
        if queue is not None:
            for chunk in self._chunks:
                queue.put_nowait(chunk)
        return self._result


@pytest.fixture
async def session_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


class TestChatMessagePersister:
    """Persistence behavior"""

    async def test_persist_turn_writes_user_and_assistant_rows(self, session_maker):
        """One turn produces a user message and an assistant message"""
        # Arrange
        persister = ChatMessagePersister(session_maker=session_maker)

        # Act
        await persister.persist_turn(
            session_id=1,
            user_id=7,
            user_message="我的手机号是13800138000，帮我查下订单",
            response="您的订单已发货",
            intent="order_query",
            sources=["doc-1"],
        )

        # Assert
        async with session_maker() as session:
            rows = (await session.execute(select(Message).order_by(Message.id))).scalars().all()
        assert [r.role for r in rows] == [MessageRole.USER, MessageRole.ASSISTANT]
        assert rows[0].intent == "order_query"
        assert rows[1].message_metadata["sources"] == ["doc-1"]

    async def test_persist_failure_is_swallowed(self):
        """A database error never propagates to the chat path"""

        # Arrange
        def broken_maker():
            raise RuntimeError("db down")

        persister = ChatMessagePersister(session_maker=broken_maker)

        # Act / Assert — no exception raised
        await persister.persist_turn(session_id=1, user_id=1, user_message="hi", response="hello")

    async def test_get_history_reads_via_request_scoped_session(self, session_maker):
        """History reads come back as ChatMessage objects"""
        # Arrange
        persister = ChatMessagePersister(session_maker=session_maker)
        await persister.persist_turn(
            session_id=5, user_id=1, user_message="你好", response="您好，请问有什么可以帮您？"
        )

        # Act
        history = await persister.get_history(session_id=5)

        # Assert
        assert [(m.role, m.content) for m in history] == [
            ("user", "你好"),
            ("assistant", "您好，请问有什么可以帮您？"),
        ]

    async def test_get_history_returns_most_recent_turns(self, session_maker):
        """limit means the newest N turns, chronological — not the oldest N"""
        from datetime import datetime, timedelta

        base = datetime(2026, 9, 24, 12, 0, 0)
        async with session_maker() as session:
            for i in (1, 2, 3):
                session.add(
                    Message(
                        session_id=1,
                        role=MessageRole.USER,
                        content=f"第{i}条",
                        created_at=base + timedelta(seconds=10 * i),
                    )
                )
            await session.commit()

        persister = ChatMessagePersister(session_maker=session_maker)
        history = await persister.get_history(session_id=1, limit=2)

        assert [m.content for m in history] == ["第2条", "第3条"]

    async def test_get_history_skips_system_summary_rows(self, session_maker):
        """Summary artifacts are not conversation turns in history either"""
        from datetime import datetime, timedelta

        base = datetime(2026, 9, 24, 12, 0, 0)
        async with session_maker() as session:
            session.add(
                Message(
                    session_id=1,
                    role=MessageRole.USER,
                    content="问题",
                    created_at=base,
                )
            )
            session.add(
                Message(
                    session_id=1,
                    role=MessageRole.SYSTEM,
                    content="压缩摘要",
                    created_at=base + timedelta(seconds=10),
                )
            )
            await session.commit()

        persister = ChatMessagePersister(session_maker=session_maker)
        history = await persister.get_history(session_id=1, limit=10)

        assert [m.content for m in history] == ["问题"]


class TestChatServicePersistence:
    """ChatService integration with the persister"""

    async def test_process_message_persists_turn(self):
        """process_message writes the turn through the persister"""
        # Arrange
        persister = AsyncMock()
        service = ChatService(
            graph=_FakeGraph(result={"response": "已发货", "intent": "order_query"}),
            persister=persister,
        )

        # Act
        response = await service.process_message(session_id=3, message="查订单", user_id=9)

        # Assert
        assert response.content == "已发货"
        persister.persist_turn.assert_awaited_once()
        kwargs = persister.persist_turn.await_args.kwargs
        assert kwargs["user_message"] == "查订单"
        assert kwargs["response"] == "已发货"
        assert kwargs["session_id"] == 3

    async def test_stream_persists_accumulated_response(self):
        """Streaming persists the concatenated chunks when the stream ends"""
        # Arrange
        persister = AsyncMock()
        service = ChatService(
            graph=_FakeGraph(result={"response": "您好"}, chunks=["您", "好"]),
            persister=persister,
        )

        # Act
        chunks = [c async for c in service.process_message_stream(3, "查订单", 9)]

        # Assert
        assert chunks == ["您", "好"]
        persister.persist_turn.assert_awaited_once()
        assert persister.persist_turn.await_args.kwargs["response"] == "您好"

    async def test_get_history_prefers_persister(self):
        """History reads route through the persister when present"""
        # Arrange
        persister = AsyncMock()
        persister.get_history.return_value = []
        # object() placeholder proves the history path never touches memory.
        service = ChatService(
            graph=_FakeGraph(),
            persister=persister,
            memory_strategy=object(),  # type: ignore[arg-type]
        )

        # Act
        await service.get_chat_history(session_id=1)

        # Assert
        persister.get_history.assert_awaited_once()


class TestSummaryAwareHistory:
    """get_history(include_summary=True) is the LLM-context read."""

    async def test_include_summary_prepends_and_cuts(self, session_maker):
        async with session_maker() as session:
            session.add(
                Message(
                    session_id=1,
                    role=MessageRole.USER,
                    content="早期问题",
                    created_at=datetime(2026, 9, 24, 10, 0, 0),
                )
            )
            session.add(
                Message(
                    session_id=1,
                    role=MessageRole.SYSTEM,
                    content="早期已摘要",
                    created_at=datetime(2026, 9, 24, 10, 0, 5),
                )
            )
            session.add(
                Message(
                    session_id=1,
                    role=MessageRole.ASSISTANT,
                    content="新回答",
                    created_at=datetime(2026, 9, 24, 10, 0, 8),
                )
            )
            await session.commit()
        persister = ChatMessagePersister(session_maker=session_maker)

        history = await persister.get_history(session_id=1, limit=10, include_summary=True)

        # Summary block first, covered row cut, post-summary turn kept
        assert [m.role for m in history] == ["system", "assistant"]
        assert history[0].content == "Previous conversation: 早期已摘要"
        assert history[1].content == "新回答"

    async def test_default_read_stays_turns_only(self, session_maker):
        async with session_maker() as session:
            session.add(
                Message(
                    session_id=1,
                    role=MessageRole.USER,
                    content="早期问题",
                    created_at=datetime(2026, 9, 24, 10, 0, 0),
                )
            )
            session.add(
                Message(
                    session_id=1,
                    role=MessageRole.SYSTEM,
                    content="早期已摘要",
                    created_at=datetime(2026, 9, 24, 10, 0, 5),
                )
            )
            await session.commit()
        persister = ChatMessagePersister(session_maker=session_maker)

        history = await persister.get_history(session_id=1, limit=10)

        # API tail view: raw turns only, no synthetic summary block
        assert [m.content for m in history] == ["早期问题"]


class TestCompressionScheduling:
    """persist_turn schedules best-effort compression post-commit."""

    async def test_persist_turn_schedules_compression(self, session_maker):
        compressor = Mock()
        compressor.maybe_compress = AsyncMock(return_value=True)
        persister = ChatMessagePersister(session_maker=session_maker, compressor=compressor)

        await persister.persist_turn(session_id=42, user_id=7, user_message="问", response="答")
        await persister.drain()

        compressor.maybe_compress.assert_awaited_once_with(42)

    async def test_compressor_failure_does_not_break_persist(self, session_maker):
        compressor = Mock()
        compressor.maybe_compress = AsyncMock(side_effect=RuntimeError("boom"))
        persister = ChatMessagePersister(session_maker=session_maker, compressor=compressor)

        await persister.persist_turn(session_id=1, user_id=7, user_message="问", response="答")
        await persister.drain()

        # The turn itself must be durable even when compression explodes
        history = await persister.get_history(session_id=1)
        assert [m.content for m in history] == ["问", "答"]

    async def test_drain_without_pending_is_a_noop(self):
        persister = ChatMessagePersister(session_maker=lambda: None)
        await persister.drain()  # must not raise


class TestExtractionScheduling:
    """persist_turn schedules user-fact extraction post-commit (Phase B).

    Same fire-and-forget contract as compression: both are background
    consumers of the same durable turns, and neither may fail the turn.
    """

    async def test_persist_turn_schedules_extraction(self, session_maker):
        extractor = Mock()
        extractor.maybe_extract = AsyncMock(return_value=True)
        persister = ChatMessagePersister(session_maker=session_maker, fact_extractor=extractor)

        await persister.persist_turn(session_id=42, user_id=7, user_message="问", response="答")
        await persister.drain()

        extractor.maybe_extract.assert_awaited_once_with(42)

    async def test_extraction_scheduled_alongside_compression(self, session_maker):
        """Both background consumers fire on the same durable turn."""
        compressor = Mock()
        compressor.maybe_compress = AsyncMock(return_value=True)
        extractor = Mock()
        extractor.maybe_extract = AsyncMock(return_value=True)
        persister = ChatMessagePersister(
            session_maker=session_maker,
            compressor=compressor,
            fact_extractor=extractor,
        )

        await persister.persist_turn(session_id=1, user_id=7, user_message="问", response="答")
        await persister.drain()

        compressor.maybe_compress.assert_awaited_once_with(1)
        extractor.maybe_extract.assert_awaited_once_with(1)

    async def test_extractor_failure_does_not_break_persist(self, session_maker):
        extractor = Mock()
        extractor.maybe_extract = AsyncMock(side_effect=RuntimeError("boom"))
        persister = ChatMessagePersister(session_maker=session_maker, fact_extractor=extractor)

        await persister.persist_turn(session_id=1, user_id=7, user_message="问", response="答")
        await persister.drain()

        history = await persister.get_history(session_id=1)
        assert [m.content for m in history] == ["问", "答"]

    async def test_drain_awaits_extractions_too(self, session_maker):
        """drain covers both pending sets — shutdown must not strand
        extraction tasks mid-write."""
        extractor = Mock()
        extractor.maybe_extract = AsyncMock(return_value=True)
        persister = ChatMessagePersister(session_maker=session_maker, fact_extractor=extractor)

        await persister.persist_turn(session_id=9, user_id=7, user_message="问", response="答")
        await persister.drain()

        extractor.maybe_extract.assert_awaited_once_with(9)
