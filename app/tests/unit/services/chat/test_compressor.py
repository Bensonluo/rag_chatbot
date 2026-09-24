"""Tests for primary-path session compression (SessionCompressor).

Industry baseline: long CS sessions are periodically compressed into a
rolling summary so the bounded history window keeps early context
(订单号、承诺过的时效) without unbounded token growth. The primary
LangGraph pipeline never ran SummarizationMemory, so sessions past the
6-turn window silently lost their beginning — this compressor closes
that write side on the production path.
"""

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession  # noqa: F401 (registers table)
from app.models.enums.message import MessageRole
from app.repositories.message_repository import MessageRepository
from app.services.chat.compressor import SessionCompressor

BASE = datetime(2026, 9, 24, 10, 0, 0)


@pytest.fixture
async def session_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


def _llm_returning(text: str) -> Any:
    llm = Mock()
    llm.generate = AsyncMock(return_value=Mock(content=text))
    return llm


def _llm_raising() -> Any:
    llm = Mock()
    llm.generate = AsyncMock(side_effect=RuntimeError("llm down"))
    return llm


async def _seed_messages(
    maker: Any,
    session_id: int,
    count: int,
    offset: int = 0,
) -> None:
    """Seed `count` alternating user/assistant rows, chronological."""
    async with maker() as session:
        for i in range(count):
            session.add(
                Message(
                    session_id=session_id,
                    role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                    content=f"msg{offset + i:02d}",
                    created_at=BASE + timedelta(seconds=offset + i),
                )
            )
        await session.commit()


async def _latest_summary(maker: Any, session_id: int) -> Message | None:
    async with maker() as session:
        return await MessageRepository(session).get_latest_summary(session_id)


class TestCompressionTrigger:
    async def test_below_threshold_writes_nothing(self, session_maker):
        await _seed_messages(session_maker, session_id=1, count=10)
        llm = _llm_returning("摘要")

        compressor = SessionCompressor(
            session_maker=session_maker, llm_service=llm, threshold=20, interval=10
        )
        wrote = await compressor.maybe_compress(session_id=1)

        assert wrote is False
        llm.generate.assert_not_awaited()
        assert await _latest_summary(session_maker, 1) is None

    async def test_off_interval_count_does_not_trigger(self, session_maker):
        # 25 = threshold + 5: between intervals, no compression
        await _seed_messages(session_maker, session_id=1, count=25)
        llm = _llm_returning("摘要")

        compressor = SessionCompressor(
            session_maker=session_maker, llm_service=llm, threshold=20, interval=10
        )
        wrote = await compressor.maybe_compress(session_id=1)

        assert wrote is False
        llm.generate.assert_not_awaited()

    async def test_at_threshold_writes_summary(self, session_maker):
        await _seed_messages(session_maker, session_id=1, count=20)
        llm = _llm_returning("用户咨询了退货，客服已受理")

        compressor = SessionCompressor(
            session_maker=session_maker, llm_service=llm, threshold=20, interval=10
        )
        wrote = await compressor.maybe_compress(session_id=1)

        assert wrote is True
        summary = await _latest_summary(session_maker, 1)
        assert summary is not None
        assert summary.role == MessageRole.SYSTEM
        assert summary.content == "用户咨询了退货，客服已受理"

    async def test_interval_crossing_writes_summary(self, session_maker):
        await _seed_messages(session_maker, session_id=1, count=30)
        llm = _llm_returning("摘要")

        compressor = SessionCompressor(
            session_maker=session_maker, llm_service=llm, threshold=20, interval=10
        )
        wrote = await compressor.maybe_compress(session_id=1)

        assert wrote is True
        assert await _latest_summary(session_maker, 1) is not None


class TestCompressionContent:
    async def test_prompt_reads_chronological_transcript(self, session_maker):
        await _seed_messages(session_maker, session_id=1, count=20)
        llm = _llm_returning("摘要")

        compressor = SessionCompressor(
            session_maker=session_maker, llm_service=llm, threshold=20, interval=10
        )
        await compressor.maybe_compress(session_id=1)

        prompt = llm.generate.await_args.kwargs["messages"][0].content
        # Repo rows come back newest-first; the transcript must read
        # oldest → newest like a chat log.
        assert prompt.index("msg00") < prompt.index("msg19")

    async def test_no_uncovered_messages_skips_llm(self, session_maker):
        """Trigger hit but everything predates is already summarized."""
        async with session_maker() as session:
            session.add(
                Message(
                    session_id=1,
                    role=MessageRole.SYSTEM,
                    content="旧摘要",
                    created_at=BASE + timedelta(seconds=10),
                )
            )
            session.add(
                Message(
                    session_id=1,
                    role=MessageRole.USER,
                    content="新消息",
                    created_at=BASE + timedelta(seconds=11),
                )
            )
            await session.commit()
        llm = _llm_returning("摘要")

        compressor = SessionCompressor(
            session_maker=session_maker, llm_service=llm, threshold=2, interval=10
        )
        wrote = await compressor.maybe_compress(session_id=1)

        assert wrote is False
        llm.generate.assert_not_awaited()


class TestCompressionResilience:
    async def test_llm_failure_never_raises(self, session_maker):
        await _seed_messages(session_maker, session_id=1, count=20)

        compressor = SessionCompressor(
            session_maker=session_maker,
            llm_service=_llm_raising(),
            threshold=20,
            interval=10,
        )
        wrote = await compressor.maybe_compress(session_id=1)

        assert wrote is False
        assert await _latest_summary(session_maker, 1) is None

    async def test_without_llm_is_a_noop(self, session_maker):
        await _seed_messages(session_maker, session_id=1, count=20)

        compressor = SessionCompressor(
            session_maker=session_maker, llm_service=None, threshold=20, interval=10
        )
        wrote = await compressor.maybe_compress(session_id=1)

        assert wrote is False
