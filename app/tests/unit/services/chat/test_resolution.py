"""Session resolution state (review 2026-09-26, #11 resolved slice).

Containment (b3341f0) is honest about what it is: the ABSENCE of
failure signals, not a verified resolution. This file pins the other
half of the review's ask — an explicit resolved status on the session,
a user confirmation signal (high-precision "problem solved" phrasing
in the user's own message; plain thanks is NOT resolution), and
reopen detection (a non-confirmation message arriving on a resolved
session flips it open and counts the reopen, so "同一问题短期重开"
becomes measurable instead of guessed).

Stats cross-tab (resolved/reopened among contained sessions) is
pinned in app/tests/unit/repositories/test_containment_stats.py.
"""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config.settings import get_settings
from app.models.database.base import Base
from app.models.database.session import ChatSession
from app.services.chat.persistence import ChatMessagePersister
from app.services.chat.resolution import is_resolution_confirmation


@pytest.fixture
async def session_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


async def _seed_session(session_maker: async_sessionmaker[AsyncSession], sid: int = 1) -> None:
    async with session_maker() as session:
        session.add(ChatSession(id=sid, user_id=1))
        await session.commit()


async def _get_session(
    session_maker: async_sessionmaker[AsyncSession], sid: int = 1
) -> ChatSession | None:
    async with session_maker() as session:
        result = await session.execute(select(ChatSession).where(ChatSession.id == sid))
        return result.scalar_one_or_none()


# ── Confirmation patterns: precision over recall ─────────────────────────


class TestResolutionPatterns:
    @pytest.mark.parametrize(
        "text",
        [
            "问题解决了",
            "已经解决了，谢谢",
            "搞定了",
            "没有问题了",
            "谢谢，问题解决了",
            "解决了，多谢",
            "That solved it, thanks!",
            "all set",
            "Problem solved!",
            "Thanks, that worked.",
        ],
    )
    def test_resolution_phrases_confirm(self, text: str):
        assert is_resolution_confirmation(text), text

    @pytest.mark.parametrize(
        "text",
        [
            "谢谢",
            "thanks",
            "Thanks a lot!",
            "还没解决",
            "没解决，我要投诉",
            "退货政策是什么",
            "What is your return policy?",
            "帮我查一下订单 ORD1001",
        ],
    )
    def test_non_resolution_phrases_do_not_confirm(self, text: str):
        assert not is_resolution_confirmation(text), text


# ── Persistence transitions ───────────────────────────────────────────────


class TestPersistTurnResolutionState:
    async def test_solve_phrase_marks_session_resolved(self, session_maker):
        await _seed_session(session_maker)
        persister = ChatMessagePersister(session_maker=session_maker)

        await persister.persist_turn(1, 1, "问题解决了，谢谢", "不客气，有需要随时找我。")

        row = await _get_session(session_maker)
        assert row is not None
        assert row.resolved_at is not None

    async def test_followup_message_reopens_resolved_session(self, session_maker):
        await _seed_session(session_maker)
        persister = ChatMessagePersister(session_maker=session_maker)
        await persister.persist_turn(1, 1, "问题解决了", "好的。")

        await persister.persist_turn(1, 1, "退货政策是什么", "支持七天无理由退货。")

        row = await _get_session(session_maker)
        assert row is not None
        assert row.resolved_at is None
        assert row.reopened_count == 1

    async def test_plain_thanks_does_not_resolve(self, session_maker):
        await _seed_session(session_maker)
        persister = ChatMessagePersister(session_maker=session_maker)

        await persister.persist_turn(1, 1, "谢谢", "不客气。")

        row = await _get_session(session_maker)
        assert row is not None
        assert row.resolved_at is None
        assert row.reopened_count == 0

    async def test_repeated_solve_does_not_reopen(self, session_maker):
        """A confirmation arriving on an already-resolved session is a
        re-confirmation, never a reopen (confirmation wins the branch)."""
        await _seed_session(session_maker)
        persister = ChatMessagePersister(session_maker=session_maker)

        await persister.persist_turn(1, 1, "问题解决了", "好的。")
        await persister.persist_turn(1, 1, "解决了，谢谢", "不客气。")

        row = await _get_session(session_maker)
        assert row is not None
        assert row.resolved_at is not None
        assert row.reopened_count == 0

    async def test_resolve_after_reopen_uses_fresh_timestamp(self, session_maker):
        await _seed_session(session_maker)
        persister = ChatMessagePersister(session_maker=session_maker)
        await persister.persist_turn(1, 1, "问题解决了", "好的。")
        before = datetime.now(UTC).timestamp()
        await persister.persist_turn(1, 1, "还有一个问题", "请讲。")

        await persister.persist_turn(1, 1, "这下解决了", "好的。")

        row = await _get_session(session_maker)
        assert row is not None
        assert row.resolved_at is not None
        assert row.reopened_count == 1  # the reopen from the follow-up stands
        stamp = row.resolved_at
        if stamp.tzinfo is None:  # sqlite drops tz on read-back
            stamp = stamp.replace(tzinfo=UTC)
        assert stamp.timestamp() >= before  # fresh resolve, not the stale one

    async def test_missing_session_row_is_silent(self, session_maker):
        """Resolution tracking is best-effort like the rest of the
        persister: no row (strict mode, deleted session) = no-op UPDATE.
        Demo mode would auto-create the row first — this scenario only
        exists with DEMO_MODE off."""
        with patch.object(get_settings(), "DEMO_MODE", False):
            persister = ChatMessagePersister(session_maker=session_maker)
            await persister.persist_turn(999, 1, "问题解决了", "好的。")

        assert await _get_session(session_maker, 999) is None
