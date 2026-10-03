"""QA review queue: sampling the containment numerator's blind spot.

Containment counts a session as contained when it has no ticket and no
downvote — but a user who gave up silently also has neither (review
2026-09-26, #11: 用户没解决就离开，也可能被算作成功). The review
queue samples exactly that inflation-risk population: served in
window, contained, and with NO positive resolution evidence
(resolved_at IS NULL, not yet QA-reviewed). The reviewer's verdict
closes the loop — confirmed resolutions join sessions_resolved via the
same resolved_at stamp the user-confirmation path writes.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.models.enums.message import MessageRole
from app.repositories.session_repository import SessionRepository

_NOW = datetime(2026, 10, 3, 0, 0, tzinfo=UTC)


@pytest.fixture
async def session_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


async def _seed(
    session_maker: async_sessionmaker[AsyncSession],
    *,
    sessions: list[int],
    assistant_turns: dict[int, int],
    user_turns: dict[int, int] | None = None,
    ticket_sessions: list[int] | None = None,
    downvoted_sessions: set[int] | frozenset[int] = frozenset(),
    resolved_sessions: set[int] | frozenset[int] = frozenset(),
    reviewed_sessions: set[int] | frozenset[int] = frozenset(),
    old_sessions: set[int] | frozenset[int] = frozenset(),
) -> None:
    """Seed sessions/turns/tickets; reviewed = qa_verdict in metadata."""
    async with session_maker() as session:
        for sid in sessions:
            session.add(
                ChatSession(
                    id=sid,
                    user_id=1,
                    resolved_at=_NOW if sid in resolved_sessions else None,
                    session_metadata=(
                        {"qa_verdict": {"resolved": True}} if sid in reviewed_sessions else None
                    ),
                )
            )
        await session.flush()
        for sid, turns in assistant_turns.items():
            for n in range(turns):
                session.add(
                    Message(
                        session_id=sid,
                        role=MessageRole.ASSISTANT,
                        content=f"a{n}",
                        created_at=_NOW - timedelta(days=10) if sid in old_sessions else _NOW,
                        user_rating=-1 if sid in downvoted_sessions else None,
                    )
                )
        for sid, turns in (user_turns or {}).items():
            for n in range(turns):
                session.add(
                    Message(
                        session_id=sid,
                        role=MessageRole.USER,
                        content=f"用户消息{n}",
                        created_at=_NOW,
                    )
                )
        for sid in ticket_sessions or []:
            session.add(HandoffTicket(session_id=sid, reason="explicit", summary="{}"))
        await session.commit()


async def _queue(session_maker: async_sessionmaker[AsyncSession], limit: int = 10) -> set[int]:
    async with session_maker() as session:
        repo = SessionRepository(session)
        rows = await repo.sample_review_queue(since=_NOW - timedelta(days=7), limit=limit)
        return {row.id for row in rows}


class TestSampleReviewQueue:
    async def test_queue_targets_contained_unresolved_only(self, session_maker):
        """The population is precise: plain contained-not-resolved (1).
        Resolved (2) has positive evidence; ticketed (3) / downvoted (4)
        already failed; reviewed (5) is done; never-served (6) is noise."""
        await _seed(
            session_maker,
            sessions=[1, 2, 3, 4, 5, 6],
            assistant_turns={1: 1, 2: 1, 3: 1, 4: 1, 5: 1},
            ticket_sessions=[3],
            downvoted_sessions={4},
            resolved_sessions={2},
            reviewed_sessions={5},
        )

        assert await _queue(session_maker) == {1}

    async def test_window_excludes_stale_sessions(self, session_maker):
        await _seed(
            session_maker,
            sessions=[1],
            assistant_turns={1: 1},
            old_sessions={1},
        )

        assert await _queue(session_maker) == set()

    async def test_limit_draws_subset_of_candidates(self, session_maker):
        await _seed(
            session_maker,
            sessions=[1, 2, 3, 4],
            assistant_turns={1: 1, 2: 1, 3: 1, 4: 1},
        )

        sampled = await _queue(session_maker, limit=2)

        assert len(sampled) == 2
        assert sampled <= {1, 2, 3, 4}

    async def test_empty_pool_returns_empty(self, session_maker):
        assert await _queue(session_maker) == set()


class TestRecordReviewVerdict:
    async def test_verdict_resolved_true_stamps_resolved_at(self, session_maker):
        await _seed(session_maker, sessions=[1], assistant_turns={1: 1})

        async with session_maker() as session:
            repo = SessionRepository(session)
            row = await repo.record_review_verdict(1, resolved=True, note="问题清晰，确认解决")

        assert row is not None
        assert row.resolved_at is not None
        assert row.session_metadata is not None
        assert row.session_metadata["qa_verdict"]["resolved"] is True
        # Positive evidence recorded → leaves the queue.
        assert await _queue(session_maker) == set()

    async def test_verdict_resolved_false_records_metadata_only(self, session_maker):
        await _seed(session_maker, sessions=[1], assistant_turns={1: 1})

        async with session_maker() as session:
            repo = SessionRepository(session)
            row = await repo.record_review_verdict(1, resolved=False, note="机器人答非所问")

        assert row is not None
        assert row.resolved_at is None  # not a resolution — no KPI inflation
        assert row.session_metadata is not None
        assert row.session_metadata["qa_verdict"]["resolved"] is False
        assert row.session_metadata["qa_verdict"]["note"] == "机器人答非所问"
        assert await _queue(session_maker) == set()

    async def test_verdict_missing_session_returns_none(self, session_maker):
        async with session_maker() as session:
            repo = SessionRepository(session)
            assert await repo.record_review_verdict(999, resolved=True) is None

    async def test_verdict_overwrites_prior_verdict(self, session_maker):
        await _seed(session_maker, sessions=[1], assistant_turns={1: 1})

        async with session_maker() as session:
            repo = SessionRepository(session)
            await repo.record_review_verdict(1, resolved=False, note="先判未解决")
            row = await repo.record_review_verdict(1, resolved=True, note="复核后确认")

        assert row is not None
        assert row.resolved_at is not None
        assert row.session_metadata is not None
        assert row.session_metadata["qa_verdict"]["note"] == "复核后确认"
        assert len(row.session_metadata) == 1  # one verdict key, not accumulating


class TestLastUserMessage:
    async def test_returns_latest_user_message(self, session_maker):
        await _seed(
            session_maker,
            sessions=[1],
            assistant_turns={1: 1},
            user_turns={1: 3},
        )

        async with session_maker() as session:
            repo = SessionRepository(session)
            assert await repo.last_user_message(1) == "用户消息2"

    async def test_session_without_user_messages_returns_none(self, session_maker):
        await _seed(session_maker, sessions=[1], assistant_turns={1: 1})

        async with session_maker() as session:
            repo = SessionRepository(session)
            assert await repo.last_user_message(1) is None
