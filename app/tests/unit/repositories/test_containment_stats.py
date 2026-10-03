"""Session-level containment stats (north star, highest weight).

The only computable containment signal was the turn-level handoff share
(a proxy): a 5-turn session with one handoff reads 20% there, while
the honest session-level answer is 0% — that session was not
contained. This pins the real metric at the real granularity:

- denominator: sessions the bot actually served (≥1 assistant
  message) inside the window
- numerator: served sessions with NO handoff ticket (any ticket
  status — a resolved ticket still means a human was pulled in) and
  NO downvoted assistant answer in-window (a thumbs-down is the
  operational proxy for "not a resolution" — counting those sessions
  as containment inflates the north star)
- tickets for never-served sessions count nowhere (a handoff without
  a served turn is queue noise, not a resolution outcome)

Naming (review 2026-09-26, #11): containment — the bot served the
session with no escalation and no downvote — is what this measures;
it is NOT verified problem resolution.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.models.enums.message import MessageRole
from app.repositories.ticket_repository import TicketRepository

_NOW = datetime(2026, 9, 25, 0, 0, tzinfo=UTC)


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
    ticket_sessions: list[int],
    old_sessions: set[int] | frozenset[int] = frozenset(),
    downvoted_sessions: set[int] | frozenset[int] = frozenset(),
    upvoted_sessions: set[int] | frozenset[int] = frozenset(),
    resolved_sessions: set[int] | frozenset[int] = frozenset(),
    reopened_counts: dict[int, int] | None = None,
) -> None:
    """Seed chat sessions, assistant turns, and handoff tickets.

    ``old_sessions`` shifts that session's assistant turns before the
    stats window so window filtering is exercised. Rating params set
    ``user_rating`` on every assistant turn of that session.
    ``resolved_sessions`` pins resolved_at to now (review #11); a
    reopened session is the CURRENT state — resolved_at=None with
    reopened_count>0 — so seed it via ``reopened_counts`` only.
    """
    async with session_maker() as session:
        for sid in sessions:
            session.add(
                ChatSession(
                    id=sid,
                    user_id=1,
                    resolved_at=_NOW if sid in resolved_sessions else None,
                    reopened_count=(reopened_counts or {}).get(sid, 0),
                )
            )
        await session.flush()
        for sid, turns in assistant_turns.items():
            for n in range(turns):
                created_at = _NOW - timedelta(days=10) if sid in old_sessions else _NOW
                rating = (
                    -1 if sid in downvoted_sessions else (1 if sid in upvoted_sessions else None)
                )
                session.add(
                    Message(
                        session_id=sid,
                        role=MessageRole.ASSISTANT,
                        content=f"a{n}",
                        created_at=created_at,
                        user_rating=rating,
                    )
                )
        for sid in ticket_sessions:
            session.add(HandoffTicket(session_id=sid, reason="explicit", summary="{}"))
        await session.commit()


async def _stats(session_maker: async_sessionmaker[AsyncSession]) -> dict[str, Any]:
    async with session_maker() as session:
        repo = TicketRepository(session)
        return await repo.get_containment_stats(since=_NOW - timedelta(days=7))


class TestContainmentStats:
    async def test_rate_excludes_escalated_sessions(self, session_maker):
        await _seed(
            session_maker,
            sessions=[1, 2, 3],
            assistant_turns={1: 1, 2: 1, 3: 1},
            ticket_sessions=[2],
        )

        stats = await _stats(session_maker)

        assert stats["sessions_served"] == 3
        assert stats["sessions_escalated"] == 1
        assert stats["containment_rate"] == pytest.approx(2 / 3, abs=1e-4)

    async def test_window_excludes_stale_served_sessions(self, session_maker):
        await _seed(
            session_maker,
            sessions=[1, 2],
            assistant_turns={1: 1, 2: 1},
            ticket_sessions=[],
            old_sessions={2},
        )

        stats = await _stats(session_maker)

        assert stats["sessions_served"] == 1

    async def test_ticket_without_served_turn_counts_nowhere(self, session_maker):
        """A handoff whose session has no assistant message in-window is
        queue noise, not a resolution outcome — excluded from BOTH counts."""
        await _seed(
            session_maker,
            sessions=[1, 2],
            assistant_turns={1: 1},
            ticket_sessions=[2],
        )

        stats = await _stats(session_maker)

        assert stats["sessions_served"] == 1
        assert stats["sessions_escalated"] == 0
        assert stats["containment_rate"] == 1.0

    async def test_empty_window_returns_none_rate(self, session_maker):
        await _seed(
            session_maker,
            sessions=[1],
            assistant_turns={1: 1},
            ticket_sessions=[],
            old_sessions={1},
        )

        stats = await _stats(session_maker)

        assert stats["sessions_served"] == 0
        assert stats["containment_rate"] is None

    async def test_multi_turn_escalated_session_is_wholly_uncontained(self, session_maker):
        """The proxy-vs-real difference in one row: five served turns and
        one ticket → 0% containment at session granularity (turn-level
        handoff share would read 20%)."""
        await _seed(
            session_maker,
            sessions=[1],
            assistant_turns={1: 5},
            ticket_sessions=[1],
        )

        stats = await _stats(session_maker)

        assert stats["sessions_served"] == 1
        assert stats["sessions_escalated"] == 1
        assert stats["containment_rate"] == 0.0

    async def test_downvoted_session_is_uncontained(self, session_maker):
        """A thumbs-down on the served answer is the operational proxy
        for "not a resolution" (same doctrine as the cache-eviction
        loop): counting this session as containment inflates the north
        star. It stays in the denominator (it WAS served) but leaves
        the numerator, surfaced as sessions_downvoted."""
        await _seed(
            session_maker,
            sessions=[1, 2],
            assistant_turns={1: 1, 2: 1},
            ticket_sessions=[],
            downvoted_sessions={1},
        )

        stats = await _stats(session_maker)

        assert stats["sessions_served"] == 2
        assert stats["sessions_escalated"] == 0
        assert stats["sessions_downvoted"] == 1
        assert stats["containment_rate"] == pytest.approx(0.5, abs=1e-4)

    async def test_upvoted_and_unrated_sessions_still_count(self, session_maker):
        """Only a negative rating excludes: NULL (unrated) and +1
        (thumbs-up) both stay in the numerator — the exclusion is the
        user's rejection signal, not the presence of any rating."""
        await _seed(
            session_maker,
            sessions=[1, 2],
            assistant_turns={1: 1, 2: 1},
            ticket_sessions=[],
            upvoted_sessions={1},
        )

        stats = await _stats(session_maker)

        assert stats["sessions_downvoted"] == 0
        assert stats["containment_rate"] == 1.0

    async def test_escalated_and_downvoted_session_is_subtracted_once(self, session_maker):
        """A session can carry both failure signals (a ticket AND a
        thumbs-down). The rate must subtract it once — sessions_
        downvoted reports the signal count, the union guards the
        numerator."""
        await _seed(
            session_maker,
            sessions=[1],
            assistant_turns={1: 2},
            ticket_sessions=[1],
            downvoted_sessions={1},
        )

        stats = await _stats(session_maker)

        assert stats["sessions_served"] == 1
        assert stats["sessions_escalated"] == 1
        assert stats["sessions_downvoted"] == 1
        assert stats["containment_rate"] == 0.0


class TestResolutionCrossTab:
    """User-confirmed resolution among served sessions (review #11).

    Containment is the absence of failure signals; sessions_resolved is
    the presence of a positive one (the user said it's solved). Both are
    reported side by side — resolved stays visible even for sessions
    that later escalated, because "user said solved then came back with
    a ticket" is exactly the case human review should see.
    """

    async def test_resolved_and_reopened_sessions_cross_tab(self, session_maker):
        await _seed(
            session_maker,
            sessions=[1, 2, 3],
            assistant_turns={1: 1, 2: 1, 3: 1},
            ticket_sessions=[],
            resolved_sessions={1},
            reopened_counts={2: 2},
        )

        stats = await _stats(session_maker)

        assert stats["sessions_served"] == 3
        assert stats["sessions_resolved"] == 1
        assert stats["sessions_reopened"] == 1
        assert stats["containment_rate"] == 1.0  # cross-tab is independent

    async def test_resolved_session_that_escalated_is_counted_in_both(self, session_maker):
        """Escalation after a user confirmation does not erase the
        confirmation — the two signals answer different questions."""
        await _seed(
            session_maker,
            sessions=[1],
            assistant_turns={1: 1},
            ticket_sessions=[1],
            resolved_sessions={1},
        )

        stats = await _stats(session_maker)

        assert stats["sessions_resolved"] == 1
        assert stats["sessions_escalated"] == 1

    async def test_resolution_fields_present_and_zero_without_any(self, session_maker):
        await _seed(
            session_maker,
            sessions=[1],
            assistant_turns={1: 1},
            ticket_sessions=[],
        )

        stats = await _stats(session_maker)

        assert stats["sessions_resolved"] == 0
        assert stats["sessions_reopened"] == 0

    async def test_resolution_ignores_never_served_and_stale_sessions(self, session_maker):
        """Same denominator discipline as the rest of the stats: only
        sessions actually served in-window count."""
        await _seed(
            session_maker,
            sessions=[1, 2, 3],
            assistant_turns={1: 1},
            ticket_sessions=[],
            old_sessions={1},
            resolved_sessions={1, 2},
            reopened_counts={3: 5},
        )

        stats = await _stats(session_maker)

        assert stats["sessions_served"] == 0
        assert stats["sessions_resolved"] == 0
        assert stats["sessions_reopened"] == 0
        assert stats["containment_rate"] is None
