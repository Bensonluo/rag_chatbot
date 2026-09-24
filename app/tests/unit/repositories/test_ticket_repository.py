"""TicketRepository queue ordering (priority → FIFO) and claim/resolve guards."""

from datetime import UTC

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.session import ChatSession  # noqa: F401 (registers table)
from app.models.database.ticket import PRIORITY_HIGH, PRIORITY_NORMAL, HandoffTicket
from app.repositories.ticket_repository import TicketRepository


@pytest.fixture
async def session_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


async def _seed(session_maker: async_sessionmaker[AsyncSession], n: int) -> list[int]:
    """Create n open tickets, returning their ids."""
    ids: list[int] = []
    async with session_maker() as session:
        repo = TicketRepository(session)
        for i in range(n):
            created = await repo.create(
                HandoffTicket(session_id=100 + i, reason="explicit", summary="{}")
            )
            ids.append(created.id)
    return ids


class TestListByStatus:
    async def test_lists_open_tickets_fifo(self, session_maker):
        ids = await _seed(session_maker, 3)
        async with session_maker() as session:
            repo = TicketRepository(session)
            tickets = await repo.list_by_status("open")
        assert [t.id for t in tickets] == ids

    async def test_high_priority_created_later_is_served_first(self, session_maker):
        """Emotion escalation jumps the queue ahead of earlier explicit tickets."""
        ids = await _seed(session_maker, 2)
        async with session_maker() as session:
            repo = TicketRepository(session)
            high = await repo.create(
                HandoffTicket(
                    session_id=999,
                    reason="emotion",
                    priority=PRIORITY_HIGH,
                    summary="{}",
                )
            )
            tickets = await repo.list_by_status("open")
        assert [t.id for t in tickets] == [high.id, ids[0], ids[1]]

    async def test_fifo_within_same_priority_tier(self, session_maker):
        async with session_maker() as session:
            repo = TicketRepository(session)
            a = await repo.create(
                HandoffTicket(session_id=1, reason="emotion", priority=PRIORITY_HIGH, summary="{}")
            )
            b = await repo.create(
                HandoffTicket(session_id=2, reason="emotion", priority=PRIORITY_HIGH, summary="{}")
            )
            tickets = await repo.list_by_status("open")
        assert [t.id for t in tickets] == [a.id, b.id]

    async def test_default_priority_is_normal(self, session_maker):
        await _seed(session_maker, 1)
        async with session_maker() as session:
            repo = TicketRepository(session)
            ticket = (await repo.list_by_status("open"))[0]
        assert ticket.priority == PRIORITY_NORMAL

    async def test_status_filter_excludes_others(self, session_maker):
        ids = await _seed(session_maker, 2)
        async with session_maker() as session:
            repo = TicketRepository(session)
            await repo.claim(ids[0], agent_id=1)
            tickets = await repo.list_by_status("open")
        assert [t.id for t in tickets] == [ids[1]]


class TestPositionInQueue:
    async def test_high_priority_later_ticket_positions_first(self, session_maker):
        ids = await _seed(session_maker, 1)
        async with session_maker() as session:
            repo = TicketRepository(session)
            high = await repo.create(
                HandoffTicket(
                    session_id=999,
                    reason="emotion",
                    priority=PRIORITY_HIGH,
                    summary="{}",
                )
            )
            assert await repo.position_in_queue(high.id) == 1
            assert await repo.position_in_queue(ids[0]) == 2

    async def test_none_after_claim(self, session_maker):
        """Claimed tickets leave the waiting queue; position is undefined."""
        ids = await _seed(session_maker, 2)
        async with session_maker() as session:
            repo = TicketRepository(session)
            await repo.claim(ids[0], agent_id=1)
            assert await repo.position_in_queue(ids[0]) is None
            assert await repo.position_in_queue(ids[1]) == 1

    async def test_none_for_unknown_id(self, session_maker):
        async with session_maker() as session:
            repo = TicketRepository(session)
            assert await repo.position_in_queue(999) is None


class TestClaim:
    async def test_claim_assigns_agent(self, session_maker):
        ids = await _seed(session_maker, 1)
        async with session_maker() as session:
            repo = TicketRepository(session)
            ticket = await repo.claim(ids[0], agent_id=42)
        assert ticket is not None
        assert ticket.status == "claimed"
        assert ticket.assigned_to == 42

    async def test_claim_open_only(self, session_maker):
        """Only an open ticket can be claimed (optimistic concurrency)."""
        ids = await _seed(session_maker, 1)
        async with session_maker() as session:
            repo = TicketRepository(session)
            first = await repo.claim(ids[0], agent_id=1)
            second = await repo.claim(ids[0], agent_id=2)
        assert first is not None
        assert second is None

    async def test_claim_missing_returns_none(self, session_maker):
        async with session_maker() as session:
            repo = TicketRepository(session)
            assert await repo.claim(999, agent_id=1) is None


class TestResolve:
    async def test_resolve_from_claimed(self, session_maker):
        ids = await _seed(session_maker, 1)
        async with session_maker() as session:
            repo = TicketRepository(session)
            await repo.claim(ids[0], agent_id=1)
            ticket = await repo.resolve(ids[0], agent_id=1)
        assert ticket is not None
        assert ticket.status == "resolved"
        assert ticket.assigned_to == 1

    async def test_resolve_open_records_agent(self, session_maker):
        ids = await _seed(session_maker, 1)
        async with session_maker() as session:
            repo = TicketRepository(session)
            ticket = await repo.resolve(ids[0], agent_id=5)
        assert ticket is not None
        assert ticket.status == "resolved"
        assert ticket.assigned_to == 5

    async def test_double_resolve_returns_none(self, session_maker):
        ids = await _seed(session_maker, 1)
        async with session_maker() as session:
            repo = TicketRepository(session)
            await repo.resolve(ids[0], agent_id=5)
            assert await repo.resolve(ids[0], agent_id=5) is None


class TestSessionLookup:
    async def test_open_ticket_for_session_finds_unresolved(self, session_maker):
        await _seed(session_maker, 1)
        async with session_maker() as session:
            repo = TicketRepository(session)
            ticket = await repo.get_open_ticket_for_session(100)
            assert ticket is not None

            await repo.claim(ticket.id, agent_id=1)
            still_open = await repo.get_open_ticket_for_session(100)
            assert still_open is not None  # claimed ≠ gone

            await repo.resolve(ticket.id, agent_id=1)
            gone = await repo.get_open_ticket_for_session(100)
            assert gone is None  # resolved = no longer open

    async def test_count_open(self, session_maker):
        await _seed(session_maker, 3)
        async with session_maker() as session:
            repo = TicketRepository(session)
            await repo.resolve((await repo.list_by_status("open"))[0].id, agent_id=1)
            assert await repo.count_open() == 2


class TestQueueWaitQueries:
    """Queue-wait SLA dimensions (GB/T 47746-2026 / HollyCRM <30s target):
    counting must be COUNT queries (never materialized rows), and the
    oldest open ticket's age must be queryable for breach detection."""

    async def test_count_by_status_matches_rows(self, session_maker):
        ids = await _seed(session_maker, 2)
        async with session_maker() as session:
            repo = TicketRepository(session)
            await repo.claim(ids[0], agent_id=7)
        async with session_maker() as session:
            repo = TicketRepository(session)
            assert await repo.count_by_status("open") == 1
            assert await repo.count_by_status("claimed") == 1
            assert await repo.count_by_status("resolved") == 0

    async def test_oldest_open_created_at_returns_earliest(self, session_maker):
        ids = await _seed(session_maker, 2)
        async with session_maker() as session:
            repo = TicketRepository(session)
            oldest = await repo.oldest_open_created_at()
            first = await repo.get_by_id(ids[0])
        assert oldest is not None
        assert first is not None and first.created_at is not None
        assert oldest <= first.created_at

    async def test_oldest_open_ignores_resolved(self, session_maker):
        ids = await _seed(session_maker, 2)
        async with session_maker() as session:
            repo = TicketRepository(session)
            await repo.resolve(ids[0], agent_id=1)
            oldest = await repo.oldest_open_created_at()
            second = await repo.get_by_id(ids[1])
        assert oldest is not None
        assert second is not None and second.created_at is not None
        assert oldest >= second.created_at  # the resolved one can never win

    async def test_count_open_older_than_bounds(self, session_maker):
        from datetime import datetime, timedelta

        ids = await _seed(session_maker, 2)
        backdated = datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=120)
        async with session_maker() as session:
            ticket = await session.get(HandoffTicket, ids[0])
            ticket.created_at = backdated
            await session.commit()
        async with session_maker() as session:
            repo = TicketRepository(session)
            assert await repo.count_open_older_than(60) == 1
            assert await repo.count_open_older_than(300) == 0
