"""TicketRepository FIFO ordering and claim/resolve transition guards."""

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.session import ChatSession  # noqa: F401 (registers table)
from app.models.database.ticket import HandoffTicket
from app.repositories.ticket_repository import TicketRepository


@pytest.fixture
async def session_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


async def _seed(session_maker, n: int) -> list[int]:
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

    async def test_status_filter_excludes_others(self, session_maker):
        ids = await _seed(session_maker, 2)
        async with session_maker() as session:
            repo = TicketRepository(session)
            await repo.claim(ids[0], agent_id=1)
            tickets = await repo.list_by_status("open")
        assert [t.id for t in tickets] == [ids[1]]


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
        assert ticket.status == "resolved"
        assert ticket.assigned_to == 1

    async def test_resolve_open_records_agent(self, session_maker):
        ids = await _seed(session_maker, 1)
        async with session_maker() as session:
            repo = TicketRepository(session)
            ticket = await repo.resolve(ids[0], agent_id=5)
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
