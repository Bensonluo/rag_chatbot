"""Real database interleavings for competing handoff state transitions."""

import asyncio
import os
from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.models.database.user import User
from app.models.enums.message import MessageRole
from app.repositories.ticket_repository import TicketRepository
from app.services.chat.coordination import ConversationBusyError, SessionCoordinator
from app.services.handoff.service import HandoffService


@pytest.fixture
async def ticket_database(tmp_path):
    # Separate connections to a file exercise committed state across sessions,
    # unlike a shared in-memory SQLite connection. Opt-in external runs must
    # use a disposable scratch database; never use the application's DATABASE_URL.
    url = os.environ.get("RAG_TEST_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'tickets.db'}")
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        async with maker() as session:
            # PostgreSQL enforces parent FKs; create real parent rows and unique
            # identities so repeated scratch runs do not share ticket state.
            user = User(email=f"ticket-{uuid4().hex}@example.test", hashed_password="test-only")
            session.add(user)
            await session.flush()
            chat = ChatSession(user_id=user.id)
            session.add(chat)
            await session.flush()
            ticket = await TicketRepository(session).create(
                HandoffTicket(session_id=chat.id, reason="explicit", summary="{}")
            )
            ticket_id = ticket.id
        yield maker, ticket_id
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    ("delayed_action", "winning_action", "delayed_succeeds", "final_status"),
    [
        ("claim", "claim", False, "claimed"),
        ("claim", "resolve", False, "resolved"),
        ("resolve", "claim", True, "resolved"),
        ("resolve", "resolve", False, "resolved"),
    ],
)
async def test_competing_transitions(
    ticket_database, delayed_action, winning_action, delayed_succeeds, final_status
):
    maker, ticket_id = ticket_database
    decision_reached = asyncio.Event()
    winner_finished = asyncio.Event()

    async with maker() as delayed_session, maker() as winning_session:
        # Keep a stale identity-map entry alive as well: a returned transition
        # must refresh it rather than exposing the pre-race state to the caller.
        stale_ticket = await TicketRepository(delayed_session).get_by_id(ticket_id)
        assert stale_ticket is not None
        assert stale_ticket.status == "open"
        original_execute = delayed_session.execute

        async def pause_at_decision(statement, *args, **kwargs):
            # A read-then-write implementation can read the old status before
            # its competitor commits. Atomic writes must check the status at
            # the write itself, so let the competitor commit before that write.
            if statement.is_select:
                result = await original_execute(statement, *args, **kwargs)
                decision_reached.set()
                await winner_finished.wait()
                return result
            decision_reached.set()
            await winner_finished.wait()
            return await original_execute(statement, *args, **kwargs)

        async def winning_transition() -> HandoffTicket | None:
            await decision_reached.wait()
            try:
                repo = TicketRepository(winning_session)
                operation = repo.claim if winning_action == "claim" else repo.resolve
                return await operation(ticket_id, agent_id=202)
            finally:
                winner_finished.set()

        with patch.object(delayed_session, "execute", side_effect=pause_at_decision):
            operation = getattr(TicketRepository(delayed_session), delayed_action)
            delayed, winner = await asyncio.wait_for(
                asyncio.gather(operation(ticket_id, agent_id=101), winning_transition()),
                timeout=5,
            )

        assert winner is not None
        assert (delayed is not None) == delayed_succeeds
        if delayed is not None:
            assert delayed.status == final_status
            assert delayed.assigned_to == 202
            assert delayed.claimed_at == winner.claimed_at
        winning_claimed_at = winner.claimed_at

    async with maker() as observer:
        persisted = await TicketRepository(observer).get_by_id(ticket_id)
        assert persisted is not None
        assert persisted.status == final_status
        assert persisted.assigned_to == 202
        assert persisted.claimed_at == winning_claimed_at
        if winning_action == "claim":
            assert persisted.claimed_at is not None
        else:
            assert persisted.claimed_at is None


async def test_transcript_read_serializes_with_resolve_on_postgresql(ticket_database):
    maker, ticket_id = ticket_database
    engine = maker.kw["bind"]
    if engine.dialect.name != "postgresql":
        pytest.skip("SQLite does not support row-level FOR UPDATE locks")

    async with maker() as session:
        ticket = await TicketRepository(session).claim(ticket_id, agent_id=42)
        assert ticket is not None
        session.add(Message(session_id=ticket.session_id, role=MessageRole.USER, content="Help"))
        await session.commit()

    locked = asyncio.Event()
    finish_read = asyncio.Event()
    resolve_attempted = asyncio.Event()
    original_read = TicketRepository.get_history_messages

    async def pause_read(repo, *args, **kwargs):
        rows = await original_read(repo, *args, **kwargs)
        locked.set()
        await finish_read.wait()
        return rows

    async with maker() as resolver:
        original_execute = resolver.execute

        async def observe_resolve(statement, *args, **kwargs):
            resolve_attempted.set()
            return await original_execute(statement, *args, **kwargs)

        with (
            patch.object(TicketRepository, "get_history_messages", pause_read),
            patch.object(resolver, "execute", side_effect=observe_resolve),
        ):
            reading = asyncio.create_task(HandoffService(maker).get_ticket_history(ticket_id, 42))
            resolving = None
            try:
                await asyncio.wait_for(locked.wait(), timeout=5)
                resolving = asyncio.create_task(TicketRepository(resolver).resolve(ticket_id, 42))
                await asyncio.wait_for(resolve_attempted.wait(), timeout=5)
                # The UPDATE has started but cannot commit while the read
                # holds its ticket lock. Shield keeps the pending write alive.
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(resolving), timeout=0.1)
            finally:
                finish_read.set()
                result = await asyncio.wait_for(reading, timeout=5)
                if resolving is not None:
                    resolved = await asyncio.wait_for(resolving, timeout=5)

    assert result.messages[0].content == "Help"
    assert resolved is not None
    assert resolved.status == "resolved"
    with pytest.raises(LookupError, match="^Ticket history not found$"):
        await HandoffService(maker).get_ticket_history(ticket_id, 42)


async def test_conversation_gate_coordinates_independent_connections(ticket_database):
    maker, ticket_id = ticket_database
    engine = maker.kw["bind"]
    # PostgreSQL runs in CI against its disposable URL. The advisory lock
    # connection pool must be separate from nested business transactions.
    lock_engine = create_async_engine(engine.url, pool_size=1, max_overflow=0)
    other_lock_engine = create_async_engine(engine.url, pool_size=1, max_overflow=0)
    locks = async_sessionmaker(lock_engine, expire_on_commit=False)
    other_locks = async_sessionmaker(other_lock_engine, expire_on_commit=False)
    first, second = SessionCoordinator(maker, locks), SessionCoordinator(maker, other_locks)
    async with maker() as db:
        ticket = await db.get(HandoffTicket, ticket_id)
        session_id = ticket.session_id
    try:
        async with first.hold(session_id):
            with pytest.raises(ConversationBusyError):
                async with second.hold(session_id):
                    pytest.fail("competing conversation action entered")
            # Claim is gated by the same conversation while ordinary reads
            # and durable writes retain access to the business connection pool.
            with pytest.raises(ConversationBusyError):
                await HandoffService(maker, coordinator=second).claim_ticket(ticket_id, 42)
            state = await second.state(session_id)
            assert state.claimed_ticket_id is None
            async with maker() as db:
                db.add(Message(session_id=session_id, role=MessageRole.USER, content="Gate held"))
                await db.commit()
        claimed = await HandoffService(maker, coordinator=second).claim_ticket(ticket_id, 42)
        assert claimed["status"] == "claimed"
        assert (await first.state(session_id)).claimed_ticket_id == ticket_id
    finally:
        await lock_engine.dispose()
        await other_lock_engine.dispose()
