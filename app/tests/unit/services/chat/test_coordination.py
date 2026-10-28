"""Conversation gates release on failure and keep runtime queries independent."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.services.chat import coordination as module
from app.services.chat.coordination import (
    ConversationAccessError,
    ConversationBusyError,
    CoordinationUnavailableError,
    SessionCoordinator,
)


@pytest.fixture
async def local_gate():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield maker, SessionCoordinator(maker), SessionCoordinator(maker)
    finally:
        await engine.dispose()


async def test_independent_instances_share_a_gate_without_blocking_other_sessions(local_gate):
    _, first, second = local_gate
    async with first.hold(11):
        with pytest.raises(ConversationBusyError):
            async with second.hold(11):
                pytest.fail("same conversation entered twice")
        async with second.hold(12):
            pass
    async with second.hold(11):
        pass


async def test_cancellation_and_application_error_release_local_gate(local_gate):
    _, first, second = local_gate
    entered = asyncio.Event()

    async def turn() -> None:
        async with first.hold(11):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(turn())
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with second.hold(11):
        pass
    failure = RuntimeError("application failure")
    with pytest.raises(RuntimeError) as caught:
        async with first.hold(11):
            raise failure
    assert caught.value is failure
    async with second.hold(11):
        pass


async def test_state_is_fresh_and_derives_checkpoint_generation(local_gate):
    maker, coordinator, _ = local_gate
    with pytest.raises(ConversationAccessError):
        await coordinator.state(11)
    async with maker() as db:
        db.add(ChatSession(id=11, user_id=7, session_metadata={"access_mode": "private"}))
        await db.commit()
    initial = await coordinator.state(11)
    assert (initial.owner_id, initial.is_private, initial.claimed_ticket_id) == (7, True, None)
    assert initial.thread_id(11) == "11"
    async with maker() as db:
        chat = await db.get(ChatSession, 11)
        chat.automation_epoch = 2
        db.add(
            HandoffTicket(id=8, session_id=11, reason="explicit", summary="{}", status="claimed")
        )
        await db.commit()
    current = await coordinator.state(11)
    assert current.claimed_ticket_id == 8
    assert current.thread_id(11) == "11:automation:2"


@pytest.mark.parametrize("value", [0, -1, True, 2_147_483_648])
async def test_invalid_session_never_enters_gate(local_gate, value):
    _, coordinator, _ = local_gate
    with pytest.raises(ValueError):
        async with coordinator.hold(value):
            pytest.fail("invalid conversation entered")


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_invalid_acquisition_budget_is_rejected(value):
    with pytest.raises(ValueError):
        SessionCoordinator(lambda: None, dialect="postgresql", lock_timeout_seconds=value)


class LockDatabase:
    """Only the advisory-lock transport is simulated; exercises real gate flow."""

    def __init__(self, result=True, failure=None) -> None:
        self.scalar = AsyncMock(return_value=result, side_effect=failure)
        self.released = False
        self.rolled_back = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    @asynccontextmanager
    async def begin(self):
        try:
            yield
        except BaseException:
            self.rolled_back = True
            raise
        finally:
            self.released = True


async def test_pg_busy_does_not_enter_body_and_finishes_transaction():
    db = LockDatabase(result=False)
    coordinator = SessionCoordinator(lambda: db, dialect="postgresql")
    with pytest.raises(ConversationBusyError):
        async with coordinator.hold(11):
            pytest.fail("busy conversation entered")
    assert db.released and db.rolled_back


async def test_pg_lock_transport_failure_is_unavailable_but_body_error_is_preserved():
    db = LockDatabase(failure=OSError("connection lost"))
    coordinator = SessionCoordinator(lambda: db, dialect="postgresql")
    with pytest.raises(CoordinationUnavailableError):
        async with coordinator.hold(11):
            pytest.fail("unavailable conversation entered")
    assert db.released
    db.scalar.side_effect = None
    error = ValueError("business validation")
    with pytest.raises(ValueError) as caught:
        async with coordinator.hold(11):
            raise error
    assert caught.value is error
    assert db.rolled_back


async def test_pg_acquisition_timeout_and_body_cancellation_release_transaction():
    db = LockDatabase()

    async def never(*args, **kwargs):
        await asyncio.Event().wait()

    db.scalar.side_effect = never
    coordinator = SessionCoordinator(lambda: db, dialect="postgresql", lock_timeout_seconds=0.01)
    with pytest.raises(CoordinationUnavailableError):
        async with coordinator.hold(11):
            pytest.fail("timed out conversation entered")
    assert db.released
    db.scalar.side_effect = None
    entered = asyncio.Event()

    async def turn() -> None:
        async with coordinator.hold(11):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(turn())
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert db.released and db.rolled_back


async def test_production_pg_factory_uses_a_separate_bounded_lock_pool(monkeypatch):
    from app.api import database

    runtime = Mock(kw={"bind": SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))})
    engine = SimpleNamespace(dispose=AsyncMock())
    created = []

    def create(url, **options):
        created.append(options)
        return engine

    def locks():
        return None

    monkeypatch.setattr(database, "async_session_maker", runtime)
    monkeypatch.setattr(module, "_default", None)
    monkeypatch.setattr(module, "_lock_engine", None)
    monkeypatch.setattr(module, "create_async_engine", create)
    monkeypatch.setattr(module, "async_sessionmaker", lambda *args, **kwargs: locks)
    coordinator = module.get_session_coordinator()
    assert coordinator.sessions is runtime
    assert coordinator._lock_sessions is locks
    assert module.get_session_coordinator() is coordinator
    assert len(created) == 1
    assert created[0]["max_overflow"] == 0 and created[0]["pool_timeout"] == 1.0
    await module.close_session_coordinator()
    engine.dispose.assert_awaited_once()
    assert module._default is None and module._lock_engine is None
