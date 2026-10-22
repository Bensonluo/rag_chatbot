"""Simultaneous handoff requests share one active ticket and retain escalation."""

import asyncio
import json
import os
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config.settings import get_settings
from app.models.database.base import Base
from app.models.database.session import ChatSession
from app.models.database.ticket import PRIORITY_HIGH, HandoffTicket
from app.models.database.user import User
from app.repositories.ticket_repository import TicketRepository
from app.services.handoff.service import HandoffService


@pytest.fixture
async def creation_db(tmp_path):
    url = os.environ.get("RAG_TEST_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'create.db'}")
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        user = User(email=f"creation-{uuid4().hex}@example.test", hashed_password="test-only")
        session.add(user)
        await session.flush()
        chat = ChatSession(user_id=user.id)
        session.add(chat)
        await session.commit()
        session_id, user_id = chat.id, user.id
    yield engine, maker, session_id, user_id
    await engine.dispose()


async def test_parallel_requests_create_one_ticket_and_keep_highest_priority(creation_db):
    engine, maker, session_id, user_id = creation_db
    both_started = asyncio.Event()
    arrived = 0

    class InterleavedSession(AsyncSession):
        first_statement = True

        async def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal arrived
            if not self.first_statement:
                return await super().execute(statement, *args, **kwargs)
            self.first_statement = False
            # Old SELECT-then-insert reads the same absence in both requests;
            # an atomic insert must arbitrate at the write instead.
            result = None
            if statement.is_select:
                result = await super().execute(statement, *args, **kwargs)
            arrived += 1
            if arrived >= 2:
                both_started.set()
            await both_started.wait()
            return (
                result if result is not None else await super().execute(statement, *args, **kwargs)
            )

    interleaved_maker = async_sessionmaker(
        engine, class_=InterleavedSession, expire_on_commit=False
    )
    service = HandoffService(interleaved_maker)
    with patch.object(get_settings(), "DEMO_MODE", False):
        results = await asyncio.wait_for(
            asyncio.gather(
                service.create_ticket_for_session(
                    session_id, user_id, "explicit", {"user_message": "human"}
                ),
                service.create_ticket_for_session(
                    session_id, user_id, "emotion", {"user_message": "still waiting"}
                ),
            ),
            timeout=10,
        )

    assert results[0]["ticket_id"] is not None
    assert results[0]["ticket_id"] == results[1]["ticket_id"]
    assert sorted(result["reused"] for result in results) == [False, True]
    async with maker() as session:
        rows = (
            (
                await session.execute(
                    select(HandoffTicket).where(HandoffTicket.session_id == session_id)
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == 1
        assert rows[0].priority == PRIORITY_HIGH
        assert rows[0].reason == "emotion"


async def test_reused_claimed_ticket_preserves_owner_and_records_escalation(creation_db):
    _, maker, session_id, user_id = creation_db
    service = HandoffService(maker)
    first = await service.create_ticket_for_session(
        session_id, user_id, "explicit", {"order_id": "DEMO-1"}
    )
    await service.claim_ticket(first["ticket_id"], agent_id=99)

    urgent = await service.create_ticket_for_session(
        session_id, user_id, "emotion", {"user_message": "太失望了"}
    )
    await service.create_ticket_for_session(
        session_id, user_id, "refund_threshold", {"amount": 2000}
    )
    again = await service.create_ticket_for_session(
        session_id, user_id, "explicit", {"user_message": "请继续处理"}
    )

    assert urgent["ticket_id"] == again["ticket_id"] == first["ticket_id"]
    async with maker() as session:
        row = await session.get(HandoffTicket, first["ticket_id"])
        assert row is not None
        assert row.priority == PRIORITY_HIGH
        assert row.reason == "emotion"
        assert row.status == "claimed"
        assert row.assigned_to == 99
        assert row.claimed_at is not None
        context = json.loads(row.summary)
        assert context["order_id"] == "DEMO-1"
        assert context["escalation_contexts"]["emotion"]["context"]["user_message"] == "太失望了"
        assert context["escalation_contexts"]["refund_threshold"]["context"]["amount"] == 2000
        assert context["latest_handoff_request"]["context"]["user_message"] == "请继续处理"


async def test_resolved_ticket_does_not_prevent_a_new_handoff(creation_db):
    _, maker, session_id, user_id = creation_db
    service = HandoffService(maker)
    first = await service.create_ticket_for_session(session_id, user_id, "explicit", {})
    await service.resolve_ticket(first["ticket_id"], agent_id=99)
    second = await service.create_ticket_for_session(session_id, user_id, "explicit", {})
    assert second["ticket_id"] is not None
    assert second["ticket_id"] != first["ticket_id"]
    assert second["reused"] is False


async def test_concurrent_close_between_conflict_and_reuse_creates_fresh_ticket(creation_db):
    engine, maker, session_id, user_id = creation_db
    if engine.dialect.name != "postgresql":
        pytest.skip("SQLite serializes writers; this interleaving requires PostgreSQL")
    first = await HandoffService(maker).create_ticket_for_session(
        session_id, user_id, "explicit", {}
    )
    async with maker() as contender, maker() as closer:
        original_execute = contender.execute
        closed = False

        async def close_before_locked_read(statement: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal closed
            # The insert conflicted, but the existing ticket can close before
            # the next statement's snapshot locks and returns the active row.
            if statement.is_select and not closed:
                closed = True
                assert await TicketRepository(closer).resolve(first["ticket_id"], 99) is not None
            return await original_execute(statement, *args, **kwargs)

        with patch.object(contender, "execute", side_effect=close_before_locked_read):
            ticket, reused = await asyncio.wait_for(
                TicketRepository(contender).create_or_reuse_active(
                    session_id=session_id,
                    user_id=user_id,
                    reason="explicit",
                    priority=2,
                    context={"user_message": "I still need help"},
                ),
                timeout=5,
            )
        assert closed
        assert not reused
        assert ticket.id != first["ticket_id"]
        assert ticket.status == "open"
    async with maker() as session:
        rows = (
            (
                await session.execute(
                    select(HandoffTicket)
                    .where(HandoffTicket.session_id == session_id)
                    .order_by(HandoffTicket.id)
                )
            )
            .scalars()
            .all()
        )
        assert [row.status for row in rows] == ["resolved", "open"]
