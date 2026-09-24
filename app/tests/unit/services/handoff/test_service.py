"""HandoffService: ticket lifecycle and the never-raise chat contract."""

from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.session import ChatSession  # noqa: F401 (registers table)
from app.models.database.ticket import (
    PRIORITY_HIGH,
    PRIORITY_NORMAL,
    HandoffTicket,
)
from app.services.handoff.service import HandoffService


@pytest.fixture
async def session_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


class TestCreateTicket:
    async def test_creates_ticket_with_queue_position(self, session_maker):
        service = HandoffService(session_maker=session_maker)

        result = await service.create_ticket_for_session(
            session_id=1,
            user_id=7,
            reason="explicit",
            context={"user_message": "转人工", "intent": "refund"},
        )

        assert result["ticket_id"] is not None
        assert result["queue_position"] == 1
        assert result["reused"] is False

        async with session_maker() as session:
            ticket = (await session.execute(select(HandoffTicket))).scalar_one()
        assert ticket.reason == "explicit"
        assert '"intent": "refund"' in ticket.summary

    async def test_repeated_request_reuses_open_ticket(self, session_maker):
        """Saying 转人工 twice must not create a second queue entry."""
        service = HandoffService(session_maker=session_maker)

        first = await service.create_ticket_for_session(1, 7, "explicit", {})
        second = await service.create_ticket_for_session(1, 7, "explicit", {})

        assert second["ticket_id"] == first["ticket_id"]
        assert second["reused"] is True

        async with session_maker() as session:
            count = len((await session.execute(select(HandoffTicket))).scalars().all())
        assert count == 1

    async def test_queue_position_counts_open_only(self, session_maker):
        service = HandoffService(session_maker=session_maker)
        await service.create_ticket_for_session(1, 7, "explicit", {})
        await service.create_ticket_for_session(2, 8, "emotion", {})
        # Resolve the first ticket; the third request should see depth 2.
        open_tickets = await service.list_tickets("open")
        await service.resolve_ticket(open_tickets[0]["id"], agent_id=99)

        third = await service.create_ticket_for_session(3, 9, "explicit", {})

        assert third["queue_position"] == 2

    async def test_invalid_reason_falls_back_to_explicit(self, session_maker):
        service = HandoffService(session_maker=session_maker)
        result = await service.create_ticket_for_session(1, 7, "weird", {})
        assert result["ticket_id"] is not None

        async with session_maker() as session:
            ticket = (await session.execute(select(HandoffTicket))).scalar_one()
        assert ticket.reason == "explicit"

    async def test_emotion_reason_gets_high_priority(self, session_maker):
        service = HandoffService(session_maker=session_maker)
        await service.create_ticket_for_session(1, 7, "emotion", {})
        await service.create_ticket_for_session(2, 8, "explicit", {})

        async with session_maker() as session:
            tickets = {
                t.session_id: t
                for t in (await session.execute(select(HandoffTicket))).scalars().all()
            }
        assert tickets[1].priority == PRIORITY_HIGH
        assert tickets[2].priority == PRIORITY_NORMAL

    async def test_queue_position_honors_priority(self, session_maker):
        """A high-priority ticket created later reports position 1."""
        service = HandoffService(session_maker=session_maker)
        first = await service.create_ticket_for_session(1, 7, "explicit", {})
        second = await service.create_ticket_for_session(2, 8, "emotion", {})

        assert first["queue_position"] == 1  # queue was empty before it
        assert second["queue_position"] == 1  # jumps ahead of the explicit ticket

        third = await service.create_ticket_for_session(3, 9, "explicit", {})
        assert third["queue_position"] == 3

    async def test_reuse_reports_current_queue_position(self, session_maker):
        """Re-requesting a ticket returns its live position, not None."""
        service = HandoffService(session_maker=session_maker)
        await service.create_ticket_for_session(1, 7, "explicit", {})
        await service.create_ticket_for_session(2, 8, "emotion", {})

        again = await service.create_ticket_for_session(1, 7, "explicit", {})

        assert again["reused"] is True
        assert again["queue_position"] == 2  # emotion ticket jumped ahead

    async def test_anonymous_user_maps_to_zero(self, session_maker):
        service = HandoffService(session_maker=session_maker)
        await service.create_ticket_for_session(1, None, "explicit", {})

        async with session_maker() as session:
            ticket = (await session.execute(select(HandoffTicket))).scalar_one()
        assert ticket.user_id == 0

    async def test_db_failure_returns_none_ticket_not_raise(self):
        """Availability over durability: the chat turn survives DB loss."""

        def broken_maker():
            raise RuntimeError("db down")

        service = HandoffService(session_maker=broken_maker)
        result = await service.create_ticket_for_session(1, 7, "explicit", {})
        assert result == {
            "ticket_id": None,
            "queue_position": None,
            "reused": False,
        }


class TestAgentWorkspace:
    async def test_claim_and_resolve_lifecycle(self, session_maker):
        service = HandoffService(session_maker=session_maker)
        created = await service.create_ticket_for_session(1, 7, "explicit", {})

        claimed = await service.claim_ticket(created["ticket_id"], agent_id=99)
        assert claimed["status"] == "claimed"
        assert claimed["assigned_to"] == 99

        resolved = await service.resolve_ticket(created["ticket_id"], agent_id=99)
        assert resolved["status"] == "resolved"

    async def test_second_claim_conflicts(self, session_maker):
        service = HandoffService(session_maker=session_maker)
        created = await service.create_ticket_for_session(1, 7, "explicit", {})
        await service.claim_ticket(created["ticket_id"], agent_id=99)

        with pytest.raises(LookupError):
            await service.claim_ticket(created["ticket_id"], agent_id=100)

    async def test_resolve_missing_ticket_raises(self, session_maker):
        service = HandoffService(session_maker=session_maker)
        with pytest.raises(LookupError):
            await service.resolve_ticket(999, agent_id=99)

    async def test_list_orders_by_priority_then_fifo(self, session_maker):
        """High-priority (emotion) escalations are served before older explicit ones."""
        service = HandoffService(session_maker=session_maker)
        first = await service.create_ticket_for_session(1, 7, "explicit", {})
        second = await service.create_ticket_for_session(2, 8, "emotion", {})
        third = await service.create_ticket_for_session(3, 9, "explicit", {})

        tickets = await service.list_tickets("open")
        assert [t["id"] for t in tickets] == [
            second["ticket_id"],  # priority 1 despite being created second
            first["ticket_id"],  # FIFO within the normal tier
            third["ticket_id"],
        ]
        assert tickets[0]["priority"] == PRIORITY_HIGH
        assert tickets[1]["priority"] == PRIORITY_NORMAL

    async def test_queue_stats_counts_each_status(self, session_maker):
        service = HandoffService(session_maker=session_maker)
        a = await service.create_ticket_for_session(1, 7, "explicit", {})
        await service.create_ticket_for_session(2, 8, "emotion", {})
        await service.claim_ticket(a["ticket_id"], agent_id=99)

        stats = await service.queue_stats()
        assert stats == {"open": 1, "claimed": 1, "resolved": 0}


class TestTicketSummary:
    """Ticket creation enriches context with an LLM conversation summary."""

    async def _seed_messages(self, session_maker: async_sessionmaker[Any]) -> None:
        from app.models.database.message import Message
        from app.models.enums.message import MessageRole

        async with session_maker() as session:
            session.add(Message(session_id=1, role=MessageRole.USER, content="订单 12345 未到"))
            session.add(
                Message(session_id=1, role=MessageRole.ASSISTANT, content="已查物流，运输中")
            )
            await session.commit()

    async def test_ticket_carries_conversation_summary(self, session_maker):
        from app.services.handoff.service import HandoffService

        await self._seed_messages(session_maker)
        llm = _SummaryLLM()
        service = HandoffService(session_maker=session_maker, llm_service=llm)

        await service.create_ticket_for_session(
            session_id=1, user_id=7, reason="explicit", context={"user_message": "转人工"}
        )

        async with session_maker() as session:
            ticket = (await session.execute(select(HandoffTicket))).scalar_one()
        assert '"conversation_summary": "用户诉求：订单 12345 未送达。"' in ticket.summary
        assert llm.calls == 1

    async def test_llm_failure_still_creates_ticket(self, session_maker):
        from app.services.handoff.service import HandoffService

        await self._seed_messages(session_maker)
        service = HandoffService(session_maker=session_maker, llm_service=_SummaryLLM(error=True))

        result = await service.create_ticket_for_session(
            session_id=1, user_id=7, reason="explicit", context={}
        )

        assert result["ticket_id"] is not None
        async with session_maker() as session:
            ticket = (await session.execute(select(HandoffTicket))).scalar_one()
        assert "conversation_summary" not in ticket.summary

    async def test_reuse_path_skips_llm(self, session_maker):
        from app.services.handoff.service import HandoffService

        llm = _SummaryLLM()
        service = HandoffService(session_maker=session_maker, llm_service=llm)
        first = await service.create_ticket_for_session(1, 7, "explicit", {})

        second = await service.create_ticket_for_session(1, 7, "explicit", {})

        assert second["ticket_id"] == first["ticket_id"]
        assert second["reused"] is True
        # No messages seeded → the first call skipped the summarizer;
        # the reuse path never reaches it either.
        assert llm.calls == 0

    async def test_summary_prompt_receives_chronological_transcript(self, session_maker):
        """The summarizer transcript reads oldest → newest, like a chat log."""
        from datetime import datetime, timedelta

        from app.models.database.message import Message
        from app.models.enums.message import MessageRole

        base = datetime(2026, 1, 1, 12, 0, 0)
        async with session_maker() as session:
            session.add(
                Message(
                    session_id=3,
                    role=MessageRole.USER,
                    content="第一条",
                    created_at=base,
                )
            )
            session.add(
                Message(
                    session_id=3,
                    role=MessageRole.ASSISTANT,
                    content="第二条",
                    created_at=base + timedelta(seconds=1),
                )
            )
            session.add(
                Message(
                    session_id=3,
                    role=MessageRole.USER,
                    content="第三条",
                    created_at=base + timedelta(seconds=2),
                )
            )
            await session.commit()

        llm = _CapturingLLM()
        service = HandoffService(session_maker=session_maker, llm_service=llm)

        await service.create_ticket_for_session(3, 7, "explicit", {})

        assert llm.prompt.index("第一条") < llm.prompt.index("第二条")
        assert llm.prompt.index("第二条") < llm.prompt.index("第三条")


class _SummaryLLM:
    """Fake LLM for service-level tests (Protocol-compatible)."""

    def __init__(self, error: bool = False) -> None:
        self.error = error
        self.calls = 0

    async def generate(self, messages: Any, **kwargs: Any) -> Any:
        from types import SimpleNamespace

        self.calls += 1
        if self.error:
            raise RuntimeError("llm down")
        return SimpleNamespace(content="用户诉求：订单 12345 未送达。")


class _CapturingLLM:
    """Fake LLM capturing the summary prompt for order assertions."""

    def __init__(self) -> None:
        self.prompt = ""

    async def generate(self, messages: Any, **kwargs: Any) -> Any:
        from types import SimpleNamespace

        self.prompt = messages[0].content
        return SimpleNamespace(content="摘要")
