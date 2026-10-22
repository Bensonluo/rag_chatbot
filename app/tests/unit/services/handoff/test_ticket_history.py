"""Persisted handoff transcripts are restricted to the ticket's current agent."""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.models.database.user import User
from app.models.enums.message import MessageRole
from app.repositories.ticket_repository import TicketRepository
from app.services.handoff.service import HandoffService


@pytest.fixture
async def history_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def seed_history(
    maker: async_sessionmaker[AsyncSession],
    *,
    status: str = "claimed",
    assigned_to: int | None = 42,
    claimed: bool = True,
) -> tuple[int, int, list[int]]:
    async with maker() as session:
        user = User(email="history@example.test", hashed_password="test-only")
        session.add(user)
        await session.flush()
        chat = ChatSession(user_id=user.id)
        other_chat = ChatSession(user_id=user.id)
        session.add_all([chat, other_chat])
        await session.flush()
        ticket = HandoffTicket(
            session_id=chat.id,
            user_id=user.id,
            reason="explicit",
            status=status,
            assigned_to=assigned_to,
            claimed_at=datetime.now(UTC) if claimed else None,
            summary='{"slot_values": {"phone": "private"}}',
        )
        session.add(ticket)
        timestamp = datetime(2026, 10, 22, 1, 0, tzinfo=UTC)
        messages = [
            Message(
                session_id=chat.id,
                role=MessageRole.USER,
                content="I need a refund",
                created_at=timestamp,
                message_metadata={
                    "sources": ["user-supplied"],
                    "citations": [{"source_type": "faq"}],
                },
            ),
            Message(
                session_id=chat.id,
                role=MessageRole.SYSTEM,
                content="Private system summary",
                created_at=timestamp,
            ),
            Message(
                session_id=chat.id,
                role=MessageRole.ASSISTANT,
                content="Please share the order number",
                created_at=timestamp,
                message_metadata={
                    "sources": ["doc-a", 13, {"secret": "hidden"}, "doc-a", " ", "x" * 513],
                    "citations": [
                        {"source_type": "document", "document_id": "doc-a", "raw": "hidden"},
                        {"source_type": "document", "page_start": False},
                        "untrusted",
                    ],
                    "slot_values": {"phone": "private"},
                    "tool_results": [{"token": "private"}],
                },
            ),
            Message(
                session_id=chat.id, role=MessageRole.USER, content="Order 123", created_at=timestamp
            ),
            Message(
                session_id=chat.id,
                role=MessageRole.ASSISTANT,
                content="Agent requested",
                created_at=timestamp,
            ),
            Message(
                session_id=other_chat.id,
                role=MessageRole.USER,
                content="Other session secret",
                created_at=timestamp,
            ),
        ]
        session.add_all(messages)
        await session.commit()
        return (
            ticket.id,
            chat.id,
            [m.id for m in messages if m.session_id == chat.id and m.role != MessageRole.SYSTEM],
        )


async def test_returns_only_persisted_customer_turns_with_stable_pagination(history_maker):
    ticket_id, session_id, message_ids = await seed_history(history_maker)
    service = HandoffService(history_maker)

    first = await service.get_ticket_history(ticket_id, agent_id=42, limit=2)
    second = await service.get_ticket_history(ticket_id, agent_id=42, skip=2, limit=2)
    past_end = await service.get_ticket_history(ticket_id, agent_id=42, skip=99, limit=2)

    assert (first.ticket_id, first.session_id, first.skip, first.limit, first.total) == (
        ticket_id,
        session_id,
        0,
        2,
        4,
    )
    assert [message.id for message in first.messages + second.messages] == message_ids
    assert [message.content for message in first.messages + second.messages] == [
        "I need a refund",
        "Please share the order number",
        "Order 123",
        "Agent requested",
    ]
    assert [message.role for message in first.messages] == ["user", "assistant"]
    assert all(message.timestamp.startswith("2026-10-22T01:00:00") for message in first.messages)
    assert past_end.messages == []
    assert past_end.total == 4


async def test_public_message_fields_allowlist_metadata_and_validate_citations(history_maker):
    ticket_id, _, _ = await seed_history(history_maker)
    result = await HandoffService(history_maker).get_ticket_history(ticket_id, agent_id=42)

    user, assistant = result.messages[:2]
    assert user.sources is None
    assert user.citations == []
    assert assistant.sources == ["doc-a"]
    assert len(assistant.citations) == 1
    assert assistant.citations[0].document_id == "doc-a"
    serialized = assistant.model_dump(mode="json")
    assert set(serialized) == {"id", "role", "content", "timestamp", "sources", "citations"}
    assert "private" not in str(serialized)
    assert "raw" not in str(serialized)


@pytest.mark.parametrize(
    ("status", "assigned_to", "claimed", "missing"),
    [
        ("open", None, False, False),
        ("resolved", 42, True, False),
        ("claimed", 43, True, False),
        ("claimed", None, True, False),
        ("claimed", 42, False, False),
        ("claimed", 42, True, True),
    ],
)
async def test_denied_tickets_are_indistinguishable_and_never_read_messages(
    history_maker, status, assigned_to, claimed, missing
):
    ticket_id, _, _ = await seed_history(
        history_maker, status=status, assigned_to=assigned_to, claimed=claimed
    )
    with (
        patch.object(TicketRepository, "get_history_messages", new_callable=AsyncMock) as rows,
        patch.object(TicketRepository, "count_history_messages", new_callable=AsyncMock) as count,
        pytest.raises(LookupError, match="^Ticket history not found$"),
    ):
        await HandoffService(history_maker).get_ticket_history(999999 if missing else ticket_id, 42)
    rows.assert_not_awaited()
    count.assert_not_awaited()


@pytest.mark.parametrize(("skip", "limit"), [(-1, 50), (2_147_483_648, 50), (0, 0), (0, 101)])
async def test_service_rejects_unbounded_pagination_before_database_access(skip, limit):
    def forbidden_maker():
        raise AssertionError("invalid pagination must not open a database session")

    with pytest.raises(ValueError):
        await HandoffService(forbidden_maker).get_ticket_history(1, 42, skip=skip, limit=limit)


@pytest.mark.parametrize(
    ("ticket_id", "agent_id"),
    [(0, 42), (-1, 42), (2_147_483_648, 42), (1, 0), (1, -1), (1, 2_147_483_648)],
)
async def test_service_rejects_invalid_identifiers_before_database_access(ticket_id, agent_id):
    def forbidden_maker():
        raise AssertionError("invalid identifiers must not reach the database driver")

    with pytest.raises(LookupError, match="^Ticket history not found$"):
        await HandoffService(forbidden_maker).get_ticket_history(ticket_id, agent_id)


async def test_authorization_is_rechecked_after_resolve(history_maker):
    ticket_id, _, _ = await seed_history(history_maker)
    service = HandoffService(history_maker)
    assert (await service.get_ticket_history(ticket_id, 42)).total == 4
    await service.resolve_ticket(ticket_id, 42)
    with pytest.raises(LookupError, match="^Ticket history not found$"):
        await service.get_ticket_history(ticket_id, 42)


async def test_authorization_and_message_reads_share_one_transaction(history_maker):
    ticket_id, _, _ = await seed_history(history_maker)
    transactions = []
    sessions = []

    def inspect_operation(name: str) -> Callable[..., Awaitable[Any]]:
        original = getattr(TicketRepository, name)

        async def inspect(repo: TicketRepository, *args: Any, **kwargs: Any) -> Any:
            assert repo.session.in_transaction()
            transactions.append(repo.session.get_transaction())
            sessions.append(repo.session)
            return await original(repo, *args, **kwargs)

        return inspect

    with (
        patch.object(
            TicketRepository,
            "get_claimed_for_history",
            inspect_operation("get_claimed_for_history"),
        ),
        patch.object(
            TicketRepository, "get_history_messages", inspect_operation("get_history_messages")
        ),
        patch.object(
            TicketRepository, "count_history_messages", inspect_operation("count_history_messages")
        ),
    ):
        result = await HandoffService(history_maker).get_ticket_history(ticket_id, 42)
    assert result.total == 4
    assert len(transactions) == 3
    assert all(transaction is transactions[0] for transaction in transactions)
    assert all(session is sessions[0] for session in sessions)
    assert not sessions[0].in_transaction()
