"""The human channel is durable, idempotent, and separate from bot messages."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.exceptions import ConflictError
from app.models.database.base import Base
from app.models.database.handoff_message import HandoffMessage
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.models.database.user import User
from app.services.chat.persistence import ChatMessagePersister
from app.services.handoff.service import HandoffService


@pytest.fixture
async def human_stack():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as db:
        db.add_all(
            User(id=i, email=f"human-{i}@unit.invalid", hashed_password="!", is_admin=i in (2, 3))
            for i in range(1, 5)
        )
        await db.flush()
        db.add_all(
            [
                ChatSession(id=10, user_id=1, session_metadata={"access_mode": "private"}),
                ChatSession(id=20, user_id=4),
            ]
        )
        await db.flush()
        db.add(
            HandoffTicket(
                id=1,
                session_id=10,
                user_id=1,
                reason="explicit",
                status="claimed",
                assigned_to=2,
                claimed_at=datetime.now(UTC),
            )
        )
        await db.commit()
    try:
        yield maker, HandoffService(maker)
    finally:
        await engine.dispose()


async def test_customer_and_staff_messages_share_only_the_human_channel(human_stack):
    maker, service = human_stack
    customer = await service.send_customer_message(10, 1, str(uuid4()), "My order is delayed")
    staff = await service.send_staff_message(1, 2, str(uuid4()), "I am checking the delivery")
    page = await service.get_customer_messages(10, 1)
    assert [message.author_kind for message in page.messages] == ["customer", "staff"]
    assert customer.message.id == page.messages[0].id
    assert staff.message.id == page.messages[1].id
    assert page.ticket.status == "claimed"
    async with maker() as db:
        assert (await db.scalar(select(func.count()).select_from(Message))) == 0


async def test_final_reply_and_generation_commit_once_and_exact_replay_is_safe(human_stack):
    maker, service = human_stack
    key = str(uuid4())
    first = await service.send_staff_message(
        1, 2, key, "The carrier confirmed delivery", resolve=True
    )
    again = await service.send_staff_message(
        1, 2, key, "The carrier confirmed delivery", resolve=True
    )
    assert first.ticket.status == "resolved"
    assert again.reused
    assert first.message.id == again.message.id
    async with maker() as db:
        assert (await db.get(ChatSession, 10)).automation_epoch == 1
        assert await db.scalar(select(func.count()).select_from(HandoffMessage)) == 1


async def test_idempotency_key_cannot_change_body_or_resolution_command(human_stack):
    _, service = human_stack
    key = str(uuid4())
    await service.send_staff_message(1, 2, key, "Checking")
    with pytest.raises(ConflictError):
        await service.send_staff_message(1, 2, key, "Different")
    with pytest.raises(ConflictError):
        await service.send_staff_message(1, 2, key, "Checking", resolve=True)


async def test_idempotency_key_cannot_change_author_or_ticket(human_stack):
    maker, service = human_stack
    key = str(uuid4())
    await service.send_customer_message(10, 1, key, "Question")
    with pytest.raises(ConflictError):
        await service.send_staff_message(1, 2, key, "Question")
    await service.resolve_ticket(1, 2)
    async with maker() as db:
        db.add(HandoffTicket(id=2, session_id=10, user_id=1, reason="explicit", status="open"))
        await db.commit()
    with pytest.raises(ConflictError):
        await service.send_customer_message(10, 1, key, "Question")


async def test_customer_replay_after_resolution_does_not_create_a_new_message(human_stack):
    _, service = human_stack
    key = str(uuid4())
    first = await service.send_customer_message(10, 1, key, "Please investigate")
    await service.resolve_ticket(1, 2)
    again = await service.send_customer_message(10, 1, key, "Please investigate")
    assert again.reused and again.message.id == first.message.id
    with pytest.raises(ConflictError):
        await service.send_customer_message(10, 1, str(uuid4()), "A new question")


async def test_foreign_owner_and_other_staff_do_not_read_or_write(human_stack):
    _, service = human_stack
    with pytest.raises(LookupError):
        await service.get_customer_messages(10, 4)
    with pytest.raises(LookupError):
        await service.send_customer_message(10, 4, str(uuid4()), "Foreign")
    with pytest.raises(LookupError):
        await service.get_staff_messages(1, 3)
    with pytest.raises(LookupError):
        await service.send_staff_message(1, 3, str(uuid4()), "Foreign")


async def test_cursor_uses_only_returned_messages_and_context_retains_attribution(human_stack):
    _, service = human_stack
    await service.send_customer_message(10, 1, str(uuid4()), "Question")
    await service.send_staff_message(1, 2, str(uuid4()), "Answer")
    first = await service.get_customer_messages(10, 1, limit=1)
    second = await service.get_customer_messages(10, 1, after_id=first.next_after_id, limit=1)
    assert first.has_more and not second.has_more
    assert first.messages[0].id < second.messages[0].id
    assert second.next_after_id == second.messages[0].id
    assert await service.get_recent_context(10) == [
        {"author_kind": "customer", "content": "Question"},
        {"author_kind": "staff", "content": "Answer"},
    ]


async def test_insert_failure_rolls_back_final_resolution_and_epoch(human_stack):
    maker, service = human_stack
    async with maker() as db:
        await db.execute(
            text(
                "CREATE TRIGGER reject_human BEFORE INSERT ON handoff_messages BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END"
            )
        )
        await db.commit()
    with pytest.raises(IntegrityError):
        await service.send_staff_message(1, 2, str(uuid4()), "Final", resolve=True)
    async with maker() as db:
        assert (await db.get(HandoffTicket, 1)).status == "claimed"
        assert (await db.get(ChatSession, 10)).automation_epoch == 0
        assert await db.scalar(select(func.count()).select_from(HandoffMessage)) == 0


async def test_shared_legacy_demo_has_no_human_channel_or_quoted_context(human_stack):
    maker, service = human_stack
    async with maker() as db:
        chat = await db.get(ChatSession, 10)
        assert chat is not None
        chat.session_metadata = None
        await db.commit()
    with pytest.raises(LookupError):
        await service.get_customer_messages(10, 1)
    with pytest.raises(LookupError):
        await service.send_customer_message(10, 1, str(uuid4()), "Private text")
    with pytest.raises(ConflictError, match="requires_private_session"):
        await service.send_staff_message(1, 2, str(uuid4()), "Private text")
    with pytest.raises(LookupError):
        await service.get_staff_messages(1, 2)
    assert await service.get_recent_context(10) == []
    async with maker() as db:
        assert await db.scalar(select(func.count()).select_from(HandoffMessage)) == 0
    # Administrative closure still works for legacy shared tickets.
    assert (await service.resolve_ticket(1, 2))["status"] == "resolved"


async def test_ticket_list_batches_private_flags_in_one_extra_query(human_stack):
    maker, service = human_stack
    engine = maker.kw["bind"]
    async with maker() as db:
        db.add(
            HandoffTicket(
                id=2,
                session_id=20,
                user_id=4,
                reason="explicit",
                status="claimed",
                assigned_to=3,
                claimed_at=datetime.now(UTC),
            )
        )
        await db.commit()
    selects = []

    def observe(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().startswith("SELECT"):
            selects.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", observe)
    tickets = await service.list_tickets("claimed")
    assert {ticket["session_id"]: ticket["is_private"] for ticket in tickets} == {
        10: True,
        20: False,
    }
    assert len(selects) == 2


@pytest.mark.parametrize(
    "content,confirmed",
    [
        ("问题解决了，谢谢", True),
        ("Problem solved", True),
        ("It's fixed now, thanks!", True),
        ("It is not solved", False),
        ("问题还没解决", False),
        ("问题解决了吗？", False),
        ("谢谢", False),
        ("I'll try it later", False),
        ("Problem solved, but I still need help", False),
        ("帮我查一下订单", False),
        ("I need more help", False),
        ("问题解决了，帮我退款", False),
    ],
)
@pytest.mark.parametrize("previously_resolved", [False, True])
async def test_human_and_bot_customer_inputs_have_identical_resolution_semantics(
    human_stack, content, confirmed, previously_resolved
):
    maker, service = human_stack
    old = datetime(2024, 1, 1, tzinfo=UTC)
    async with maker() as db:
        for session_id in (10, 20):
            chat = await db.get(ChatSession, session_id)
            assert chat is not None
            chat.resolved_at = old if previously_resolved else None
            chat.reopened_count = 2
        await db.commit()
    await service.send_customer_message(10, 1, str(uuid4()), content)
    await ChatMessagePersister(maker).persist_turn(20, 4, content, "Bot response")
    async with maker() as db:
        human = await db.get(ChatSession, 10)
        bot = await db.get(ChatSession, 20)
        assert human is not None and bot is not None
        assert (human.resolved_at is not None) == (bot.resolved_at is not None) == confirmed
        expected_reopens = 2 + int(previously_resolved and not confirmed)
        assert human.reopened_count == bot.reopened_count == expected_reopens
        if confirmed:
            assert human.resolved_at is not None
            assert human.resolved_at.replace(tzinfo=UTC) > old


async def test_old_positive_and_negative_replays_cannot_overwrite_later_customer_state(human_stack):
    maker, service = human_stack
    positive, negative = str(uuid4()), str(uuid4())
    await service.send_customer_message(10, 1, positive, "Problem solved")
    await service.send_customer_message(10, 1, negative, "It is not solved")
    assert (await service.send_customer_message(10, 1, positive, "Problem solved")).reused
    async with maker() as db:
        chat = await db.get(ChatSession, 10)
        assert chat is not None
        assert chat.resolved_at is None and chat.reopened_count == 1
    await service.send_customer_message(10, 1, str(uuid4()), "问题解决了")
    async with maker() as db:
        chat = await db.get(ChatSession, 10)
        assert chat is not None and chat.resolved_at is not None
        latest_confirmation = chat.resolved_at
    assert (await service.send_customer_message(10, 1, negative, "It is not solved")).reused
    async with maker() as db:
        chat = await db.get(ChatSession, 10)
        assert chat is not None
        assert chat.resolved_at == latest_confirmation and chat.reopened_count == 1


@pytest.mark.parametrize("content", ["Problem solved", "It is not solved"])
@pytest.mark.parametrize("close", [False, True])
async def test_staff_text_and_ticket_closure_do_not_change_customer_resolution(
    human_stack, content, close
):
    maker, service = human_stack
    old = datetime(2024, 1, 1, tzinfo=UTC)
    async with maker() as db:
        chat = await db.get(ChatSession, 10)
        assert chat is not None
        chat.resolved_at, chat.reopened_count = old, 2
        await db.commit()
    await service.send_staff_message(1, 2, str(uuid4()), content, resolve=close)
    async with maker() as db:
        chat = await db.get(ChatSession, 10)
        assert chat is not None and chat.resolved_at is not None
        assert chat.resolved_at.replace(tzinfo=UTC) == old and chat.reopened_count == 2


@pytest.mark.parametrize(
    "content,previously_resolved", [("Problem solved", False), ("It is not solved", True)]
)
async def test_human_insert_rejection_rolls_back_resolution_and_reopen(
    human_stack, content, previously_resolved
):
    maker, service = human_stack
    old = datetime(2024, 1, 1, tzinfo=UTC)
    async with maker() as db:
        chat = await db.get(ChatSession, 10)
        assert chat is not None
        chat.resolved_at, chat.reopened_count = old if previously_resolved else None, 2
        await db.execute(
            text(
                "CREATE TRIGGER reject_resolution_message BEFORE INSERT ON handoff_messages BEGIN SELECT RAISE(ABORT, 'synthetic refusal'); END"
            )
        )
        await db.commit()
    with pytest.raises(IntegrityError):
        await service.send_customer_message(10, 1, str(uuid4()), content)
    async with maker() as db:
        chat = await db.get(ChatSession, 10)
        assert chat is not None
        assert (chat.resolved_at is not None) == previously_resolved
        assert chat.reopened_count == 2
        assert await db.scalar(select(func.count()).select_from(HandoffMessage)) == 0


async def test_foreign_conflicting_and_closed_new_messages_cannot_transition_resolution(
    human_stack,
):
    maker, service = human_stack
    key = str(uuid4())
    await service.send_customer_message(10, 1, key, "Problem solved")
    async with maker() as db:
        chat = await db.get(ChatSession, 10)
        assert chat is not None and chat.resolved_at is not None
        confirmed_at = chat.resolved_at
    with pytest.raises(LookupError):
        await service.send_customer_message(10, 4, str(uuid4()), "It is not solved")
    with pytest.raises(ConflictError):
        await service.send_customer_message(10, 1, key, "It is not solved")
    await service.resolve_ticket(1, 2)
    with pytest.raises(ConflictError):
        await service.send_customer_message(10, 1, str(uuid4()), "It is not solved")
    async with maker() as db:
        chat = await db.get(ChatSession, 10)
        assert chat is not None
        assert chat.resolved_at == confirmed_at and chat.reopened_count == 0
