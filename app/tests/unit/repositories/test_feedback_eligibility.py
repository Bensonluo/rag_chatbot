"""Only completed assistant answers are user-rateable, even with a real ID."""

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.models.enums.message import MessageRole, MessageStatus
from app.repositories.feedback_repository import FeedbackRepository


@pytest.mark.parametrize(
    "role,status,eligible",
    [
        (MessageRole.USER, MessageStatus.COMPLETED, False),
        (MessageRole.SYSTEM, MessageStatus.COMPLETED, False),
        (MessageRole.ASSISTANT, MessageStatus.FAILED, False),
        (MessageRole.ASSISTANT, MessageStatus.PENDING, False),
        (MessageRole.ASSISTANT, MessageStatus.PROCESSING, False),
        (MessageRole.ASSISTANT, MessageStatus.COMPLETED, True),
    ],
)
@pytest.mark.parametrize("rating", [-1, 0, 1])
async def test_feedback_requires_completed_assistant(role, status, eligible, rating):
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with maker() as db:
            db.add(User(id=1, email="rating@unit.invalid", hashed_password="!"))
            await db.flush()
            chat = ChatSession(id=1, user_id=1)
            db.add(chat)
            await db.flush()
            db.add(Message(id=1, session_id=1, role=role, status=status, content="answer"))
            await db.commit()
            result = await FeedbackRepository(db).submit_feedback(
                1,
                rating,
                "user feedback",
                user_id=1,
                session_id=1,
                session_key=chat.session_key,
            )
            assert (result is not None) == eligible
        async with maker() as db:
            row = await db.get(Message, 1)
            assert row is not None
            assert row.user_rating == (rating if eligible else None)
            assert row.feedback_text == ("user feedback" if eligible else None)
    finally:
        await engine.dispose()


@pytest.fixture
async def scoped_db() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with maker() as db:
            db.add_all(
                [
                    User(id=1, email="rating-one@unit.invalid", hashed_password="!"),
                    User(id=2, email="rating-two@unit.invalid", hashed_password="!"),
                ]
            )
            await db.flush()
            db.add_all(
                [
                    ChatSession(id=1, user_id=1, session_metadata={"access_mode": "private"}),
                    ChatSession(id=2, user_id=2),
                ]
            )
            await db.flush()
            db.add(
                Message(
                    id=1,
                    session_id=1,
                    role=MessageRole.ASSISTANT,
                    status=MessageStatus.COMPLETED,
                    content="committed answer",
                )
            )
            await db.commit()
        yield maker
    finally:
        await engine.dispose()


@pytest.mark.parametrize("change", ["owner", "key", "parent", "role", "status"])
async def test_atomic_feedback_predicates_reject_state_changed_after_read(scoped_db, change):
    async with scoped_db() as db:
        repo = FeedbackRepository(db)
        target = await repo.get_feedback_target(1, 1)
        assert target is not None
        await db.rollback()
        async with scoped_db() as other:
            if change == "owner":
                await other.execute(
                    update(ChatSession).where(ChatSession.id == 1).values(user_id=2)
                )
            elif change == "key":
                await other.execute(
                    update(ChatSession).where(ChatSession.id == 1).values(session_key=str(uuid4()))
                )
            else:
                changes: dict[str, dict[str, object]] = {
                    "parent": {"session_id": 2},
                    "role": {"role": MessageRole.USER},
                    "status": {"status": MessageStatus.FAILED},
                }
                await other.execute(
                    update(Message).where(Message.id == 1).values(**changes[change])
                )
            await other.commit()
        result = await repo.submit_feedback(
            1,
            -1,
            "stale feedback",
            user_id=1,
            session_id=target.session_id,
            session_key=target.session_key,
        )
        assert result is None
    async with scoped_db() as fresh:
        message = await fresh.get(Message, 1)
        assert message is not None
        assert message.user_rating is None
        assert message.feedback_text is None


async def test_feedback_target_reads_current_scalars_despite_cached_parent(scoped_db):
    async with scoped_db() as db:
        cached = await db.get(ChatSession, 1)
        assert cached is not None
        old_key = cached.session_key
        await db.commit()
        new_key = str(uuid4())
        async with scoped_db() as other:
            await other.execute(
                update(ChatSession).where(ChatSession.id == 1).values(session_key=new_key)
            )
            await other.commit()
        assert cached.session_key == old_key
        target = await FeedbackRepository(db).get_feedback_target(1, 1)
        assert target is not None
        assert target.session_key == new_key


@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("allow_shared", [False, True])
async def test_atomic_feedback_enforces_foreign_owner_and_shared_policy(
    scoped_db, private, allow_shared
):
    async with scoped_db() as db:
        chat = await db.get(ChatSession, 1)
        assert chat is not None
        chat.user_id = 2
        chat.session_metadata = {"access_mode": "private"} if private else None
        key = chat.session_key
        await db.commit()
        result = await FeedbackRepository(db).submit_feedback(
            1,
            1,
            user_id=1,
            session_id=1,
            session_key=key,
            allow_shared=allow_shared,
        )
        permitted = allow_shared and not private
        assert (result is not None) == permitted
    async with scoped_db() as fresh:
        message = await fresh.get(Message, 1)
        assert message is not None
        assert message.user_rating == (1 if permitted else None)
