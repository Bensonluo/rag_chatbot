"""Only completed assistant answers are user-rateable, even with a real ID."""

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

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
            db.add(ChatSession(id=1, user_id=1))
            await db.flush()
            db.add(Message(id=1, session_id=1, role=role, status=status, content="answer"))
            await db.commit()
            result = await FeedbackRepository(db).submit_feedback(1, rating, "user feedback")
            assert (result is not None) == eligible
        async with maker() as db:
            row = await db.get(Message, 1)
            assert row is not None
            assert row.user_rating == (rating if eligible else None)
            assert row.feedback_text == ("user feedback" if eligible else None)
    finally:
        await engine.dispose()
