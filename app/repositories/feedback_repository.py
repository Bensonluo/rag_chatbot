"""Feedback repository for message rating operations."""

from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.enums.message import MessageRole, MessageStatus


class FeedbackRepository:
    """Async repository for feedback operations on messages."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_owned_message(
        self, message_id: int, user_id: int, *, allow_shared: bool = False
    ) -> Message | None:
        """The message when its session belongs to user_id, else None.

        Feedback is an object-level authorization surface (review #1):
        the rating — and a downvote's cache eviction — must never touch
        another user's message, and a foreign id must read exactly like
        a missing one (BOLA guidance).
        """
        access = ChatSession.user_id == user_id
        if allow_shared:
            access = or_(
                access,
                func.coalesce(ChatSession.session_metadata["access_mode"].as_string(), "")
                != "private",
            )
        stmt = (
            select(Message)
            .join(ChatSession, Message.session_id == ChatSession.id)
            .where(Message.id == message_id, access)
        )
        return (await self._db.execute(stmt)).scalar_one_or_none()

    async def submit_feedback(
        self,
        message_id: int,
        rating: int,
        text: str | None = None,
    ) -> Message | None:
        """Rate a completed assistant answer, else return an indistinguishable miss."""
        statement = select(Message).where(
            Message.id == message_id,
            Message.role == MessageRole.ASSISTANT,
            Message.status == MessageStatus.COMPLETED,
        )
        message = (await self._db.execute(statement)).scalar_one_or_none()
        if message is None:
            return None

        message.user_rating = rating
        message.feedback_text = text
        await self._db.commit()
        await self._db.refresh(message)
        return message

    async def get_feedback_stats(self) -> dict[str, Any]:
        """Get aggregate feedback statistics."""
        result = await self._db.execute(
            select(Message.user_rating).where(Message.user_rating.isnot(None))
        )
        ratings = [r for (r,) in result.fetchall() if r is not None]
        if not ratings:
            return {"total": 0, "positive": 0, "negative": 0, "average": 0.0}
        return {
            "total": len(ratings),
            "positive": sum(1 for r in ratings if r > 0),
            "negative": sum(1 for r in ratings if r < 0),
            "average": sum(ratings) / len(ratings),
        }
