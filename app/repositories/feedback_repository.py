"""Feedback repository for message rating operations."""

from dataclasses import dataclass
from typing import Any

from sqlalchemy import ColumnElement, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.enums.message import MessageRole, MessageStatus


@dataclass(frozen=True)
class FeedbackTarget:
    """Scalar snapshot; later queries never reuse an ORM identity-map value."""

    id: int
    session_id: int
    session_key: str
    is_private: bool


@dataclass(frozen=True)
class FeedbackReceipt:
    """Committed update values, safe to return after leaving its transaction."""

    id: int
    session_id: int
    session_key: str
    user_rating: int
    content: str


def _feedback_access(user_id: int, allow_shared: bool) -> ColumnElement[bool]:
    access = ChatSession.user_id == user_id
    if allow_shared:
        access = or_(
            access,
            func.coalesce(ChatSession.session_metadata["access_mode"].as_string(), "") != "private",
        )
    return access


class FeedbackRepository:
    """Async repository for feedback operations on messages."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_feedback_target(
        self,
        message_id: int,
        user_id: int,
        *,
        allow_shared: bool = False,
        session_id: int | None = None,
    ) -> FeedbackTarget | None:
        """Discover an authorized, rateable target without retaining ORM state."""
        stmt = (
            select(
                Message.id,
                Message.session_id,
                ChatSession.session_key,
                ChatSession.session_metadata["access_mode"].as_string().label("access_mode"),
            )
            .join(ChatSession, Message.session_id == ChatSession.id)
            .where(
                Message.id == message_id,
                Message.role == MessageRole.ASSISTANT,
                Message.status == MessageStatus.COMPLETED,
                _feedback_access(user_id, allow_shared),
            )
        )
        if session_id is not None:
            stmt = stmt.where(Message.session_id == session_id)
        row = (await self._db.execute(stmt)).one_or_none()
        if row is None:
            return None
        return FeedbackTarget(row.id, row.session_id, row.session_key, row.access_mode == "private")

    async def submit_feedback(
        self,
        message_id: int,
        rating: int,
        text: str | None = None,
        *,
        user_id: int,
        session_id: int,
        session_key: str,
        allow_shared: bool = False,
    ) -> FeedbackReceipt | None:
        """Atomically bind the write to the observed owner, parent and incarnation.

        There is deliberately no unscoped writer: both access and rateability
        remain SQL predicates even if state changes after the caller's read.
        """
        parent = select(ChatSession.id).where(
            ChatSession.id == Message.session_id,
            ChatSession.id == session_id,
            ChatSession.session_key == session_key,
            _feedback_access(user_id, allow_shared),
        )
        statement = (
            update(Message)
            .where(
                Message.id == message_id,
                Message.session_id == session_id,
                Message.role == MessageRole.ASSISTANT,
                Message.status == MessageStatus.COMPLETED,
                parent.exists(),
            )
            .values(user_rating=rating, feedback_text=text)
            .returning(Message.id, Message.session_id, Message.user_rating, Message.content)
            .execution_options(synchronize_session=False)
        )
        row = (await self._db.execute(statement)).one_or_none()
        if row is None:
            await self._db.rollback()
            return None
        receipt = FeedbackReceipt(row.id, row.session_id, session_key, row.user_rating, row.content)
        await self._db.commit()
        return receipt

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
