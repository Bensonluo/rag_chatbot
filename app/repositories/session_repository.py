"""
Session repository for chat session data access.

Provides database operations specific to the ChatSession model.
"""

import random
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.models.enums.message import MessageRole
from app.repositories.base import BaseRepository


class SessionRepository(BaseRepository[ChatSession]):
    """
    Repository for ChatSession entity operations.

    Extends BaseRepository with session-specific queries.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the session repository.

        Args:
            session: Async database session
        """
        super().__init__(session)

    async def get_by_id(
        self,
        id: int,
        model: type[ChatSession] = ChatSession,
    ) -> ChatSession | None:
        """Get a session by ID without requiring callers to repeat the model."""
        return await super().get_by_id(id, model)

    async def get_by_user_id(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> list[ChatSession]:
        """
        Get all sessions for a user with pagination.

        Args:
            user_id: User ID
            skip: Number of sessions to skip
            limit: Maximum number of sessions to return

        Returns:
            List[ChatSession]: List of user's sessions
        """
        stmt = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_with_messages(self, session_id: int) -> ChatSession | None:
        """
        Get a session with messages preloaded.

        Args:
            session_id: Session ID

        Returns:
            ChatSession | None: Session with messages if found, None otherwise
        """
        stmt = (
            select(ChatSession)
            .where(ChatSession.id == session_id)
            .options(selectinload(ChatSession.messages))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def count_by_user_id(self, user_id: int) -> int:
        """
        Count sessions for a user.

        Args:
            user_id: User ID

        Returns:
            int: Number of sessions
        """
        stmt = select(func.count()).select_from(ChatSession).where(ChatSession.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def get_latest_session(self, user_id: int) -> ChatSession | None:
        """
        Get the most recent session for a user.

        Args:
            user_id: User ID

        Returns:
            ChatSession | None: Latest session if found, None otherwise
        """
        stmt = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def sample_review_queue(
        self,
        since: datetime,
        limit: int = 10,
    ) -> list[ChatSession]:
        """QA review sample: contained sessions with NO positive
        resolution evidence (review 2026-09-26, #11 人工抽检).

        This is the containment numerator's inflation-risk population —
        served in window, no handoff ticket, no downvoted answer, no
        user confirmation (resolved_at IS NULL). A user who gave up
        silently looks identical to a satisfied one until a human reads
        the transcript. Already-reviewed sessions (``session_metadata.
        qa_verdict``) are filtered in Python: cross-database JSON
        filtering would mean dialect SQL for one key. Candidates come
        newest-first; the sample is drawn randomly so repeat pulls
        cover the pool instead of always re-serving the head. Best
        effort, not census: fetches ``limit * 3`` candidates (capped at
        90) before the Python-side dedup.
        """
        in_window = (
            Message.role == MessageRole.ASSISTANT,
            Message.created_at >= since.replace(tzinfo=None),
        )
        served = select(func.distinct(Message.session_id)).where(*in_window)
        ticketed = select(func.distinct(HandoffTicket.session_id)).where(
            HandoffTicket.session_id.in_(served)
        )
        downvoted = select(func.distinct(Message.session_id)).where(
            *in_window,
            Message.user_rating < 0,
        )
        stmt = (
            select(ChatSession)
            .where(
                ChatSession.id.in_(served),
                ChatSession.resolved_at.is_(None),
                ChatSession.id.not_in(ticketed),
                ChatSession.id.not_in(downvoted),
            )
            .order_by(ChatSession.updated_at.desc())
            .limit(min(limit * 3, 90))
        )
        result = await self.session.execute(stmt)
        candidates = [
            row
            for row in result.scalars().all()
            if not (row.session_metadata or {}).get("qa_verdict")
        ]
        return random.sample(candidates, min(limit, len(candidates)))

    async def record_review_verdict(
        self,
        session_id: int,
        *,
        resolved: bool,
        note: str | None = None,
    ) -> ChatSession | None:
        """Record a QA reviewer's verdict on a sampled session.

        The verdict always lands in ``session_metadata.qa_verdict``
        (overwriting any prior verdict — one verdict per session, not
        an audit log) so the session leaves the review queue. A
        "resolved" verdict additionally stamps ``resolved_at`` with the
        same semantics as the user-confirmation path, feeding
        sessions_resolved in the containment cross-tab. Read-modify-
        write on the JSON column; concurrent double-verdicts collapse
        to last-write-wins, acceptable for ops tooling.
        """
        session_row = await self.session.get(ChatSession, session_id)
        if session_row is None:
            return None
        reviewed_at = datetime.now(UTC)
        session_row.session_metadata = {
            **(session_row.session_metadata or {}),
            "qa_verdict": {
                "resolved": resolved,
                "note": note,
                "reviewed_at": reviewed_at.isoformat(),
            },
        }
        if resolved:
            session_row.resolved_at = reviewed_at
        await self.session.commit()
        await self.session.refresh(session_row)
        return session_row

    async def last_user_message(self, session_id: int) -> str | None:
        """Most recent user-role message content (review preview)."""
        stmt = (
            select(Message.content)
            .where(Message.session_id == session_id, Message.role == MessageRole.USER)
            .order_by(Message.id.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
