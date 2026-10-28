"""
Session repository for chat session data access.

Provides database operations specific to the ChatSession model.
"""

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import ConflictError
from app.models.database.handoff_message import HandoffMessage
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.models.enums.message import MessageRole
from app.repositories.base import BaseRepository


class ReviewChangedError(ConflictError):
    """The review proposal no longer describes the committed evidence."""


@dataclass(frozen=True)
class ReviewSnapshot:
    """Server evidence version, independent of a chat's mutable display title."""

    row: ChatSession
    version: str
    last_message_id: int
    last_human_message_id: int
    eligible: bool

    def response(self, preview: str | None = None) -> dict[str, Any]:
        return {
            "session_id": self.row.id,
            "session_key": self.row.session_key,
            "review_version": self.version,
            "last_message_id": self.last_message_id,
            "last_human_message_id": self.last_human_message_id,
            "title": self.row.title,
            "updated_at": self.row.updated_at.isoformat(),
            "reopened_count": self.row.reopened_count,
            "resolved_at": self.row.resolved_at.isoformat() if self.row.resolved_at else None,
            "qa_verdict": (self.row.session_metadata or {}).get("qa_verdict"),
            "last_user_message": preview[:200] if preview else None,
        }


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

    async def update_fields_if_key(
        self, session_id: int, user_id: int, session_key: str, values: dict[str, object]
    ) -> ChatSession | None:
        result = await self.session.execute(
            update(ChatSession)
            .where(
                ChatSession.id == session_id,
                ChatSession.user_id == user_id,
                ChatSession.session_key == session_key,
            )
            .values(**values)
            .returning(ChatSession)
            .execution_options(populate_existing=True)
        )
        row = result.scalar_one_or_none()
        await self.session.commit()
        return row

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

    async def get_review_snapshot(
        self, session_id: int, *, since: datetime | None = None
    ) -> ReviewSnapshot | None:
        """Read under the caller's conversation gate; never trust an ORM identity cache."""
        row = (
            await self.session.execute(
                select(ChatSession)
                .where(ChatSession.id == session_id)
                .execution_options(populate_existing=True)
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        message_rows = await self.session.stream(
            select(
                Message.id,
                Message.role,
                Message.status,
                Message.user_rating,
                Message.feedback_text,
                Message.created_at,
            )
            .where(Message.session_id == session_id)
            .order_by(Message.id)
            .execution_options(yield_per=256)
        )
        message_digest = hashlib.sha256()
        last_message_id = 0
        served_in_window = False
        downvoted_in_window = False
        window_start = (
            since if since is None or since.tzinfo is not None else since.replace(tzinfo=UTC)
        )
        try:
            async for message in message_rows:
                # The opaque version covers every feedback change without
                # materializing an unbounded transcript in application memory.
                message_digest.update(
                    json.dumps(tuple(message), separators=(",", ":"), default=str).encode()
                )
                message_digest.update(b"\n")
                last_message_id = message.id
                created_at = message.created_at
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=UTC)
                if message.role == MessageRole.ASSISTANT and (
                    window_start is None or created_at >= window_start
                ):
                    served_in_window = True
                    if message.user_rating is not None and message.user_rating < 0:
                        downvoted_in_window = True
        finally:
            await message_rows.close()
        last_human = int(
            await self.session.scalar(
                select(func.coalesce(func.max(HandoffMessage.id), 0)).where(
                    HandoffMessage.session_id == session_id
                )
            )
            or 0
        )
        tickets = (
            await self.session.execute(
                select(HandoffTicket.id, HandoffTicket.status, HandoffTicket.assigned_to)
                .where(HandoffTicket.session_id == session_id)
                .order_by(HandoffTicket.id)
            )
        ).all()
        qa_verdict = (row.session_metadata or {}).get("qa_verdict")
        evidence = {
            "session_key": row.session_key,
            "owner": row.user_id,
            "epoch": row.automation_epoch,
            "resolved_at": row.resolved_at,
            "reopened_count": row.reopened_count,
            "qa_verdict": qa_verdict,
            "messages_sha256": message_digest.hexdigest(),
            "last_human_message_id": last_human,
            "tickets": [tuple(ticket) for ticket in tickets],
        }
        version = hashlib.sha256(
            json.dumps(evidence, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
        eligible = (
            row.resolved_at is None
            and not qa_verdict
            and not tickets
            and served_in_window
            and not downvoted_in_window
        )
        return ReviewSnapshot(row, version, last_message_id, last_human, eligible)

    async def record_review_verdict(
        self,
        session_id: int,
        *,
        resolved: bool,
        note: str | None = None,
        expected_session_key: str | None = None,
        expected_review_version: str | None = None,
    ) -> ChatSession | None:
        """Record a QA reviewer's verdict on a sampled session.

        The verdict always lands in ``session_metadata.qa_verdict``
        (overwriting any prior verdict — one verdict per session, not
        an audit log) so the session leaves the review queue. A
        "resolved" verdict additionally stamps ``resolved_at`` with the
        same semantics as the user-confirmation path, feeding
        sessions_resolved in the containment cross-tab. An unresolved
        verdict clears that stamp, including when correcting a prior
        verdict. QA corrections do not increment ``reopened_count``;
        that counter tracks user follow-ups. Read-modify-write on the
        JSON column. The HTTP caller holds the conversation gate across snapshot
        validation and commit. The conditional UPDATE additionally binds incarnation
        and transcript watermarks; stale proposals never mutate ORM state first.
        """
        snapshot = await self.get_review_snapshot(session_id)
        if snapshot is None:
            return None
        session_row = snapshot.row
        if expected_session_key is None or session_row.session_key != expected_session_key:
            raise ReviewChangedError("Review conversation changed; refresh the review")
        if expected_review_version is None or expected_review_version != snapshot.version:
            raise ReviewChangedError("Review evidence changed; refresh the review")
        reviewed_at = datetime.now(UTC)
        metadata = {
            **(session_row.session_metadata or {}),
            "qa_verdict": {
                "resolved": resolved,
                "note": note,
                "reviewed_at": reviewed_at.isoformat(),
            },
        }
        last_message = (
            select(func.coalesce(func.max(Message.id), 0))
            .where(Message.session_id == session_id)
            .scalar_subquery()
        )
        last_human = (
            select(func.coalesce(func.max(HandoffMessage.id), 0))
            .where(HandoffMessage.session_id == session_id)
            .scalar_subquery()
        )
        prior = (session_row.session_metadata or {}).get("qa_verdict") or {}
        old_reviewed_at = prior.get("reviewed_at", "") if isinstance(prior, dict) else ""
        result = await self.session.execute(
            update(ChatSession)
            .where(
                ChatSession.id == session_id,
                ChatSession.user_id == session_row.user_id,
                ChatSession.session_key == session_row.session_key,
                ChatSession.automation_epoch == session_row.automation_epoch,
                ChatSession.reopened_count == session_row.reopened_count,
                ChatSession.resolved_at == session_row.resolved_at
                if session_row.resolved_at is not None
                else ChatSession.resolved_at.is_(None),
                last_message == snapshot.last_message_id,
                last_human == snapshot.last_human_message_id,
                func.coalesce(
                    ChatSession.session_metadata["qa_verdict"]["reviewed_at"].as_string(), ""
                )
                == old_reviewed_at,
            )
            .values(session_metadata=metadata, resolved_at=reviewed_at if resolved else None)
            .returning(ChatSession)
            .execution_options(populate_existing=True)
        )
        updated = result.scalar_one_or_none()
        if updated is None:
            raise ReviewChangedError("Review evidence changed; refresh the review")
        await self.session.commit()
        return updated

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
