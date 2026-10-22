"""
Chat message persistence.

Writes each dialogue turn (user message + assistant response) to the
database using a request-scoped session so no repository is ever bound
to the startup session (which closes once the lifespan context exits).

Turn-write failures are logged without interrupting chat. Explicit history
deletion propagates database errors so callers cannot report a false reset.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, update

from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.enums.message import MessageRole, MessageStatus
from app.repositories.message_repository import MessageRepository
from app.repositories.user_fact_repository import UserFactRepository
from app.services.chat.chat_service import ChatMessage
from app.services.chat.citations import normalize_citations
from app.services.chat.compressor import SessionCompressor
from app.services.chat.resolution import is_resolution_confirmation
from app.services.chat.user_fact_extractor import UserFactExtractor

logger = logging.getLogger(__name__)

# Dedicated owner for auto-created demo sessions (finding ③): demo
# traffic is anonymous, but chat_sessions.user_id is NOT NULL.
_DEMO_USER_EMAIL = "demo-visitor@local.invalid"


async def ensure_demo_session_row(session_maker: Callable[[], Any], session_id: int) -> None:
    """DEMO_MODE: get-or-create the chat_sessions row for this id.

    Shared by the persister (messages FK, d0d2b2a) and handoff ticket
    creation (handoff_tickets FK, live 2026-10-03): both run before the
    turn is persisted, and a first-turn escalation on a fresh demo
    session would otherwise insert into a missing parent. Strict
    deployments 404 unknown ids upfront and never reach here; this runs
    in its own transaction so a lost insert race collapses to
    IntegrityError → the row exists, which is the goal either way.
    """
    from sqlalchemy import select
    from sqlalchemy.exc import IntegrityError

    from app.config.settings import get_settings
    from app.models.database.session import ChatSession
    from app.models.database.user import User

    if not get_settings().DEMO_MODE:
        return
    try:
        async with session_maker() as session:
            if await session.get(ChatSession, session_id) is not None:
                return
            demo_user = (
                await session.execute(select(User).where(User.email == _DEMO_USER_EMAIL))
            ).scalar_one_or_none()
            if demo_user is None:
                demo_user = User(
                    email=_DEMO_USER_EMAIL,
                    hashed_password="!",  # unusable: sign-in is never the point
                    full_name="Demo Visitor",
                )
                session.add(demo_user)
                await session.flush()
            session.add(
                ChatSession(
                    id=session_id,
                    user_id=demo_user.id,
                    title="Demo Chat",
                )
            )
            await session.commit()
    except IntegrityError:
        # Parallel first-turn on the same id: the row exists now.
        pass
    except Exception:  # noqa: BLE001 - availability over durability
        logger.warning(
            "Demo session row ensure failed (session=%s); turn may not persist",
            session_id,
            exc_info=True,
        )


class ChatMessagePersister:
    """Persists chat turns with request-scoped database sessions."""

    def __init__(
        self,
        session_maker: Callable[[], Any],
        compressor: SessionCompressor | None = None,
        fact_extractor: UserFactExtractor | None = None,
    ) -> None:
        """
        Args:
            session_maker: Factory producing context-managed async sessions
                (e.g. ``app.api.database.async_session_maker``).
            compressor: Optional rolling-summary writer scheduled after
                each durable turn (fire-and-forget; ``drain`` awaits it).
            fact_extractor: Optional cross-session user-fact extractor
                (Phase B) with the same fire-and-forget contract.
        """
        self._session_maker = session_maker
        self._compressor = compressor
        self._fact_extractor = fact_extractor
        self._pending_compressions: dict[asyncio.Task[bool], int] = {}
        self._pending_extractions: dict[asyncio.Task[bool], int] = {}

    async def persist_turn(
        self,
        session_id: int,
        user_id: int | None,
        user_message: str,
        response: str,
        intent: str | None = None,
        sources: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        status: MessageStatus = MessageStatus.COMPLETED,
    ) -> None:
        """Persist one dialogue turn (user + assistant messages).

        Never raises: persistence problems are logged and swallowed so a
        database hiccup cannot fail an in-flight chat response.
        """
        if not user_message and not response:
            return
        try:
            await ensure_demo_session_row(self._session_maker, session_id)
            async with self._session_maker() as session:
                repo = MessageRepository(session)
                await self._apply_resolution_transition(session, session_id, user_message)
                assistant_metadata = dict(metadata or {})
                citations = normalize_citations(assistant_metadata.get("citations"))
                if citations:
                    assistant_metadata["citations"] = citations
                else:
                    assistant_metadata.pop("citations", None)
                if sources:
                    assistant_metadata["sources"] = sources

                if user_message:
                    # Message has no user_id column; keep attribution in
                    # metadata until the ownership migration promotes it.
                    user_metadata = {"user_id": user_id} if user_id else None
                    await repo.create(
                        Message(
                            session_id=session_id,
                            role=MessageRole.USER,
                            content=user_message,
                            intent=intent,
                            status=MessageStatus.COMPLETED,
                            message_metadata=user_metadata,
                        )
                    )
                if response:
                    await repo.create(
                        Message(
                            session_id=session_id,
                            role=MessageRole.ASSISTANT,
                            content=response,
                            status=status,
                            message_metadata=assistant_metadata or None,
                        )
                    )
                await session.commit()
            # Compression and fact extraction are post-durability
            # housekeeping: scheduled off the request path so the
            # user-visible turn never waits on an LLM call.
            self._schedule_compression(session_id)
            self._schedule_extraction(session_id)
        except Exception as exc:  # noqa: BLE001 - availability over durability
            logger.warning(
                "Failed to persist chat turn (session=%s): %s",
                session_id,
                exc,
            )

    @staticmethod
    async def _apply_resolution_transition(
        dbsession: Any, session_id: int, user_message: str
    ) -> None:
        """Session resolution state machine (review 2026-09-26, #11).

        Confirmation wins over reopen by construction — the confirm
        branch runs first, so a resolve phrase on an already-resolved
        session just refreshes the timestamp (repeated "解决了" never
        reopens). Any other user message on a resolved session flips it
        open and counts the reopen. Runs inside the caller's fail-open
        transaction: a missing row (strict mode, deleted session) is a
        no-op UPDATE, and any failure degrades to untracked resolution,
        never a failed chat turn.
        """
        if not user_message:
            return
        if is_resolution_confirmation(user_message):
            await dbsession.execute(
                update(ChatSession)
                .where(ChatSession.id == session_id)
                .values(resolved_at=datetime.now(UTC))
            )
            return
        await dbsession.execute(
            update(ChatSession)
            .where(
                ChatSession.id == session_id,
                ChatSession.resolved_at.is_not(None),
            )
            .values(resolved_at=None, reopened_count=ChatSession.reopened_count + 1)
        )

    async def get_history(
        self,
        session_id: int,
        limit: int = 50,
        include_summary: bool = False,
    ) -> list[ChatMessage]:
        """Read chat history for a session using a request-scoped session.

        With ``include_summary=True`` (the LLM-context read), the latest
        session summary is prepended as a system block and turns it
        already covers are cut — same overlap-free contract as the
        optimized memory strategy. The default (API tail view) stays
        raw turns only.
        """
        try:
            async with self._session_maker() as session:
                repo = MessageRepository(session)
                # Newest N, not oldest N: get_by_session paginates ASC
                # from the session's first message, but a history read
                # (API tail view / LLM context) wants the recent turns.
                # DESC fetch + reverse = chronological most-recent-N.
                recent = await repo.get_recent_messages(session_id, limit=limit)
                summary = await repo.get_latest_summary(session_id) if include_summary else None
                # System-role rows are summary artifacts, not turns.
                rows = [msg for msg in reversed(recent) if msg.role != MessageRole.SYSTEM]
                if summary is not None:
                    # Overlap-free: the summary already carries whatever
                    # happened before it was written.
                    rows = [msg for msg in rows if msg.created_at >= summary.created_at]
                turns = []
                for msg in rows:
                    metadata = msg.message_metadata or {}
                    sources = metadata.get("sources")
                    turns.append(
                        ChatMessage(
                            role=msg.role.value,
                            content=msg.content,
                            timestamp=msg.created_at.isoformat(),
                            sources=(
                                [source for source in sources if isinstance(source, str)]
                                if isinstance(sources, list) and msg.role == MessageRole.ASSISTANT
                                else None
                            ),
                            citations=(
                                normalize_citations(metadata.get("citations"))
                                if msg.role == MessageRole.ASSISTANT
                                else []
                            ),
                        )
                    )
                if summary is not None and summary.content:
                    turns.insert(
                        0,
                        ChatMessage(
                            role="system",
                            content=f"Previous conversation: {summary.content}",
                        ),
                    )
                return turns
        except Exception as exc:  # noqa: BLE001 - degrade to empty history
            logger.warning(
                "Failed to read chat history (session=%s): %s",
                session_id,
                exc,
            )
            return []

    async def get_user_facts(self, *, user_id: int, limit: int = 8) -> list[str]:
        """Newest durable facts for a user (Phase B2 recall side).

        Best-effort like every persister read: a memory outage degrades
        to no personalization, never a failed chat.
        """
        try:
            async with self._session_maker() as session:
                repo = UserFactRepository(session)
                rows = await repo.recent_for_user(user_id=user_id, limit=limit)
                return [row.fact for row in rows]
        except Exception as exc:  # noqa: BLE001 - degrade to no personalization
            logger.warning("Failed to read user facts (user=%s): %s", user_id, exc)
            return []

    async def clear_history(self, session_id: int) -> None:
        """Delete durable turns and summaries after this process's writers settle.

        Other sessions, user facts and handoff tickets are retained. Concurrent
        new turns or writers in other processes require separate coordination.
        Unlike best-effort turn writes, a failed clear must reach the caller.
        """
        await self.drain(session_id=session_id)
        async with self._session_maker() as session:
            await session.execute(delete(Message).where(Message.session_id == session_id))
            await session.commit()

    def _schedule_compression(self, session_id: int) -> None:
        """Schedule best-effort compression; never disturbs the caller."""
        if self._compressor is None:
            return
        try:
            task: asyncio.Task[bool] = asyncio.create_task(
                self._compressor.maybe_compress(session_id)
            )
        except RuntimeError:  # no running loop (sync caller in tests)
            return
        self._pending_compressions[task] = session_id
        task.add_done_callback(lambda finished: self._pending_compressions.pop(finished, None))

    def _schedule_extraction(self, session_id: int) -> None:
        """Schedule best-effort user-fact extraction (Phase B); same
        fire-and-forget contract as compression."""
        if self._fact_extractor is None:
            return
        try:
            task: asyncio.Task[bool] = asyncio.create_task(
                self._fact_extractor.maybe_extract(session_id)
            )
        except RuntimeError:  # no running loop (sync caller in tests)
            return
        self._pending_extractions[task] = session_id
        task.add_done_callback(lambda finished: self._pending_extractions.pop(finished, None))

    async def drain(self, session_id: int | None = None) -> None:
        """Await current background work, optionally scoped to one conversation."""
        tracked = self._pending_compressions | self._pending_extractions
        pending = [
            task for task, owner in tracked.items() if session_id is None or owner == session_id
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def create_chat_persister() -> ChatMessagePersister:
    """Build the default persister bound to the app's session maker."""
    from app.api.database import async_session_maker

    return ChatMessagePersister(session_maker=async_session_maker)
