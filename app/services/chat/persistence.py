"""
Chat message persistence.

Writes each dialogue turn (user message + assistant response) to the
database using a request-scoped session so no repository is ever bound
to the startup session (which closes once the lifespan context exits).

Persistence failures are logged but never propagated: chat
availability takes priority over durability, and the missing rows are
recoverable from upstream logs if needed.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from app.models.database.message import Message
from app.models.enums.message import MessageRole, MessageStatus
from app.repositories.message_repository import MessageRepository
from app.services.chat.chat_service import ChatMessage
from app.services.chat.compressor import SessionCompressor

logger = logging.getLogger(__name__)


class ChatMessagePersister:
    """Persists chat turns with request-scoped database sessions."""

    def __init__(
        self,
        session_maker: Callable[[], Any],
        compressor: SessionCompressor | None = None,
    ) -> None:
        """
        Args:
            session_maker: Factory producing context-managed async sessions
                (e.g. ``app.api.database.async_session_maker``).
            compressor: Optional rolling-summary writer scheduled after
                each durable turn (fire-and-forget; ``drain`` awaits it).
        """
        self._session_maker = session_maker
        self._compressor = compressor
        self._pending_compressions: set[asyncio.Task[bool]] = set()

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
            async with self._session_maker() as session:
                repo = MessageRepository(session)
                assistant_metadata = dict(metadata or {})
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
            # Compression is post-durability housekeeping: scheduled off
            # the request path so the user-visible turn never waits on
            # an LLM summary call.
            self._schedule_compression(session_id)
        except Exception as exc:  # noqa: BLE001 - availability over durability
            logger.warning(
                "Failed to persist chat turn (session=%s): %s",
                session_id,
                exc,
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
                turns = [ChatMessage(role=msg.role.value, content=msg.content) for msg in rows]
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
        self._pending_compressions.add(task)
        task.add_done_callback(self._pending_compressions.discard)

    async def drain(self) -> None:
        """Await outstanding compression tasks (shutdown / test seams)."""
        if self._pending_compressions:
            await asyncio.gather(*self._pending_compressions, return_exceptions=True)


def create_chat_persister() -> ChatMessagePersister:
    """Build the default persister bound to the app's session maker."""
    from app.api.database import async_session_maker

    return ChatMessagePersister(session_maker=async_session_maker)
