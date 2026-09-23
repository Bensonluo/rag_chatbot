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

import logging
from collections.abc import Callable
from typing import Any

from app.models.database.message import Message
from app.models.enums.message import MessageRole, MessageStatus
from app.repositories.message_repository import MessageRepository
from app.services.chat.chat_service import ChatMessage

logger = logging.getLogger(__name__)


class ChatMessagePersister:
    """Persists chat turns with request-scoped database sessions."""

    def __init__(self, session_maker: Callable[[], Any]) -> None:
        """
        Args:
            session_maker: Factory producing context-managed async sessions
                (e.g. ``app.api.database.async_session_maker``).
        """
        self._session_maker = session_maker

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
    ) -> list[ChatMessage]:
        """Read chat history for a session using a request-scoped session."""
        try:
            async with self._session_maker() as session:
                repo = MessageRepository(session)
                messages = await repo.get_by_session(session_id, limit=limit)
                return [ChatMessage(role=msg.role.value, content=msg.content) for msg in messages]
        except Exception as exc:  # noqa: BLE001 - degrade to empty history
            logger.warning(
                "Failed to read chat history (session=%s): %s",
                session_id,
                exc,
            )
            return []


def create_chat_persister() -> ChatMessagePersister:
    """Build the default persister bound to the app's session maker."""
    from app.api.database import async_session_maker

    return ChatMessagePersister(session_maker=async_session_maker)
