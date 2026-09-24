"""Session compression for the primary chat pipeline.

Periodically folds a session's pre-window turns into a rolling summary
row so the bounded history window (persister reads the newest 6 turns)
retains early context — order numbers, promised deadlines, agreed
refunds — without unbounded token growth. This is the write side the
legacy SummarizationMemory strategy never ran on the LangGraph path.

Best-effort by contract: compression problems are logged and swallowed
(a failed summary must never fail a chat turn), the LLM call is
time-boxed, and it runs on the persister's schedule after the turn is
already durable.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from app.models.database.message import Message
from app.models.enums.message import MessageRole, MessageStatus
from app.repositories.message_repository import MessageRepository
from app.services.llm.base import LLMMessage, LLMServiceBase
from app.services.llm.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


class SessionCompressor:
    """Writes rolling session summaries on threshold/interval crossings."""

    def __init__(
        self,
        session_maker: Callable[[], Any],
        llm_service: LLMServiceBase | None,
        threshold: int = 20,
        interval: int = 10,
        timeout_seconds: float = 3.0,
    ) -> None:
        """
        Args:
            session_maker: Factory producing context-managed async sessions
            llm_service: LLM used to summarize; None disables compression
            threshold: Message count that triggers the first summary
            interval: Messages between subsequent summaries
            timeout_seconds: Budget before the summary is dropped
        """
        self._session_maker = session_maker
        self._llm_service = llm_service
        self._threshold = threshold
        self._interval = interval
        self._timeout_seconds = timeout_seconds

    async def maybe_compress(self, session_id: int) -> bool:
        """Compress the session when a trigger boundary was crossed.

        Returns True when a summary row was written. Never raises.
        """
        if self._llm_service is None:
            return False
        try:
            return await self._compress(session_id)
        except Exception as exc:  # noqa: BLE001 - availability over durability
            logger.warning("Session compression failed (session=%s): %s", session_id, exc)
            return False

    async def _compress(self, session_id: int) -> bool:
        llm = self._llm_service
        if llm is None:
            return False
        # Same trigger cadence as SummarizationMemory: first summary at
        # the threshold, then one every `interval` messages.
        async with self._session_maker() as session:
            repo = MessageRepository(session)
            # Summaries are not turns: count only user/assistant rows
            # so writing a summary never shifts the next trigger.
            count = await repo.count_turns(session_id)
            triggered = count == self._threshold or (
                count > self._threshold and (count - self._threshold) % self._interval == 0
            )
            if not triggered:
                return False
            summary = await repo.get_latest_summary(session_id)
            # Newest-first rows; reverse so the transcript reads like a log.
            recent = list(
                reversed(await repo.get_recent_messages(session_id, limit=self._threshold))
            )
        # Cumulative rolling summary: only turns AFTER the previous
        # summary enter the window; the previous summary's text is folded
        # in so early-session facts never drop out of context.
        window = [msg for msg in recent if msg.role != MessageRole.SYSTEM]
        if summary is not None:
            window = [msg for msg in window if msg.created_at > summary.created_at]
        if not window:
            return False

        # .value keeps the transcript human-readable for the LLM;
        # the str-Enum repr ("MessageRole.USER") is noise it must not parse.
        transcript = "\n".join(f"{msg.role.value}: {msg.content}" for msg in window)
        if summary is not None and summary.content:
            conversation_text = (
                f"已有的对话摘要：\n{summary.content}\n\n自上次摘要以来的新对话：\n{transcript}"
            )
        else:
            conversation_text = transcript
        prompt = PromptTemplates.get_summarization_prompt(text=conversation_text, max_length=500)
        # LLM call outside any DB session — it never holds a connection.
        response = await asyncio.wait_for(
            llm.generate(
                messages=[LLMMessage(role="user", content=prompt)],
                max_tokens=300,
                temperature=0.3,
            ),
            timeout=self._timeout_seconds,
        )

        async with self._session_maker() as session:
            await MessageRepository(session).create_summary(
                Message(
                    session_id=session_id,
                    role=MessageRole.SYSTEM,
                    content=response.content,
                    status=MessageStatus.COMPLETED,
                )
            )
            await session.commit()
        return True
