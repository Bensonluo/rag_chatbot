"""
Chat orchestration service.

Delegates dialogue management to a compiled LangGraph, while retaining
optional backward-compatible services (LLM, memory, guardrails) for
history management and fallback use.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.chat.persistence import ChatMessagePersister

# Sentinel yielded by process_message_stream when no token has arrived
# within the heartbeat window. The SSE layer translates it to a keepalive
# comment; transport-agnostic consumers simply skip it. Nodes never emit
# empty strings, so it cannot collide with real content.
HEARTBEAT = ""


@dataclass
class ChatResponse:
    """
    Chat response data.

    Attributes:
        content: Response text content
        session_id: Session identifier
        intent: Detected intent
        sources: Optional list of source document IDs
        metadata: Optional response metadata
    """

    content: str
    session_id: int
    intent: str
    sources: list[str] | None = None
    metadata: dict | None = None


@dataclass
class ChatMessage:
    """
    Chat message data.

    Attributes:
        role: Message role (user/assistant/system)
        content: Message content
        timestamp: Optional timestamp
    """

    role: str
    content: str
    timestamp: str | None = None


class ChatService:
    """
    Chat orchestration service backed by a LangGraph dialogue graph.

    The graph handles intent detection, slot filling, retrieval routing,
    guardrails, and LLM generation internally. This service is a thin
    adapter that invokes the graph and translates results into
    ``ChatResponse`` / ``ChatMessage`` dataclasses.
    """

    def __init__(
        self,
        graph,  # Compiled LangGraph
        llm_service=None,  # backward compat
        memory_strategy=None,  # backward compat
        guardrail_service=None,  # backward compat
        persister: ChatMessagePersister | None = None,
    ) -> None:
        self.graph = graph
        self.llm_service = llm_service
        self.memory_strategy = memory_strategy
        self.guardrail_service = guardrail_service
        self.persister = persister

    async def process_message(
        self,
        session_id: int,
        message: str,
        user_id: int,
        max_tokens: int | None = None,  # noqa: ARG002 - reserved for LLM budget wiring
    ) -> ChatResponse:
        """
        Process a user message through the LangGraph dialogue graph.

        Args:
            session_id: Session identifier
            message: User message
            user_id: User identifier
            max_tokens: Optional max tokens (reserved, not yet forwarded)

        Returns:
            ChatResponse with graph output
        """
        config = {"configurable": {"thread_id": str(session_id)}}
        result = await self.graph.ainvoke(
            {"message": message, "session_id": session_id, "user_id": user_id},
            config,
        )

        if self.persister is not None:
            await self.persister.persist_turn(
                session_id=session_id,
                user_id=user_id,
                user_message=message,
                response=result.get("response", ""),
                intent=result.get("intent"),
                sources=result.get("sources"),
                metadata={
                    "confidence": result.get("confidence"),
                    "pending_slots": result.get("pending_slots", []),
                    "filled_slots": result.get("filled_slots", {}),
                },
            )

        return ChatResponse(
            content=result.get("response", ""),
            session_id=session_id,
            intent=result.get("intent", "unknown"),
            sources=result.get("sources"),
            metadata={
                "confidence": result.get("confidence"),
                "pending_slots": result.get("pending_slots", []),
                "filled_slots": result.get("filled_slots", {}),
            },
        )

    async def process_message_stream(
        self,
        session_id: int,
        message: str,
        user_id: int,
        heartbeat_seconds: float = 15.0,
    ) -> AsyncGenerator[str, None]:
        """
        Process a user message with true token streaming.

        The graph runs in a background task; terminal nodes push LLM
        tokens (or complete template responses) onto a per-request
        queue carried in the invoke config, and this coroutine forwards
        them as they arrive. The queue — not node-name parsing — is the
        single source of streamed content.

        When nothing arrives for ``heartbeat_seconds`` a ``HEARTBEAT``
        sentinel is yielded so the SSE layer can emit a keepalive. On
        client disconnect the generator is closed: the graph task is
        cancelled and the partial turn is persisted (checkpointer state
        was already committed by the graph itself).

        Args:
            session_id: Session identifier
            message: User message
            user_id: User identifier
            heartbeat_seconds: Idle window before yielding a heartbeat

        Yields:
            str: Response text chunks (HEARTBEAT sentinel on idle)
        """
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        config = {"configurable": {"thread_id": str(session_id), "stream_queue": queue}}
        # The sentinel is enqueued only after the graph task settles, so
        # the consumer can never exit before the task is observed done.
        invoke_task = asyncio.create_task(
            self.graph.ainvoke(
                {"message": message, "session_id": session_id, "user_id": user_id},
                config,
            )
        )
        invoke_task.add_done_callback(lambda _task: queue.put_nowait(None))
        streamed_content: list[str] = []
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
                except TimeoutError:
                    yield HEARTBEAT
                    continue
                if chunk is None:
                    break
                streamed_content.append(chunk)
                yield chunk
            # Surface graph failures (sentinel already delivered above).
            invoke_task.result()
        finally:
            if not invoke_task.done():
                invoke_task.cancel()
            with contextlib.suppress(BaseException):
                await invoke_task
            # Persist the full streamed turn (partial content if the client
            # disconnected mid-stream) so history and memory stay accurate.
            if self.persister is not None and streamed_content:
                await self.persister.persist_turn(
                    session_id=session_id,
                    user_id=user_id,
                    user_message=message,
                    response="".join(streamed_content),
                )

    async def get_chat_history(
        self,
        session_id: int,
        limit: int = 50,
    ) -> list[ChatMessage]:
        """
        Get chat history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages

        Returns:
            List[ChatMessage]: List of chat messages
        """
        if self.persister is not None:
            return await self.persister.get_history(session_id=session_id, limit=limit)
        if not self.memory_strategy:
            return []
        context = await self.memory_strategy.get_context(session_id=session_id)
        return [ChatMessage(role=msg.role, content=msg.content) for msg in context[:limit]]

    async def clear_chat_history(self, session_id: int) -> None:
        """
        Clear chat history for a session.

        Args:
            session_id: Session identifier
        """
        if self.memory_strategy:
            await self.memory_strategy.clear_session(session_id=session_id)
