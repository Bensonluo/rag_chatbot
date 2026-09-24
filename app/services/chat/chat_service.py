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
from typing import TYPE_CHECKING, Any

from app.config.settings import settings
from app.services.llm.budget import (
    BudgetState,
    enter_llm_budget,
    record_budget_on_span,
)
from app.services.observability.pipeline_tracer import traced_stage

if TYPE_CHECKING:
    from app.services.chat.knowledge_gap_recorder import KnowledgeGapRecorder
    from app.services.chat.persistence import ChatMessagePersister
    from app.services.guardrails.base import GuardrailService
    from app.services.llm.base import LLMServiceBase
    from app.services.memory.base import MemoryStrategy

# Sentinel yielded by process_message_stream when no token has arrived
# within the heartbeat window. The SSE layer translates it to a keepalive
# comment; transport-agnostic consumers simply skip it. Nodes never emit
# empty strings, so it cannot collide with real content.
HEARTBEAT = ""
# Yielded when the stream ends abnormally (graph failure or duration
# budget exhausted) so the SSE layer can emit an explicit error frame
# instead of dropping the connection.
STREAM_ERROR = "[[STREAM_ERROR]]"


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
    metadata: dict[str, Any] | None = None


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
        graph: Any,  # Compiled LangGraph — no public stable type to reference
        llm_service: LLMServiceBase | None = None,  # backward compat
        memory_strategy: MemoryStrategy | None = None,  # backward compat
        guardrail_service: GuardrailService | None = None,  # backward compat
        persister: ChatMessagePersister | None = None,
        gap_recorder: KnowledgeGapRecorder | None = None,
    ) -> None:
        self.graph = graph
        self.llm_service = llm_service
        self.memory_strategy = memory_strategy
        self.guardrail_service = guardrail_service
        self.persister = persister
        self.gap_recorder = gap_recorder

    @traced_stage("cs.pipeline")
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
        budget: BudgetState | None = None
        async with enter_llm_budget(settings.CHAT_LLM_CALL_BUDGET) as budget_state:
            budget = budget_state
            result = await self.graph.ainvoke(
                {"message": message, "session_id": session_id, "user_id": user_id},
                config,
            )
        if budget is not None:
            # Budget consumption on the root span and in response
            # metadata: a cap nobody can see is a cap nobody calibrates.
            record_budget_on_span(budget)

        metadata: dict[str, Any] = {
            "confidence": result.get("confidence"),
            "pending_slots": result.get("pending_slots", []),
            "filled_slots": result.get("filled_slots", {}),
            "llm_calls": budget.used if budget else 0,
            "llm_refused": budget.refused if budget else 0,
        }
        # Durable audit trail of agent tool executions (refunds and
        # other irreversible support actions must be traceable).
        executed_tools = result.get("executed_tools") or []
        if executed_tools:
            metadata["executed_tools"] = executed_tools

        if self.persister is not None:
            await self.persister.persist_turn(
                session_id=session_id,
                user_id=user_id,
                user_message=message,
                response=result.get("response", ""),
                intent=result.get("intent"),
                sources=result.get("sources"),
                metadata=metadata,
            )
        if self.gap_recorder is not None:
            await self.gap_recorder.record_if_gap(
                query=message,
                intent=result.get("intent"),
                retrieved_docs=result.get("retrieved_docs"),
                session_id=session_id,
                user_id=user_id,
            )

        return ChatResponse(
            content=result.get("response", ""),
            session_id=session_id,
            intent=result.get("intent", "unknown"),
            sources=result.get("sources"),
            metadata=metadata,
        )

    @traced_stage("cs.pipeline.stream")
    async def process_message_stream(
        self,
        session_id: int,
        message: str,
        user_id: int,
        heartbeat_seconds: float = 15.0,
        stream_max_seconds: float = 120.0,
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
            stream_max_seconds: Total budget before the stream is cut off

        Yields:
            str: Response text chunks (HEARTBEAT sentinel on idle)
        """
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        config = {"configurable": {"thread_id": str(session_id), "stream_queue": queue}}
        # The sentinel is enqueued only after the graph task settles, so
        # the consumer can never exit before the task is observed done.
        # Budget scope wraps the whole stream: the graph task inherits
        # the ContextVar (asyncio copies context at task creation), so
        # every node's LLM call in this turn counts against it.
        budget: BudgetState | None = None
        async with enter_llm_budget(settings.CHAT_LLM_CALL_BUDGET) as budget_state:
            budget = budget_state
            invoke_task = asyncio.create_task(
                self.graph.ainvoke(
                    {"message": message, "session_id": session_id, "user_id": user_id},
                    config,
                )
            )
        invoke_task.add_done_callback(lambda _task: queue.put_nowait(None))
        streamed_content: list[str] = []
        loop = asyncio.get_running_loop()
        deadline = loop.time() + stream_max_seconds
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    # Budget exhausted — end the stream rather than hold the
                    # connection open on heartbeats forever.
                    yield STREAM_ERROR
                    return
                try:
                    chunk = await asyncio.wait_for(
                        queue.get(), timeout=min(heartbeat_seconds, remaining)
                    )
                except TimeoutError:
                    if loop.time() >= deadline:
                        yield STREAM_ERROR
                        return
                    yield HEARTBEAT
                    continue
                if chunk is None:
                    break
                streamed_content.append(chunk)
                yield chunk
            if invoke_task.cancelled():
                yield STREAM_ERROR
                return
            if invoke_task.exception() is not None:
                # The client gets an explicit failure frame instead of a
                # dropped connection; partial content is still persisted.
                yield STREAM_ERROR
                return
            result = invoke_task.result()
            if self.gap_recorder is not None:
                await self.gap_recorder.record_if_gap(
                    query=message,
                    intent=result.get("intent"),
                    retrieved_docs=result.get("retrieved_docs"),
                    session_id=session_id,
                    user_id=user_id,
                )
        finally:
            if not invoke_task.done():
                invoke_task.cancel()
            with contextlib.suppress(BaseException):
                await invoke_task
            # Persist the full streamed turn (partial content if the client
            # disconnected mid-stream) so history and memory stay accurate.
            if budget is not None:
                # After the task settles, the state carries this turn's
                # real consumption (the scope only wrapped task creation).
                record_budget_on_span(budget)
            if self.persister is not None and streamed_content:
                await self.persister.persist_turn(
                    session_id=session_id,
                    user_id=user_id,
                    user_message=message,
                    response="".join(streamed_content),
                    metadata={
                        "llm_calls": budget.used if budget else 0,
                        "llm_refused": budget.refused if budget else 0,
                    },
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
        # Memory strategies return MessageContent dicts, not ORM objects.
        return [ChatMessage(role=msg["role"], content=msg["content"]) for msg in context[:limit]]

    async def clear_chat_history(self, session_id: int) -> None:
        """
        Clear chat history for a session.

        Args:
            session_id: Session identifier
        """
        if self.memory_strategy:
            await self.memory_strategy.clear_session(session_id=session_id)
