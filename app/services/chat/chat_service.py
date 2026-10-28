"""
Chat orchestration service.

Delegates dialogue management to a compiled LangGraph, while retaining
optional backward-compatible services (LLM, memory, guardrails) for
history management and fallback use.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import anyio

from app.config.settings import settings
from app.models.enums.message import MessageStatus
from app.services.chat.citations import normalize_citations
from app.services.chat.coordination import (
    ConversationAccessError,
    ConversationBusyError,
    CoordinationUnavailableError,
    HumanHandlingError,
    SessionCoordinator,
)
from app.services.chat.metrics import (
    CHAT_FIRST_TOKEN_SECONDS,
    CHAT_STREAM_DURATION_SECONDS,
    CHAT_STREAM_OUTCOMES,
)
from app.services.chat.session_identity import ConversationChangedError, require_session_key
from app.services.llm.budget import (
    BudgetState,
    enter_llm_budget,
    record_budget_on_span,
)
from app.services.observability.pipeline_tracer import traced_stage
from app.services.observability.trace_events import TraceEvent, TraceSink, trace_sink_active

if TYPE_CHECKING:
    from app.services.chat.answer_cache import AnswerCacheService
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

# Disconnects cancel Starlette's AnyIO scope. Finalization must finish
# outside that cancellation while keeping a stalled writer bounded.
_STREAM_FINALIZE_TIMEOUT_SECONDS = 2.0


class GraphUnavailableError(RuntimeError):
    """The dialogue graph was never built — chat cannot run.

    The factory's graceful fallback keeps a legacy service object
    alive without a graph; both transports refuse it with this typed
    error so the API can answer 503 instead of letting
    ``NoneType.ainvoke`` surface as a 500 (review 2026-09-26, #12).
    """


logger = logging.getLogger(__name__)


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
        citations: Versioned evidence snapshots used to generate this answer
    """

    content: str
    session_id: int
    intent: str
    sources: list[str] | None = None
    metadata: dict[str, Any] | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    message_id: int | None = None
    status: str | None = None


@dataclass
class ChatMessage:
    """
    Chat message data.

    Attributes:
        role: Message role (user/assistant/system)
        content: Message content
        timestamp: Optional timestamp
        sources: Legacy source document identifiers
        citations: Evidence snapshots stored when the answer was generated
    """

    role: str
    content: str
    timestamp: str | None = None
    sources: list[str] | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    message_id: int | None = None
    status: str | None = None
    user_rating: int | None = None
    ui: dict[str, Any] | None = None


@dataclass(frozen=True)
class PersistedReply:
    """Identity and processing status copied from a committed assistant row."""

    message_id: int
    session_id: int
    status: str


@dataclass(frozen=True)
class CompletionEvent:
    """Opted-in identity of a durably stored, normally completed stream reply."""

    message_id: int
    session_id: int
    status: str = "completed"


@dataclass(frozen=True)
class CitationEvent:
    """Final citation snapshot, emitted only for an opted-in completed stream."""

    citations: list[dict[str, Any]]


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
        answer_cache: AnswerCacheService | None = None,
        coordinator: SessionCoordinator | None = None,
    ) -> None:
        self.graph = graph
        self.llm_service = llm_service
        self.memory_strategy = memory_strategy
        self.guardrail_service = guardrail_service
        self.persister = persister
        self.gap_recorder = gap_recorder
        self.answer_cache = answer_cache
        self.coordinator = coordinator

    async def _automation_thread(
        self,
        session_id: int,
        authenticated_user_id: int | None,
        expected_session_key: str | None = None,
    ) -> str:
        assert self.coordinator is not None
        from app.services.chat.persistence import ensure_demo_session_row

        await ensure_demo_session_row(self.coordinator.sessions, session_id)
        state = await self.coordinator.state(session_id)
        if state.is_private and authenticated_user_id != state.owner_id:
            raise ConversationAccessError("Session unavailable")
        require_session_key(state.session_key, expected_session_key, required=state.is_private)
        if state.claimed_ticket_id is not None:
            raise HumanHandlingError("Human support is handling this conversation")
        return state.thread_id(session_id)

    async def assert_automation_available(
        self,
        session_id: int,
        authenticated_user_id: int | None = None,
        expected_session_key: str | None = None,
    ) -> None:
        """Short preflight before SSE headers; the body reacquires its own gate."""
        if self.coordinator is not None:
            async with self.coordinator.hold(session_id):
                await self._automation_thread(
                    session_id, authenticated_user_id, expected_session_key
                )

    @traced_stage("cs.pipeline")
    async def process_message(
        self,
        session_id: int,
        message: str,
        user_id: int,
        max_tokens: int | None = None,
        *,
        authenticated_user_id: int | None = None,
        expected_session_key: str | None = None,
    ) -> ChatResponse:
        if self.coordinator is None:
            return await self._process_message(session_id, message, user_id, max_tokens)
        async with self.coordinator.hold(session_id):
            async with asyncio.timeout(settings.CHAT_STREAM_MAX_SECONDS):
                thread_id = await self._automation_thread(
                    session_id, authenticated_user_id, expected_session_key
                )
                return await self._process_message(
                    session_id, message, user_id, max_tokens, thread_id=thread_id
                )

    async def _process_message(
        self,
        session_id: int,
        message: str,
        user_id: int,
        max_tokens: int | None = None,  # noqa: ARG002 - reserved for LLM budget wiring
        *,
        thread_id: str | None = None,
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
        if self.graph is None:
            raise GraphUnavailableError("dialogue graph is not initialized")
        cache_epoch = await self.answer_cache.capture_epoch() if self.answer_cache else None
        config = {"configurable": {"thread_id": thread_id or str(session_id)}}
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
        if result.get("turn_id"):
            metadata["turn_id"] = result["turn_id"]
        citations = normalize_citations(result.get("citations"))
        if citations:
            metadata["citations"] = citations
        if result.get("handoff_outcome"):
            metadata["handoff_outcome"] = result["handoff_outcome"]
            if result.get("handoff_ticket_id"):
                metadata["handoff_ticket_id"] = result["handoff_ticket_id"]
        # Durable audit trail of agent tool executions (refunds and
        # other irreversible support actions must be traceable).
        executed_tools = result.get("executed_tools") or []
        if executed_tools:
            metadata["executed_tools"] = executed_tools
        if result.get("action_execution"):
            metadata["action_execution"] = result["action_execution"]

        persisted_reply: PersistedReply | None = None
        if self.persister is not None:
            receipt = await self.persister.persist_turn(
                session_id=session_id,
                user_id=user_id,
                user_message=message,
                response=result.get("response", ""),
                intent=result.get("intent"),
                sources=result.get("sources"),
                metadata=metadata,
            )
            if isinstance(receipt, PersistedReply):
                persisted_reply = receipt
        if self.gap_recorder is not None:
            await self.gap_recorder.record_if_gap(
                query=message,
                intent=result.get("intent"),
                retrieved_docs=result.get("retrieved_docs"),
                session_id=session_id,
                user_id=user_id,
                retrieval_ran=bool(result.get("retrieval_ran", False)),
                retrieval_degraded=bool(result.get("retrieval_degraded", False)),
            )
        await self._maybe_cache_answer(message, result, user_id, expected_epoch=cache_epoch)

        return ChatResponse(
            content=result.get("response", ""),
            session_id=session_id,
            intent=result.get("intent", "unknown"),
            sources=result.get("sources"),
            metadata=metadata,
            citations=citations,
            message_id=persisted_reply.message_id if persisted_reply else None,
            status=persisted_reply.status if persisted_reply else None,
        )

    @traced_stage("cs.pipeline.stream")
    async def process_message_stream(
        self,
        session_id: int,
        message: str,
        user_id: int,
        heartbeat_seconds: float = 15.0,
        stream_max_seconds: float = 120.0,
        include_citations: bool = False,
        include_completion: bool = False,
        *,
        authenticated_user_id: int | None = None,
        expected_session_key: str | None = None,
    ) -> AsyncGenerator[str | TraceEvent | CitationEvent | CompletionEvent, None]:
        options: dict[str, Any] = {
            "heartbeat_seconds": heartbeat_seconds,
            "stream_max_seconds": stream_max_seconds,
            "include_citations": include_citations,
            "include_completion": include_completion,
        }
        if self.coordinator is None:
            async with aclosing(
                self._process_message_stream(session_id, message, user_id, **options)
            ) as stream:
                async for chunk in stream:
                    yield chunk
            return
        final_events: list[CitationEvent | CompletionEvent] = []
        try:
            async with self.coordinator.hold(session_id):
                async with asyncio.timeout(
                    stream_max_seconds + _STREAM_FINALIZE_TIMEOUT_SECONDS + 1
                ):
                    thread_id = await self._automation_thread(
                        session_id, authenticated_user_id, expected_session_key
                    )
                    async with aclosing(
                        self._process_message_stream(
                            session_id, message, user_id, **options, thread_id=thread_id
                        )
                    ) as stream:
                        async for chunk in stream:
                            if isinstance(chunk, (CitationEvent, CompletionEvent)):
                                final_events.append(chunk)
                            else:
                                yield chunk
        except (
            ConversationBusyError,
            CoordinationUnavailableError,
            HumanHandlingError,
            ConversationAccessError,
            ConversationChangedError,
            TimeoutError,
        ):
            # A claim may win after the HTTP preflight. No graph/business
            # operation is retried, and no completion receipt is invented.
            yield STREAM_ERROR
            return
        for event in final_events:
            yield event

    async def _process_message_stream(
        self,
        session_id: int,
        message: str,
        user_id: int,
        heartbeat_seconds: float = 15.0,
        stream_max_seconds: float = 120.0,
        include_citations: bool = False,
        include_completion: bool = False,
        *,
        thread_id: str | None = None,
    ) -> AsyncGenerator[str | TraceEvent | CitationEvent | CompletionEvent, None]:
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

        When ``CHAT_TRACE_STREAM_ENABLED`` a TraceSink is published into
        the graph task's context: nodes mirror their stage lifecycle and
        stage-specific detail onto this queue as TraceEvent items, which
        this consumer yields interleaved with content (in arrival order)
        for the live execution-chain demo panel.

        Args:
            session_id: Session identifier
            message: User message
            user_id: User identifier
            heartbeat_seconds: Idle window before yielding a heartbeat
            stream_max_seconds: Total budget before the stream is cut off
            include_citations: Emit a final citation event after a completed turn
            include_completion: Emit committed reply identity after a completed turn

        Yields:
            Response text, heartbeats and trace events; an opted-in completed
            stream ends with a CitationEvent after persistence finishes.
        """
        if self.graph is None:
            # STREAM_ERROR frame per the stream failure convention —
            # raising inside an async generator would only surface as
            # a dropped connection after SSE headers were already sent.
            yield STREAM_ERROR
            return
        cache_epoch = await self.answer_cache.capture_epoch() if self.answer_cache else None
        queue: asyncio.Queue[str | TraceEvent | None] = asyncio.Queue()
        config = {
            "configurable": {"thread_id": thread_id or str(session_id), "stream_queue": queue}
        }
        # The sentinel is enqueued only after the graph task settles, so
        # the consumer can never exit before the task is observed done.
        # Budget scope wraps the whole stream: the graph task inherits
        # the ContextVar (asyncio copies context at task creation), so
        # every node's LLM call in this turn counts against it. The
        # trace sink rides the same mechanism — created inside the
        # budget scope, scoped exactly around task creation so nodes
        # (and only nodes) inherit it.
        trace_sink = TraceSink(queue) if settings.CHAT_TRACE_STREAM_ENABLED else None
        budget: BudgetState | None = None
        async with enter_llm_budget(settings.CHAT_LLM_CALL_BUDGET) as budget_state:
            budget = budget_state
            if trace_sink is not None:
                with trace_sink_active(trace_sink):
                    invoke_task = asyncio.create_task(
                        self.graph.ainvoke(
                            {"message": message, "session_id": session_id, "user_id": user_id},
                            config,
                        )
                    )
            else:
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
        # 时效 observability (GB/T 47746 响应速度): TTFT to first content,
        # total duration, and the outcome of whichever exit path runs.
        stream_started = loop.time()
        first_token_seen = False
        outcome = "completed"
        result: dict[str, Any] = {}
        persisted_reply: PersistedReply | None = None
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    # Budget exhausted — end the stream rather than hold the
                    # connection open on heartbeats forever.
                    outcome = "budget_exhausted"
                    yield STREAM_ERROR
                    return
                try:
                    chunk = await asyncio.wait_for(
                        queue.get(), timeout=min(heartbeat_seconds, remaining)
                    )
                except TimeoutError:
                    if loop.time() >= deadline:
                        outcome = "budget_exhausted"
                        yield STREAM_ERROR
                        return
                    yield HEARTBEAT
                    continue
                if chunk is None:
                    break
                if isinstance(chunk, TraceEvent):
                    # Pipeline observability riding the same channel —
                    # forwarded in arrival order, excluded from content
                    # accounting (TTFT / persistence stay text-only).
                    yield chunk
                    continue
                if not first_token_seen:
                    first_token_seen = True
                    CHAT_FIRST_TOKEN_SECONDS.observe(loop.time() - stream_started)
                streamed_content.append(chunk)
                yield chunk
            if invoke_task.cancelled():
                outcome = "graph_error"
                yield STREAM_ERROR
                return
            if invoke_task.exception() is not None:
                # The client gets an explicit failure frame instead of a
                # dropped connection; partial content is still persisted.
                outcome = "graph_error"
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
                    retrieval_ran=bool(result.get("retrieval_ran", False)),
                    retrieval_degraded=bool(result.get("retrieval_degraded", False)),
                )
            # Only completed streams cache: budget-exhausted / errored /
            # disconnected turns return before this point, so partial
            # or truncated content is never stored for replay.
            await self._maybe_cache_answer(message, result, user_id, expected_epoch=cache_epoch)
        except (GeneratorExit, asyncio.CancelledError):
            # Client disconnected mid-stream: record it, let the close
            # proceed (partial-content persistence still runs in finally).
            outcome = "client_disconnect"
            raise
        finally:
            # A plain finally is still inside Starlette's cancelled scope:
            # every database await would be cancelled again on disconnect.
            # Shield only this bounded cleanup, then propagate the original
            # cancellation. No detached persistence task outlives the request.
            try:
                with anyio.move_on_after(
                    _STREAM_FINALIZE_TIMEOUT_SECONDS, shield=True
                ) as finalization:
                    if not invoke_task.done():
                        invoke_task.cancel()
                    await asyncio.gather(invoke_task, return_exceptions=True)
                    # Persist exactly the text emitted before termination.
                    if budget is not None:
                        record_budget_on_span(budget)
                    if self.persister is not None and streamed_content:
                        metadata: dict[str, Any] = {
                            "llm_calls": budget.used if budget else 0,
                            "llm_refused": budget.refused if budget else 0,
                        }
                        if result.get("turn_id"):
                            metadata["turn_id"] = result["turn_id"]
                        citations = (
                            normalize_citations(result.get("citations"))
                            if outcome == "completed"
                            else []
                        )
                        if citations:
                            metadata["citations"] = citations
                        if result.get("handoff_outcome"):
                            metadata["handoff_outcome"] = result["handoff_outcome"]
                            if result.get("handoff_ticket_id"):
                                metadata["handoff_ticket_id"] = result["handoff_ticket_id"]
                        executed_tools = result.get("executed_tools") or []
                        if executed_tools:
                            metadata["executed_tools"] = executed_tools
                        if result.get("action_execution"):
                            metadata["action_execution"] = result["action_execution"]
                        receipt = await self.persister.persist_turn(
                            session_id=session_id,
                            user_id=user_id,
                            user_message=message,
                            response="".join(streamed_content),
                            intent=result.get("intent"),
                            sources=result.get("sources") if outcome == "completed" else None,
                            status=(
                                MessageStatus.COMPLETED
                                if outcome == "completed"
                                else MessageStatus.FAILED
                            ),
                            metadata=metadata,
                        )
                        if isinstance(receipt, PersistedReply):
                            persisted_reply = receipt
                if finalization.cancel_called:
                    logger.warning("Chat stream finalization timed out (session=%s)", session_id)
                    if outcome == "completed":
                        outcome = "finalization_timeout"

            finally:
                CHAT_STREAM_DURATION_SECONDS.observe(loop.time() - stream_started)
                CHAT_STREAM_OUTCOMES.labels(outcome=outcome).inc()

        if outcome == "finalization_timeout":
            yield STREAM_ERROR
            return

        if include_citations and outcome == "completed":
            yield CitationEvent(citations=normalize_citations(result.get("citations")))
        if (
            include_completion
            and outcome == "completed"
            and persisted_reply is not None
            and persisted_reply.status == MessageStatus.COMPLETED.value
        ):
            yield CompletionEvent(
                message_id=persisted_reply.message_id,
                session_id=persisted_reply.session_id,
            )

    async def _maybe_cache_answer(
        self,
        message: str,
        result: dict[str, Any],
        user_id: int,
        *,
        expected_epoch: str | None = None,
    ) -> None:
        """Store a completed grounded turn in the L0 answer cache.

        Eligibility is deliberately strict — a turn is replayable only
        when it is stateless (anonymous, no slots in flight, no staged
        irreversible action, no executed tools, not blocked) and
        grounded (sources present). Personalized turns must never be
        written: a hit replays the stored text verbatim at any later
        anonymous visitor, so one user's context must not be baked in.
        That includes session history: an anonymous session's follow-up
        answer is personal to that dialogue (review 2026-09-26, #5) —
        the graph flags such turns ``personalized`` when the generation
        folded in history or user facts.
        The write key uses the post-guardrail sanitized message (the
        read site looks up with sanitized text); put() itself is
        fail-open, so an outage only means "not cached".
        """
        if self.answer_cache is None or user_id:
            return
        response = result.get("response")
        sources = result.get("sources")
        if not response or not sources:
            return
        if (
            result.get("pending_slots")
            or result.get("filled_slots")
            or result.get("pending_confirmation")
            or result.get("state_stack")
            or result.get("task_executed")
            or result.get("pending_refund_review")
            or result.get("executed_tools")
            or result.get("blocked")
            or result.get("personalized")
            or result.get("shared_cache_blocked")
        ):
            return
        await self.answer_cache.put(
            str(result.get("message") or message),
            response=response,
            sources=list(sources),
            intent=result.get("intent", "unknown"),
            expected_epoch=expected_epoch,
            citations=normalize_citations(result.get("citations")),
        )

    async def get_chat_history(
        self,
        session_id: int,
        limit: int = 50,
        *,
        authenticated_user_id: int | None = None,
        expected_session_key: str | None = None,
    ) -> list[ChatMessage]:
        if self.coordinator is None:
            return await self._get_chat_history(session_id, limit)
        async with self.coordinator.hold(session_id):
            state = await self.coordinator.state(session_id)
            if state.is_private and authenticated_user_id != state.owner_id:
                raise ConversationAccessError("Session unavailable")
            require_session_key(state.session_key, expected_session_key, required=state.is_private)
            return await self._get_chat_history(session_id, limit)

    async def _get_chat_history(self, session_id: int, limit: int) -> list[ChatMessage]:
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

    async def clear_chat_history(
        self,
        session_id: int,
        *,
        authenticated_user_id: int | None = None,
        expected_session_key: str | None = None,
    ) -> None:
        if self.coordinator is None:
            await self.discard_session_state(session_id)
            return
        async with self.coordinator.hold(session_id):
            await self._automation_thread(session_id, authenticated_user_id, expected_session_key)
            state = await self.coordinator.state(session_id)
            from uuid import uuid4

            from sqlalchemy import update

            from app.models.database.session import ChatSession

            # Invalidate old commands before any destructive history/checkpoint work.
            # The new epoch cannot read a partially cleared old checkpoint on failure.
            async with self.coordinator.sessions() as db, db.begin():
                changed = await db.execute(
                    update(ChatSession)
                    .where(
                        ChatSession.id == session_id,
                        ChatSession.session_key == state.session_key,
                    )
                    .values(
                        session_key=str(uuid4()), automation_epoch=ChatSession.automation_epoch + 1
                    )
                    .returning(ChatSession.id)
                )
                if changed.scalar_one_or_none() != session_id:
                    raise ConversationChangedError("Conversation changed")
            await self.discard_session_state(session_id, state.epoch)
            from sqlalchemy import delete

            from app.models.database.handoff_message import HandoffMessage

            async with self.coordinator.sessions() as db, db.begin():
                await db.execute(
                    delete(HandoffMessage).where(HandoffMessage.session_id == session_id)
                )

    async def discard_session_state(self, session_id: int, epoch: int = 0) -> None:
        """
        Clear all checkpoint generations and bot history. Caller holds its gate.

        Resetting the conversation must reset the whole dialogue, not
        just the legacy memory strategy: a staged irreversible action
        (``pending_confirmation``) lives in the checkpoint thread and
        would otherwise survive the clear — a later bare "确认" would
        execute a refund the user believed they had discarded (review
        2026-09-26, #8). Every required reset must succeed before the
        API acknowledges completion; failures remain visible for retry.

        Args:
            session_id: Session identifier
        """
        if self.memory_strategy:
            await self.memory_strategy.clear_session(session_id=session_id)
        checkpointer = getattr(self.graph, "checkpointer", None)
        if checkpointer is not None:
            for version in range(epoch + 1):
                thread_id = (
                    str(session_id) if version == 0 else f"{session_id}:automation:{version}"
                )
                await checkpointer.adelete_thread(thread_id)
        if self.persister is not None:
            await self.persister.clear_history(session_id)
