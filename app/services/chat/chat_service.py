"""
Chat orchestration service.

Delegates dialogue management to a compiled LangGraph, while retaining
optional backward-compatible services (LLM, memory, guardrails) for
history management and fallback use.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.config.settings import settings
from app.models.enums.message import MessageStatus
from app.services.chat.metrics import (
    CHAT_FIRST_TOKEN_SECONDS,
    CHAT_STREAM_DURATION_SECONDS,
    CHAT_STREAM_OUTCOMES,
)
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
        answer_cache: AnswerCacheService | None = None,
    ) -> None:
        self.graph = graph
        self.llm_service = llm_service
        self.memory_strategy = memory_strategy
        self.guardrail_service = guardrail_service
        self.persister = persister
        self.gap_recorder = gap_recorder
        self.answer_cache = answer_cache

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
        if self.graph is None:
            raise GraphUnavailableError("dialogue graph is not initialized")
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
        if result.get("turn_id"):
            metadata["turn_id"] = result["turn_id"]
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
                retrieval_ran=bool(result.get("retrieval_ran", False)),
                retrieval_degraded=bool(result.get("retrieval_degraded", False)),
            )
        await self._maybe_cache_answer(message, result, user_id)

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
    ) -> AsyncGenerator[str | TraceEvent, None]:
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

        Yields:
            str | TraceEvent: Response text chunks (HEARTBEAT sentinel
            on idle) interleaved with pipeline trace events
        """
        if self.graph is None:
            # STREAM_ERROR frame per the stream failure convention —
            # raising inside an async generator would only surface as
            # a dropped connection after SSE headers were already sent.
            yield STREAM_ERROR
            return
        queue: asyncio.Queue[str | TraceEvent | None] = asyncio.Queue()
        config = {"configurable": {"thread_id": str(session_id), "stream_queue": queue}}
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
            await self._maybe_cache_answer(message, result, user_id)
        except GeneratorExit:
            # Client disconnected mid-stream: record it, let the close
            # proceed (partial-content persistence still runs in finally).
            outcome = "client_disconnect"
            raise
        finally:
            CHAT_STREAM_DURATION_SECONDS.observe(loop.time() - stream_started)
            CHAT_STREAM_OUTCOMES.labels(outcome=outcome).inc()
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
                metadata: dict[str, Any] = {
                    "llm_calls": budget.used if budget else 0,
                    "llm_refused": budget.refused if budget else 0,
                }
                if result.get("turn_id"):
                    metadata["turn_id"] = result["turn_id"]
                # Durable audit trail of agent tool executions — parity
                # with the non-stream path (review 2026-09-26, #8).
                executed_tools = result.get("executed_tools") or []
                if executed_tools:
                    metadata["executed_tools"] = executed_tools
                await self.persister.persist_turn(
                    session_id=session_id,
                    user_id=user_id,
                    user_message=message,
                    response="".join(streamed_content),
                    intent=result.get("intent"),
                    sources=result.get("sources"),
                    # Abnormal exits (disconnect / graph error / budget
                    # exhaustion) never reach the result assignment, so
                    # intent and sources stay None and the status below
                    # marks them FAILED — a cut-off turn must remain
                    # distinguishable from a finished one in the audit
                    # trail (review 2026-09-26, #8).
                    status=(
                        MessageStatus.COMPLETED if outcome == "completed" else MessageStatus.FAILED
                    ),
                    metadata=metadata,
                )

    async def _maybe_cache_answer(self, message: str, result: dict[str, Any], user_id: int) -> None:
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
            or result.get("pending_confirmation")
            or result.get("executed_tools")
            or result.get("blocked")
            or result.get("personalized")
        ):
            return
        await self.answer_cache.put(
            str(result.get("message") or message),
            response=response,
            sources=list(sources),
            intent=result.get("intent", "unknown"),
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

        Resetting the conversation must reset the whole dialogue, not
        just the legacy memory strategy: a staged irreversible action
        (``pending_confirmation``) lives in the checkpoint thread and
        would otherwise survive the clear — a later bare "确认" would
        execute a refund the user believed they had discarded (review
        2026-09-26, #8). Checkpoint deletion is fail-open: a failing
        clear must never 500 the history endpoint.

        Args:
            session_id: Session identifier
        """
        if self.memory_strategy:
            await self.memory_strategy.clear_session(session_id=session_id)
        checkpointer = getattr(self.graph, "checkpointer", None)
        if checkpointer is not None:
            try:
                await checkpointer.adelete_thread(str(session_id))
            except Exception:  # noqa: BLE001 - clear must never fail the API
                logger.warning(
                    "Failed to clear dialogue checkpoint for session %s",
                    session_id,
                    exc_info=True,
                )
