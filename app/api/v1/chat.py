"""
Chat API endpoints.

Provides REST API for chat interactions including message processing,
streaming responses, and history management.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.types import Send

from app.api.deps import get_current_user
from app.api.deps.authorization import ensure_session_access, resolve_chat_identity
from app.api.deps.private_response import PrivateAPIRoute
from app.api.deps.publication import get_publication_guard
from app.api.v1.sessions import get_session_service
from app.config.settings import settings
from app.middleware.rate_limiter_redis import EndpointRateLimiter, client_ip_from_request
from app.models.database.user import User
from app.models.enums.message import MessageStatus
from app.models.schemas.citation import SourceCitation
from app.models.schemas.session_key import SessionKey
from app.services.chat.chat_service import (
    HEARTBEAT,
    STREAM_ERROR,
    ChatService,
    CitationEvent,
    CompletionEvent,
    GraphUnavailableError,
)
from app.services.chat.citations import normalize_citations
from app.services.chat.coordination import (
    ConversationAccessError,
    ConversationBusyError,
    CoordinationUnavailableError,
    HumanHandlingError,
    get_session_coordinator,
)
from app.services.chat.factory import ChatServiceFactory
from app.services.chat.session_identity import ConversationChangedError
from app.services.embeddings import EmbeddingFactory
from app.services.handoff.containment_metrics import start_containment_refresher
from app.services.llm import LLMFactory
from app.services.observability.trace_events import TraceEvent
from app.services.retrieval import RetrievalFactory
from app.services.retrieval.keyword_refresh import (
    KEYWORD_INDEX_CHUNKS,
    register_keyword_index,
    start_keyword_index_refresher,
)
from app.services.session_service import SessionService

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"], route_class=PrivateAPIRoute)

# Global chat service instance (initialized on startup)
_chat_service: ChatService | None = None

# Tighter per-IP budget for LLM-backed turns. Redis-backed sliding window
# so N replicas enforce one shared limit (a per-process limiter would give
# each replica its own budget at scale).
_chat_rate_limiter = EndpointRateLimiter(
    scope="chat",
    requests_per_minute=settings.CHAT_RATE_LIMIT_REQUESTS_PER_MINUTE,
    window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
)


async def _enforce_chat_rate_limit(http_req: Request) -> None:
    """Reject with 429 when the caller has exhausted the chat-path budget."""
    if not settings.RATE_LIMIT_ENABLED:
        return
    allowed, _remaining = await _chat_rate_limiter.check(client_ip_from_request(http_req))
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Chat request rate limit exceeded. Please retry shortly.",
            headers={"Retry-After": str(settings.RATE_LIMIT_WINDOW_SECONDS)},
        )


async def initialize_chat_service(
    db: AsyncSession, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> None:
    """Initialize the RAG chat service with all dependencies."""
    global _chat_service

    from app.config.settings import get_settings

    settings = get_settings()
    from sqlalchemy import select

    from app.models.database.action_execution import ActionExecution
    from app.models.database.handoff_message import HandoffMessage
    from app.models.database.session import ChatSession

    # A graph alone cannot provide the private support contract on an older
    # schema. Refuse chat readiness before constructing external providers.
    await db.execute(select(ChatSession.automation_epoch, ChatSession.session_key).limit(0))
    await db.execute(select(HandoffMessage.id).limit(0))
    await db.execute(select(ActionExecution.action_id).limit(0))
    llm_service = LLMFactory.create_from_settings()

    # Two tiers, one budget: classification/extraction (rerank, slot
    # filling, Text-to-Cypher) run on the light model when
    # CHAT_LLM_LIGHT_MODEL is set; generation and agent tool rounds
    # stay on the primary. Request-path consumers get the budget
    # wrapper — the reranker and slot filler run INSIDE graph nodes,
    # so an unwrapped service would silently bypass the per-request
    # LLM budget. The background compressor keeps the raw light
    # service (off request budget by design).
    from app.services.llm.budget import BudgetedLLMService

    light_llm_service = LLMFactory.create_light_from_settings(llm_service)
    budgeted_light_llm = BudgetedLLMService(light_llm_service)

    from app.repositories.message_repository import MessageRepository
    from app.repositories.session_repository import SessionRepository

    message_repo = MessageRepository(db)
    session_repo = SessionRepository(db)

    # ── Retrieval pipeline ────────────────────────────────────────────────
    # Keep the publication policy even when a retrieval dependency cannot
    # initialize: cached answers still need committed-source verification.
    publication_guard = get_publication_guard()
    retrieval_pipeline: dict[str, Any] = {"publication_guard": publication_guard}
    try:
        embedding_service = EmbeddingFactory.create_from_settings()
        qdrant_client = RetrievalFactory.create_vector_client(
            client_type="qdrant",
            url=settings.VECTOR_DB_URL,
            collection_name=settings.VECTOR_COLLECTION_NAME,
            api_key=settings.VECTOR_API_KEY,
            embedding_service=embedding_service,
        )
        retrieval_pipeline["hybrid_search"] = RetrievalFactory.create_hybrid_search(
            vector_client=qdrant_client,
            publication_guard=publication_guard,
        )
        # Publish this process's keyword leg as the delete-time purge
        # target (see purge_keyword_document): a deleted document must
        # leave the BM25 index when the vectors leave Qdrant, not one
        # refresh interval later (review 2026-09-26, #7).
        register_keyword_index(retrieval_pipeline["hybrid_search"])
        # Warm the BM25 leg from the vector corpus so hybrid RRF runs on
        # both legs; failure degrades to vector-only (availability over
        # strictness).
        if settings.KEYWORD_INDEX_WARMUP_ENABLED:
            try:
                warmed = await RetrievalFactory.warm_keyword_index(
                    hybrid_search=retrieval_pipeline["hybrid_search"],
                    vector_client=qdrant_client,
                )
                KEYWORD_INDEX_CHUNKS.set(warmed)
                logger.info("Keyword index warmed with %d chunks", warmed)
            except Exception as e:
                logger.warning("Keyword index warmup skipped (vector-only): %s", e)
        # Periodic rebuild keeps the BM25 leg in sync for documents
        # ingested or deleted after boot (stale entries would surface
        # deleted content — a correctness bug, not just a recall gap).
        if settings.KEYWORD_INDEX_REFRESH_SECONDS > 0:
            start_keyword_index_refresher(
                hybrid_search=retrieval_pipeline["hybrid_search"],
                vector_client=qdrant_client,
                interval_seconds=settings.KEYWORD_INDEX_REFRESH_SECONDS,
            )
    except Exception as e:
        logger.warning("Failed to initialize retrieval pipeline: %s", e)

    # North-star KPI bridge: the session-level containment rate lives in
    # Postgres, so a periodic refresh exports it to Prometheus Gauges
    # (Grafana panel + ContainmentRateLow alert consume them). Independent
    # of the retrieval stack above — a retrieval outage must not blind
    # the KPI, and vice versa. Fail-open: a DB outage freezes the
    # gauges at their last values.
    if settings.CONTAINMENT_STATS_REFRESH_SECONDS > 0:
        try:
            from app.api.database import async_session_maker

            start_containment_refresher(
                session_maker=async_session_maker,
                interval_seconds=settings.CONTAINMENT_STATS_REFRESH_SECONDS,
                window_days=settings.CONTAINMENT_STATS_WINDOW_DAYS,
            )
        except Exception as e:
            logger.warning("Failed to start containment KPI refresher: %s", e)

    # Initialize reranker
    try:
        if settings.RERANKER_ENABLED and retrieval_pipeline is not None:
            reranker = RetrievalFactory.create_reranker_from_settings(
                llm_service=budgeted_light_llm,
            )
            retrieval_pipeline["reranker"] = reranker
    except Exception as e:
        logger.warning("Failed to initialize reranker: %s", e)

    # ── GraphRAG services ─────────────────────────────────────────────────
    graph_retrieval_service = None
    global_search_service = None
    multi_path_fusion = None
    commerce_graph_service = None
    try:
        if settings.GRAPH_RAG_ENABLED:
            from app.services.graph import GraphFactory
            from app.services.graph.retrieval import (
                GraphRetrievalService,
                MultiPathRetrievalFusion,
            )

            graph_client = GraphFactory.create_from_settings()
            if graph_client:
                await graph_client.connect()

                # Reuse the connected client for the optional GraphRAG HTTP API.
                from app.api.v1.graph import set_graph_client

                set_graph_client(graph_client)

                graph_retrieval_service = GraphRetrievalService(
                    text_to_cypher=None,
                    graph_embedding_search=None,
                )
                global_search_service = None
                multi_path_fusion = MultiPathRetrievalFusion(
                    vector_weight=1.0 - settings.GRAPH_RAG_FUSION_WEIGHT,
                    graph_weight=settings.GRAPH_RAG_FUSION_WEIGHT,
                )

                if settings.GRAPH_RAG_TEXT_TO_CYPHER_ENABLED:
                    from app.services.graph.retrieval import TextToCypherService

                    graph_retrieval_service._cypher = TextToCypherService(
                        llm_service=budgeted_light_llm,
                        graph_client=graph_client,
                    )

                from app.services.graph.retrieval import GraphEmbeddingSearch

                graph_embedding = GraphEmbeddingSearch(
                    graph_client=graph_client,
                    embedding_service=embedding_service,
                    max_hops=settings.GRAPH_RAG_MAX_HOPS,
                )
                graph_retrieval_service._embedding_search = graph_embedding

                # Product semantics use the same Neo4j infrastructure, but as
                # a typed projection separate from document provenance. The
                # provider taxonomy remains authoritative and no SKUs/prices
                # are copied into the graph.
                from app.api.database import async_session_maker
                from app.services.commerce import create_commerce_port
                from app.services.commerce.graph_service import CommerceGraphService

                commerce_graph_service = CommerceGraphService(
                    graph_client=graph_client,
                    embedding_service=embedding_service,
                    semantic_planner=llm_service,
                )
                try:
                    from app.services.commerce.taxonomy import CommerceTaxonomyRegistry

                    taxonomy_registry = CommerceTaxonomyRegistry(async_session_maker)
                    taxonomy = await create_commerce_port(async_session_maker).catalog_categories(1)
                    semantic_data = await taxonomy_registry.load_semantics()
                    projected = await commerce_graph_service.refresh_projection(
                        taxonomy,
                        semantic_data,
                    )
                    logger.info("Projected commerce taxonomy graph: %s", projected)
                except Exception as e:
                    logger.warning(
                        "Commerce taxonomy graph projection failed; graph tool stays available for existing nodes: %s",
                        e,
                    )

                if settings.GRAPH_RAG_COMMUNITY_ENABLED:
                    from app.services.graph.community import GlobalSearchService

                    global_search_service = GlobalSearchService(
                        graph_client=graph_client,
                        embedding_service=embedding_service,
                    )
    except Exception as e:
        logger.warning("Failed to initialize GraphRAG services: %s", e)

    # ── Slot filling ──────────────────────────────────────────────────────
    slot_filler = None
    try:
        if settings.SLOT_FILLING_ENABLED:
            from app.services.slot_filling.factory import SlotFillerFactory

            slot_filler = SlotFillerFactory.create(
                filler_type=settings.SLOT_FILLING_TYPE,
                llm_service=budgeted_light_llm,
            )
    except Exception as e:
        logger.warning("Failed to initialize slot filling: %s", e)

    # ── Guardrails ────────────────────────────────────────────────────────
    guardrail_service = None
    try:
        from app.services.guardrails.factory import GuardrailFactory

        guardrail_service = GuardrailFactory.create_from_settings()
    except Exception as e:
        logger.warning("Failed to initialize guardrails: %s", e)

    # ── Build ChatService via factory (graph constructed internally) ──────
    from app.api.database import async_session_maker
    from app.services.chat.compressor import SessionCompressor
    from app.services.chat.knowledge_gap_recorder import create_knowledge_gap_recorder
    from app.services.chat.persistence import ChatMessagePersister
    from app.services.chat.user_fact_extractor import UserFactExtractor

    _chat_service = ChatServiceFactory.create_with_defaults(
        llm_service=llm_service,
        light_llm_service=light_llm_service,
        message_repo=message_repo,
        session_repo=session_repo,
        memory_type="optimized",
        intent_type="hybrid",
        retrieval_pipeline=retrieval_pipeline,
        graph_retrieval_service=graph_retrieval_service,
        global_search_service=global_search_service,
        multi_path_fusion=multi_path_fusion,
        commerce_graph_service=commerce_graph_service,
        slot_filler=slot_filler,
        guardrail_service=guardrail_service,
        persister=ChatMessagePersister(
            session_maker=async_session_maker,
            compressor=SessionCompressor(
                session_maker=async_session_maker,
                llm_service=light_llm_service,
                threshold=settings.CHAT_SUMMARY_THRESHOLD,
                interval=settings.CHAT_SUMMARY_INTERVAL,
            ),
            fact_extractor=UserFactExtractor(
                session_maker=async_session_maker,
                llm_service=light_llm_service if settings.USER_FACT_EXTRACTION_ENABLED else None,
                threshold=settings.USER_FACT_EXTRACTION_THRESHOLD,
                interval=settings.USER_FACT_EXTRACTION_INTERVAL,
            ),
        ),
        gap_recorder=create_knowledge_gap_recorder(),
        checkpointer=checkpointer,
        coordinator=get_session_coordinator(),
    )
    if _chat_service.graph is None:
        # Factory-level graceful fallback keeps a graphless service for
        # legacy callers, but the demo API cannot serve chat without
        # the graph — refuse readiness instead of a green /ready that
        # 500s on every message (review 2026-09-26, #12).
        raise RuntimeError("Dialogue graph failed to build — refusing readiness")


def get_chat_service() -> ChatService:
    """Get chat service instance. Raises 503 if not initialized."""
    global _chat_service
    if _chat_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service not initialized. Please ensure all dependencies are configured.",
        )
    return _chat_service


# ── Request / Response schemas ────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Chat message request."""

    message: str = Field(..., min_length=1, description="User message")
    session_id: int = Field(..., gt=0, le=2_147_483_647, description="Session ID")
    expected_session_key: SessionKey | None = None
    user_id: int | None = Field(None, gt=0, description="User ID (optional)")
    max_tokens: int | None = Field(None, gt=0, le=4096, description="Max tokens for response")
    include_citations: bool = Field(False, description="Emit the final citations event in SSE")
    include_completion: bool = Field(False, description="Emit committed reply identity in SSE")


class ChatResponse(BaseModel):
    """Chat message response."""

    content: str
    session_id: int
    session_key: SessionKey | None = None
    intent: str
    sources: list[str] | None = None
    citations: list[SourceCitation] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None
    dialogue_state: dict[str, Any] | None = None
    message_id: int | None = None
    status: str | None = None
    ui: dict[str, Any] | None = None


class ChatMessageResponse(BaseModel):
    """Chat message in history."""

    role: str
    content: str
    timestamp: str | None = None
    sources: list[str] | None = None
    citations: list[SourceCitation] = Field(default_factory=list)
    message_id: int | None = None
    status: str | None = None
    user_rating: int | None = None
    ui: dict[str, Any] | None = None


class ChatHistoryResponse(BaseModel):
    """Chat history response."""

    messages: list[ChatMessageResponse]
    session_id: int
    session_key: SessionKey | None = None


def _stored_message_fields(message: Any) -> dict[str, Any]:
    """Legacy service adapters may not carry durable-message fields yet."""
    message_id = getattr(message, "message_id", None)
    processing_status = getattr(message, "status", None)
    return {
        "message_id": message_id if type(message_id) is int and message_id > 0 else None,
        "status": (
            processing_status
            if isinstance(processing_status, str)
            and processing_status in {member.value for member in MessageStatus}
            else None
        ),
    }


def _stored_user_rating(message: Any) -> int | None:
    rating = getattr(message, "user_rating", None)
    return rating if type(rating) is int and rating in {-1, 0, 1} else None


def _stored_commerce_ui(value: Any) -> dict[str, Any] | None:
    """Only typed server snapshots become executable client data."""
    from pydantic import ValidationError

    from app.models.schemas.commerce import CommerceView

    if not isinstance(value, dict):
        return None
    try:
        return CommerceView.model_validate(value).model_dump(mode="json")
    except ValidationError:
        # Preserve the ordinary transcript if an old or corrupt UI snapshot
        # cannot be interpreted. Domain status can still be queried explicitly.
        return None


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    request: ChatRequest,
    http_req: Request,
    current_user: User | None = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
    session_service: SessionService = Depends(get_session_service),
) -> ChatResponse:
    """Process a chat message and generate response."""
    await _enforce_chat_rate_limit(http_req)
    # Identity comes from the auth context; the body user_id only
    # matters in DEMO_MODE. Before the try block on purpose — the
    # 401/403 must not be swallowed into a 500 by the handler below
    # (review 2026-09-26, #1).
    user_id = resolve_chat_identity(current_user, request.user_id)
    if isinstance(chat_service, ChatService) and chat_service.graph is None:
        raise HTTPException(503, "Chat service graph not initialized")
    session_key = await ensure_session_access(
        request.session_id,
        current_user,
        session_service,
        expected_session_key=request.expected_session_key,
        require_key=True,
    )

    try:
        identity_options: dict[str, Any] = (
            {
                "authenticated_user_id": current_user.id if current_user else None,
                "expected_session_key": request.expected_session_key,
            }
            if isinstance(chat_service, ChatService)
            else {}
        )
        response = await chat_service.process_message(
            session_id=request.session_id,
            message=request.message,
            user_id=user_id or 0,
            max_tokens=request.max_tokens,
            **identity_options,
        )

        # Build dialogue_state from metadata when present
        dialogue_state = None
        if response.metadata:
            dialogue_state = {
                "phase": response.intent,
                "pending_slots": response.metadata.get("pending_slots", []),
                "filled_slots": response.metadata.get("filled_slots", {}),
            }

        return ChatResponse(
            content=response.content,
            session_id=response.session_id,
            session_key=session_key,
            intent=response.intent,
            sources=response.sources,
            citations=[
                SourceCitation.model_validate(citation)
                for citation in normalize_citations(getattr(response, "citations", None))
            ],
            metadata=response.metadata,
            dialogue_state=dialogue_state,
            **_stored_message_fields(response),
        )

    except ConversationChangedError as e:
        raise HTTPException(409, "Conversation changed; reopen it before continuing") from e
    except ConversationAccessError as e:
        raise HTTPException(404, "Session not found") from e
    except ConversationBusyError as e:
        raise HTTPException(409, "A conversation action is still in progress") from e
    except HumanHandlingError as e:
        raise HTTPException(423, "Human support is handling this conversation") from e
    except CoordinationUnavailableError as e:
        raise HTTPException(503, "Conversation coordination is unavailable") from e
    except TimeoutError as e:
        raise HTTPException(
            504, "Conversation processing timed out; verify its status before retrying"
        ) from e
    except GraphUnavailableError as e:
        # Degraded window before /ready flips: retryable 503, not a
        # generic 500 (review 2026-09-26, #12).
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat engine not initialized",
        ) from e

    except Exception as e:
        logger.error("Failed to process chat message: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message",
        ) from e


def _sse_frame(chunk: str) -> str:
    """Encode a text chunk as one SSE event (safe for embedded newlines)."""
    # Each SSE data field contributes a newline; the parser removes only the
    # final separator. Keep empty trailing fields so a newline in the actual
    # content, including a newline-only chunk, survives that decoding step.
    # Canonicalize CR delimiters first so they cannot inject SSE fields.
    lines = chunk.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "".join(f"data: {line}\n" for line in lines) + "\n"


def _sse_trace_frame(event: TraceEvent) -> str:
    """Encode a pipeline trace event as a NAMED SSE event.

    ``event: trace`` frames are invisible to EventSource consumers that
    listen only to ``message`` — the content contract is unchanged —
    while the execution-chain demo panel subscribes to ``trace``.
    """
    payload = json.dumps(event.payload(), ensure_ascii=False)
    return f"event: trace\ndata: {payload}\n\n"


class _ClosingStreamingResponse(StreamingResponse):
    """Close the body in the sending task, including ASGI send failures."""

    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            # ASGI 2.4 reports a disconnect by raising from send(), while
            # the body generator is paused at yield. Close it here so its
            # graph/persistence finalizer runs before the request returns.
            close = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                await close()


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    http_req: Request,
    current_user: User | None = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
    session_service: SessionService = Depends(get_session_service),
) -> StreamingResponse:
    """Process a chat message with a true token-streaming SSE response."""
    await _enforce_chat_rate_limit(http_req)
    # Same identity rule as the non-streaming entry (review 2026-09-26, #1):
    # token identity in strict mode, body id only under DEMO_MODE.
    user_id = resolve_chat_identity(current_user, request.user_id)
    if isinstance(chat_service, ChatService) and chat_service.graph is None:
        raise HTTPException(503, "Chat service graph not initialized")
    await ensure_session_access(
        request.session_id,
        current_user,
        session_service,
        expected_session_key=request.expected_session_key,
        require_key=True,
    )

    try:
        if isinstance(chat_service, ChatService):
            await chat_service.assert_automation_available(
                request.session_id,
                current_user.id if current_user else None,
                request.expected_session_key,
            )

        async def generate() -> AsyncIterator[str]:
            stream_options: dict[str, Any] = {}
            if request.include_citations:
                stream_options["include_citations"] = True
            if request.include_completion:
                stream_options["include_completion"] = True
            if isinstance(chat_service, ChatService):
                stream_options["authenticated_user_id"] = current_user.id if current_user else None
                stream_options["expected_session_key"] = request.expected_session_key
            async with aclosing(
                chat_service.process_message_stream(
                    session_id=request.session_id,
                    message=request.message,
                    user_id=user_id or 0,
                    stream_max_seconds=settings.CHAT_STREAM_MAX_SECONDS,
                    **stream_options,
                )
            ) as stream:
                async for chunk in stream:
                    if isinstance(chunk, CompletionEvent):
                        if request.include_completion:
                            receipt: dict[str, Any] = {
                                "message_id": chunk.message_id,
                                "session_id": chunk.session_id,
                                "status": chunk.status,
                            }
                            if request.expected_session_key is not None:
                                receipt["session_key"] = request.expected_session_key
                            payload = json.dumps(receipt)
                            yield f"event: completion\ndata: {payload}\n\n"
                    elif isinstance(chunk, CitationEvent):
                        if request.include_citations:
                            payload = json.dumps(
                                {"citations": normalize_citations(chunk.citations)},
                                ensure_ascii=False,
                            )
                            yield f"event: citations\ndata: {payload}\n\n"
                    elif isinstance(chunk, TraceEvent):
                        # Pipeline execution chain for the demo panel — a
                        # named event, so content-only consumers never see it.
                        yield _sse_trace_frame(chunk)
                    elif chunk == HEARTBEAT:
                        # SSE comment keepalive — ignored by EventSource parsers,
                        # keeps proxies from closing an idle connection.
                        yield ": ping\n\n"
                    elif chunk == STREAM_ERROR:
                        # Explicit failure frame so the client can surface an
                        # error instead of waiting on a dropped connection.
                        yield "data: [STREAM_ERROR]\n\n"
                    else:
                        yield _sse_frame(chunk)
            yield "data: [DONE]\n\n"

        return _ClosingStreamingResponse(
            generate(),
            media_type="text/event-stream",
        )

    except ConversationChangedError as e:
        raise HTTPException(409, "Conversation changed; reopen it before continuing") from e
    except ConversationAccessError as e:
        raise HTTPException(404, "Session not found") from e
    except ConversationBusyError as e:
        raise HTTPException(409, "A conversation action is still in progress") from e
    except HumanHandlingError as e:
        raise HTTPException(423, "Human support is handling this conversation") from e
    except CoordinationUnavailableError as e:
        raise HTTPException(503, "Conversation coordination is unavailable") from e
    except Exception as e:
        logger.error("Failed to process streaming message: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message",
        ) from e


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: int = Query(..., gt=0, description="Session ID"),
    expected_session_key: SessionKey | None = Query(None),
    limit: int = Query(50, gt=0, le=100, description="Max messages to return"),
    current_user: User | None = Depends(get_current_user),
    session_service: SessionService = Depends(get_session_service),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatHistoryResponse:
    """Get chat history for a session.

    Strict mode (DEMO_MODE=false) verifies the caller owns the
    session first — foreign and missing sessions both read as 404
    (review 2026-09-26, #1).
    """
    session_key = await ensure_session_access(
        session_id,
        current_user,
        session_service,
        expected_session_key=expected_session_key,
        require_key=True,
    )
    try:
        identity_options: dict[str, Any] = (
            {
                "authenticated_user_id": current_user.id if current_user else None,
                "expected_session_key": expected_session_key,
            }
            if isinstance(chat_service, ChatService)
            else {}
        )
        messages = await chat_service.get_chat_history(
            session_id=session_id,
            limit=limit,
            **identity_options,
        )

        return ChatHistoryResponse(
            messages=[
                ChatMessageResponse(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp,
                    sources=msg.sources if isinstance(msg.sources, list) else None,
                    user_rating=_stored_user_rating(msg),
                    ui=_stored_commerce_ui(getattr(msg, "ui", None)),
                    **_stored_message_fields(msg),
                    citations=[
                        SourceCitation.model_validate(citation)
                        for citation in normalize_citations(getattr(msg, "citations", None))
                    ],
                )
                for msg in messages
            ],
            session_id=session_id,
            session_key=session_key,
        )

    except ConversationChangedError as e:
        raise HTTPException(409, "Conversation changed; reopen it before continuing") from e
    except ConversationAccessError as e:
        raise HTTPException(404, "Session not found") from e
    except ConversationBusyError as e:
        raise HTTPException(409, "A conversation action is still in progress") from e
    except CoordinationUnavailableError as e:
        raise HTTPException(503, "Conversation coordination is unavailable") from e
    except Exception as e:
        logger.error("Failed to get chat history: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get history",
        ) from e


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
async def clear_chat_history(
    session_id: int = Query(..., gt=0, description="Session ID"),
    expected_session_key: SessionKey | None = Query(None),
    current_user: User | None = Depends(get_current_user),
    session_service: SessionService = Depends(get_session_service),
    chat_service: ChatService = Depends(get_chat_service),
) -> None:
    """Clear chat history for a session.

    Same ownership rule as the history read (review 2026-09-26, #1):
    wiping another visitor's dialogue must not be a bare
    ``DELETE ?session_id=`` away.
    """
    await ensure_session_access(
        session_id,
        current_user,
        session_service,
        expected_session_key=expected_session_key,
        require_key=True,
    )
    try:
        identity_options: dict[str, Any] = (
            {
                "authenticated_user_id": current_user.id if current_user else None,
                "expected_session_key": expected_session_key,
            }
            if isinstance(chat_service, ChatService)
            else {}
        )
        await chat_service.clear_chat_history(session_id=session_id, **identity_options)
    except ConversationChangedError as e:
        raise HTTPException(409, "Conversation changed; reopen it before continuing") from e
    except ConversationAccessError as e:
        raise HTTPException(404, "Session not found") from e
    except ConversationBusyError as e:
        raise HTTPException(409, "A conversation action is still in progress") from e
    except HumanHandlingError as e:
        raise HTTPException(423, "Human support is handling this conversation") from e
    except CoordinationUnavailableError as e:
        raise HTTPException(503, "Conversation coordination is unavailable") from e
    except Exception as e:
        logger.error("Failed to clear chat history: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear history",
        ) from e
