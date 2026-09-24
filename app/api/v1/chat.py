"""
Chat API endpoints.

Provides REST API for chat interactions including message processing,
streaming responses, and history management.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.config.settings import settings
from app.middleware.rate_limiter_redis import EndpointRateLimiter, client_ip_from_request
from app.models.database.user import User
from app.services.chat.chat_service import HEARTBEAT, STREAM_ERROR, ChatService
from app.services.chat.factory import ChatServiceFactory
from app.services.embeddings import EmbeddingFactory
from app.services.llm import LLMFactory
from app.services.retrieval import RetrievalFactory
from app.services.retrieval.keyword_refresh import (
    KEYWORD_INDEX_CHUNKS,
    start_keyword_index_refresher,
)

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

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
    llm_service = LLMFactory.create_from_settings()

    from app.repositories.message_repository import MessageRepository
    from app.repositories.session_repository import SessionRepository

    message_repo = MessageRepository(db)
    session_repo = SessionRepository(db)

    # ── Retrieval pipeline ────────────────────────────────────────────────
    retrieval_pipeline: dict[str, Any] | None = None
    try:
        embedding_service = EmbeddingFactory.create_from_settings()
        qdrant_client = RetrievalFactory.create_vector_client(
            client_type="qdrant",
            url=settings.VECTOR_DB_URL,
            collection_name=settings.VECTOR_COLLECTION_NAME,
            api_key=settings.VECTOR_API_KEY,
            embedding_service=embedding_service,
        )
        retrieval_pipeline = {
            "hybrid_search": RetrievalFactory.create_hybrid_search(
                vector_client=qdrant_client,
            ),
        }
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

    # Initialize reranker
    try:
        if settings.RERANKER_ENABLED and retrieval_pipeline is not None:
            reranker = RetrievalFactory.create_reranker_from_settings(
                llm_service=llm_service,
            )
            retrieval_pipeline["reranker"] = reranker
    except Exception as e:
        logger.warning("Failed to initialize reranker: %s", e)

    # ── GraphRAG services ─────────────────────────────────────────────────
    graph_retrieval_service = None
    global_search_service = None
    multi_path_fusion = None

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
                        llm_service=llm_service,
                        graph_client=graph_client,
                    )

                from app.services.graph.retrieval import GraphEmbeddingSearch

                graph_embedding = GraphEmbeddingSearch(
                    graph_client=graph_client,
                    embedding_service=embedding_service,
                    max_hops=settings.GRAPH_RAG_MAX_HOPS,
                )
                graph_retrieval_service._embedding_search = graph_embedding

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
                llm_service=llm_service,
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

    _chat_service = ChatServiceFactory.create_with_defaults(
        llm_service=llm_service,
        message_repo=message_repo,
        session_repo=session_repo,
        memory_type="optimized",
        intent_type="hybrid",
        retrieval_pipeline=retrieval_pipeline,
        graph_retrieval_service=graph_retrieval_service,
        global_search_service=global_search_service,
        multi_path_fusion=multi_path_fusion,
        slot_filler=slot_filler,
        guardrail_service=guardrail_service,
        persister=ChatMessagePersister(
            session_maker=async_session_maker,
            compressor=SessionCompressor(
                session_maker=async_session_maker,
                llm_service=llm_service,
                threshold=settings.CHAT_SUMMARY_THRESHOLD,
                interval=settings.CHAT_SUMMARY_INTERVAL,
            ),
        ),
        gap_recorder=create_knowledge_gap_recorder(),
        checkpointer=checkpointer,
    )


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
    session_id: int = Field(..., gt=0, description="Session ID")
    user_id: int | None = Field(None, gt=0, description="User ID (optional)")
    max_tokens: int | None = Field(None, gt=0, le=4096, description="Max tokens for response")


class ChatResponse(BaseModel):
    """Chat message response."""

    content: str
    session_id: int
    intent: str
    sources: list[str] | None = None
    metadata: dict[str, Any] | None = None
    dialogue_state: dict[str, Any] | None = None


class ChatMessageResponse(BaseModel):
    """Chat message in history."""

    role: str
    content: str
    timestamp: str | None = None


class ChatHistoryResponse(BaseModel):
    """Chat history response."""

    messages: list[ChatMessageResponse]
    session_id: int


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    request: ChatRequest,
    http_req: Request,
    current_user: User | None = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Process a chat message and generate response."""
    await _enforce_chat_rate_limit(http_req)

    try:
        user_id = current_user.id if current_user else request.user_id

        response = await chat_service.process_message(
            session_id=request.session_id,
            message=request.message,
            user_id=user_id or 0,
            max_tokens=request.max_tokens,
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
            intent=response.intent,
            sources=response.sources,
            metadata=response.metadata,
            dialogue_state=dialogue_state,
        )

    except Exception as e:
        logger.error("Failed to process chat message: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message",
        ) from e


def _sse_frame(chunk: str) -> str:
    """Encode a text chunk as one SSE event (safe for embedded newlines)."""
    lines = chunk.splitlines() or [""]
    return "".join(f"data: {line}\n" for line in lines) + "\n"


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    http_req: Request,
    current_user: User | None = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """Process a chat message with a true token-streaming SSE response."""
    await _enforce_chat_rate_limit(http_req)

    try:
        user_id = current_user.id if current_user else request.user_id

        async def generate() -> AsyncIterator[str]:
            async for chunk in chat_service.process_message_stream(
                session_id=request.session_id,
                message=request.message,
                user_id=user_id or 0,
                stream_max_seconds=settings.CHAT_STREAM_MAX_SECONDS,
            ):
                if chunk == HEARTBEAT:
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

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )

    except Exception as e:
        logger.error("Failed to process streaming message: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message",
        ) from e


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: int = Query(..., gt=0, description="Session ID"),
    limit: int = Query(50, gt=0, le=100, description="Max messages to return"),
    # Ownership check lands with real auth wiring; DI kept so the
    # signature does not change twice.
    current_user: User | None = Depends(get_current_user),  # noqa: ARG001
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatHistoryResponse:
    """Get chat history for a session."""
    try:
        messages = await chat_service.get_chat_history(
            session_id=session_id,
            limit=limit,
        )

        return ChatHistoryResponse(
            messages=[
                ChatMessageResponse(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp,
                )
                for msg in messages
            ],
            session_id=session_id,
        )

    except Exception as e:
        logger.error("Failed to get chat history: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get history",
        ) from e


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
async def clear_chat_history(
    session_id: int = Query(..., gt=0, description="Session ID"),
    # Ownership check lands with real auth wiring; DI kept so the
    # signature does not change twice.
    current_user: User | None = Depends(get_current_user),  # noqa: ARG001
    chat_service: ChatService = Depends(get_chat_service),
) -> None:
    """Clear chat history for a session."""
    try:
        await chat_service.clear_chat_history(session_id=session_id)
    except Exception as e:
        logger.error("Failed to clear chat history: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear history",
        ) from e
