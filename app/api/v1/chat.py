"""
Chat API endpoints.

Provides REST API for chat interactions including message processing,
streaming responses, and history management.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from fastapi.responses import StreamingResponse
from typing import Optional, List
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.api.rate_limit import check_rate_limit
from app.models.database.user import User
from app.services.chat.chat_service import ChatService
from app.services.chat.factory import ChatServiceFactory
from app.services.llm import LLMFactory
from app.services.embeddings import EmbeddingFactory
from app.services.retrieval import RetrievalFactory
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# Global chat service instance (initialized on startup)
_chat_service: Optional[ChatService] = None


async def initialize_chat_service(db: AsyncSession):
    """Initialize the RAG chat service with all dependencies."""
    global _chat_service

    llm_service = LLMFactory.create_from_settings()

    from app.repositories.message_repository import MessageRepository
    from app.repositories.session_repository import SessionRepository
    message_repo = MessageRepository(db)
    session_repo = SessionRepository(db)

    try:
        embedding_service = EmbeddingFactory.create_from_settings()
        qdrant_client = RetrievalFactory.create_vector_client(
            client_type="qdrant",
            url="http://qdrant:6333",
            collection_name="documents",
            embedding_service=embedding_service,
        )
        retrieval_pipeline = {
            "hybrid_search": RetrievalFactory.create_hybrid_search(
                vector_client=qdrant_client,
            ),
        }
    except Exception as e:
        logger.warning("Failed to initialize retrieval pipeline: %s", e)
        retrieval_pipeline = None

    # Initialize reranker
    try:
        from app.config.settings import get_settings
        settings = get_settings()
        if settings.RERANKER_ENABLED and retrieval_pipeline is not None:
            reranker = RetrievalFactory.create_reranker_from_settings(
                llm_service=llm_service,
            )
            retrieval_pipeline["reranker"] = reranker
    except Exception as e:
        logger.warning("Failed to initialize reranker: %s", e)

    _chat_service = ChatServiceFactory.create_with_defaults(
        llm_service=llm_service,
        message_repo=message_repo,
        session_repo=session_repo,
        memory_type="optimized",
        intent_type="hybrid",
        retrieval_pipeline=retrieval_pipeline,
    )

    # Initialize GraphRAG services if enabled
    try:
        from app.config.settings import get_settings
        settings = get_settings()
        if settings.GRAPH_RAG_ENABLED:
            from app.services.graph import GraphFactory
            from app.services.graph.retrieval import (
                GraphRetrievalService,
                MultiPathRetrievalFusion,
            )

            graph_client = GraphFactory.create_from_settings()
            if graph_client:
                await graph_client.connect()

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

                _chat_service.graph_retrieval_service = graph_retrieval_service
                _chat_service.global_search_service = global_search_service
                _chat_service.multi_path_fusion = multi_path_fusion
    except Exception as e:
        logger.warning("Failed to initialize GraphRAG services: %s", e)

    # Initialize slot filling
    try:
        from app.config.settings import get_settings
        settings = get_settings()
        if settings.SLOT_FILLING_ENABLED:
            from app.services.slot_filling.factory import SlotFillerFactory
            slot_filler = SlotFillerFactory.create(
                filler_type=settings.SLOT_FILLING_TYPE,
                llm_service=llm_service,
            )
            _chat_service.slot_filler = slot_filler
    except Exception as e:
        logger.warning("Failed to initialize slot filling: %s", e)

    # Initialize guardrails
    try:
        from app.services.guardrails.factory import GuardrailFactory
        guardrail_service = GuardrailFactory.create_from_settings()
        if guardrail_service:
            _chat_service.guardrail_service = guardrail_service
    except Exception as e:
        logger.warning("Failed to initialize guardrails: %s", e)


def get_chat_service() -> ChatService:
    """Get chat service instance. Raises 503 if not initialized."""
    global _chat_service
    if _chat_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service not initialized. Please ensure all dependencies are configured.",
        )
    return _chat_service


# Request/Response Schemas
class ChatRequest(BaseModel):
    """Chat message request."""
    message: str = Field(..., min_length=1, description="User message")
    session_id: int = Field(..., gt=0, description="Session ID")
    user_id: Optional[int] = Field(None, gt=0, description="User ID (optional)")
    max_tokens: Optional[int] = Field(None, gt=0, le=4096, description="Max tokens for response")


class ChatResponse(BaseModel):
    """Chat message response."""
    content: str
    session_id: int
    intent: str
    sources: Optional[List[str]] = None
    metadata: Optional[dict] = None


class ChatMessageResponse(BaseModel):
    """Chat message in history."""
    role: str
    content: str
    timestamp: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    """Chat history response."""
    messages: List[ChatMessageResponse]
    session_id: int


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    request: ChatRequest,
    http_req: Request,
    current_user: Optional[User] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Process a chat message and generate response."""
    check_rate_limit(http_req)

    try:
        user_id = current_user.id if current_user else request.user_id

        response = await chat_service.process_message(
            session_id=request.session_id,
            message=request.message,
            user_id=user_id or 0,
            max_tokens=request.max_tokens,
        )

        return ChatResponse(
            content=response.content,
            session_id=response.session_id,
            intent=response.intent,
            sources=response.sources,
            metadata=response.metadata,
        )

    except Exception as e:
        logger.error("Failed to process chat message: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process message",
        )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    http_req: Request,
    current_user: Optional[User] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Process a chat message with streaming response."""
    check_rate_limit(http_req)

    try:
        user_id = current_user.id if current_user else request.user_id

        async def generate():
            async for chunk in chat_service.process_message_stream(
                session_id=request.session_id,
                message=request.message,
                user_id=user_id or 0,
            ):
                yield f"data: {chunk}\n\n"
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
        )


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: int = Query(..., gt=0, description="Session ID"),
    limit: int = Query(50, gt=0, le=100, description="Max messages to return"),
    current_user: Optional[User] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
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
        )


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
async def clear_chat_history(
    session_id: int = Query(..., gt=0, description="Session ID"),
    current_user: Optional[User] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Clear chat history for a session."""
    try:
        await chat_service.clear_chat_history(session_id=session_id)
    except Exception as e:
        logger.error("Failed to clear chat history: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear history",
        )
