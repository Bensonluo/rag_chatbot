"""
Chat API endpoints.

Provides REST API for chat interactions including message processing,
streaming responses, and history management.
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from fastapi.responses import StreamingResponse
from typing import Optional, List
from pydantic import BaseModel, Field

from app.api.deps import get_current_user, get_db, get_user_repository, get_session_repository
from app.api.rate_limit import check_rate_limit
from app.models.database.user import User
from app.services.chat.chat_service import ChatService
from app.services.chat.factory import ChatServiceFactory
from app.services.llm import LLMFactory
from app.services.embeddings import EmbeddingFactory
from app.services.retrieval import RetrievalFactory
from sqlalchemy.ext.asyncio import AsyncSession


router = APIRouter(prefix="/chat", tags=["chat"])


@router.on_event("startup")
async def startup_event():
    """Initialize chat service on startup."""
    from app.api.database import get_db as _get_db

    async for db in _get_db():
        try:
            await initialize_chat_service(db)
        except Exception as e:
            import logging
            logging.error(f"Failed to initialize chat service: {e}")
        break  # Only need one db connection

# Global chat service instance (initialized on startup)
_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """
    Get chat service instance (singleton).

    Returns:
        ChatService: Configured chat service

    Raises:
        HTTPException: If service not initialized
    """
    global _chat_service
    if _chat_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service not initialized. Please ensure all dependencies are configured.",
        )
    return _chat_service


async def initialize_chat_service(db: AsyncSession):
    """
    Initialize the RAG chat service with all dependencies.

    Call this during application startup.
    """
    global _chat_service

    # Create LLM service
    llm_service = LLMFactory.create_from_settings()

    # Create repositories
    from app.repositories.message_repository import MessageRepository
    from app.repositories.session_repository import SessionRepository

    message_repo = MessageRepository(db)
    session_repo = SessionRepository(db)

    # Create retrieval pipeline (optional - will work without documents)
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
        # Log but don't fail - chat will work without retrieval
        import logging
        logging.warning(f"Failed to initialize retrieval pipeline: {e}")
        retrieval_pipeline = None

    # Create chat service with all dependencies
    _chat_service = ChatServiceFactory.create_with_defaults(
        llm_service=llm_service,
        message_repo=message_repo,
        session_repo=session_repo,
        memory_type="optimized",
        intent_type="hybrid",
        retrieval_pipeline=retrieval_pipeline,
    )


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


# Dependency
def get_chat_service() -> ChatService:
    """
    Get chat service instance (demo mode).

    Returns demo chat service for open API testing without authentication.
    """
    return _chat_service


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    request: ChatRequest,
    http_req: Request,
    current_user: Optional[User] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
    """
    Process a chat message and generate response.

    Rate limited: 10 requests per minute per IP.

    Args:
        request: Chat message request
        http_req: HTTP request (for rate limiting)
        current_user: Optional authenticated user
        chat_service: Chat service instance

    Returns:
        ChatResponse: Generated response

    Raises:
        HTTPException: If processing fails or rate limit exceeded
    """
    # Check rate limit
    check_rate_limit(http_req)
    """
    Process a chat message and generate response.

    Args:
        request: Chat message request
        current_user: Optional authenticated user
        chat_service: Chat service instance

    Returns:
        ChatResponse: Generated response

    Raises:
        HTTPException: If processing fails
    """
    try:
        # Use authenticated user ID if available, otherwise use request user_id
        user_id = current_user.id if current_user else request.user_id

        # Process message
        response = await chat_service.process_message(
            session_id=request.session_id,
            message=request.message,
            user_id=user_id or 0,  # Default to 0 if no user
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}"
        )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    http_req: Request,
    current_user: Optional[User] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
    """
    Process a chat message with streaming response.

    Rate limited: 10 requests per minute per IP.

    Args:
        request: Chat message request
        http_req: HTTP request (for rate limiting)
        current_user: Optional authenticated user
        chat_service: Chat service instance

    Returns:
        StreamingResponse: Server-sent events stream

    Raises:
        HTTPException: If processing fails or rate limit exceeded
    """
    # Check rate limit
    check_rate_limit(http_req)
    """
    Process a chat message with streaming response.

    Args:
        request: Chat message request
        current_user: Optional authenticated user
        chat_service: Chat service instance

    Returns:
        StreamingResponse: Server-sent events stream

    Raises:
        HTTPException: If processing fails
    """
    try:
        user_id = current_user.id if current_user else request.user_id

        async def generate():
            """Generate streaming response."""
            async for chunk in chat_service.process_message_stream(
                session_id=request.session_id,
                message=request.message,
                user_id=user_id or 0,
            ):
                # Send as server-sent event
                yield f"data: {chunk}\n\n"

            # Send completion signal
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}"
        )


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: int = Query(..., gt=0, description="Session ID"),
    limit: int = Query(50, gt=0, le=100, description="Max messages to return"),
    current_user: Optional[User] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
    """
    Get chat history for a session.

    Args:
        session_id: Session ID
        limit: Max messages to return
        current_user: Optional authenticated user
        chat_service: Chat service instance

    Returns:
        ChatHistoryResponse: Chat history

    Raises:
        HTTPException: If retrieval fails
    """
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get history: {str(e)}"
        )


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
async def clear_chat_history(
    session_id: int = Query(..., gt=0, description="Session ID"),
    current_user: Optional[User] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
    """
    Clear chat history for a session.

    Args:
        session_id: Session ID
        current_user: Optional authenticated user
        chat_service: Chat service instance

    Raises:
        HTTPException: If deletion fails
    """
    try:
        await chat_service.clear_chat_history(session_id=session_id)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear history: {str(e)}"
        )
