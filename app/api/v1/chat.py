"""
Chat API endpoints.

Provides REST API for chat interactions including message processing,
streaming responses, and history management.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import Optional
from pydantic import BaseModel, Field

from app.services.chat.chat_service import ChatService, ChatResponse as ServiceChatResponse
from app.services.chat.chat_service import ChatMessage as ServiceChatMessage
from app.api.deps import get_current_user
from app.models.database.user import User


router = APIRouter(prefix="/chat", tags=["chat"])


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
    sources: Optional[list[str]] = None
    metadata: Optional[dict] = None


class ChatMessageResponse(BaseModel):
    """Chat message in history."""
    role: str
    content: str
    timestamp: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    """Chat history response."""
    messages: list[ChatMessageResponse]
    session_id: int


# Dependency
def get_chat_service() -> ChatService:
    """
    Get chat service instance.

    In production, this would be properly initialized with all dependencies.
    For now, returns a mock or placeholder.
    """
    # TODO: Properly initialize chat service with dependencies
    # This is a placeholder - should use dependency injection
    from app.core.exceptions import BaseServiceError
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Chat service not configured"
    )


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    request: ChatRequest,
    current_user: Optional[User] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
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
    current_user: Optional[User] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
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
    session_id: int = Field(..., gt=0, description="Session ID"),
    limit: int = Field(50, gt=0, le=100, description="Max messages to return"),
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
    session_id: int = Field(..., gt=0, description="Session ID"),
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
