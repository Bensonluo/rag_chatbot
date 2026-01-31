"""
Demo chat service for open API testing.

Provides simple responses without requiring full RAG pipeline setup.
"""
from dataclasses import dataclass, field
from typing import Optional, List, AsyncGenerator
import random


@dataclass
class DemoChatResponse:
    """Demo chat response data."""
    content: str
    session_id: int
    intent: str
    sources: Optional[List[str]] = None
    metadata: Optional[dict] = None


@dataclass
class DemoChatMessage:
    """Demo chat message data."""
    role: str
    content: str
    timestamp: Optional[str] = None


class DemoChatService:
    """
    Demo chat service for open API testing.

    Returns simple responses without requiring LLM, embeddings, or vector DB.
    """

    def __init__(self):
        """Initialize demo chat service."""
        self.demo_responses = {
            "hello": "Hello! I'm a demo RAG chatbot. How can I help you today?",
            "hi": "Hi there! I'm here to help. What would you like to know?",
            "help": "I can help you with various questions. This is a demo mode, so I'll provide simple responses.",
            "what": "That's a great question! In production mode, I would search our knowledge base for relevant information.",
            "how": "Let me explain how this works. In production, I use intent detection, memory, and retrieval to provide accurate answers.",
            "default": "Thank you for your message! This is a demo response. In production, I would process your query using our RAG pipeline.",
        }

    async def process_message(
        self,
        session_id: int,
        message: str,
        user_id: int,
        max_tokens: Optional[int] = None,
    ) -> DemoChatResponse:
        """
        Process a user message and generate demo response.

        Args:
            session_id: Session identifier
            message: User message
            user_id: User identifier
            max_tokens: Optional max tokens for response

        Returns:
            DemoChatResponse: Generated response
        """
        # Simple intent detection based on keywords
        message_lower = message.lower()

        if any(word in message_lower for word in ["hello", "hey"]):
            intent = "greeting"
            response = self.demo_responses["hello"]
        elif any(word in message_lower for word in ["help", "assist"]):
            intent = "question"
            response = self.demo_responses["help"]
        elif "what" in message_lower:
            intent = "question"
            response = self.demo_responses["what"]
        elif "how" in message_lower:
            intent = "how_to"
            response = self.demo_responses["how"]
        else:
            intent = "chitchat"
            response = self.demo_responses["default"]

        return DemoChatResponse(
            content=response,
            session_id=session_id,
            intent=intent,
            sources=None,
            metadata={
                "demo": True,
                "tokens_used": len(response.split()),
            },
        )

    async def process_message_stream(
        self,
        session_id: int,
        message: str,
        user_id: int,
    ) -> AsyncGenerator[str, None]:
        """
        Process message with streaming demo response.

        Args:
            session_id: Session identifier
            message: User message
            user_id: User identifier

        Yields:
            str: Response chunks
        """
        response = await self.process_message(session_id, message, user_id)
        words = response.content.split()

        for word in words:
            yield word + " "

    async def get_chat_history(
        self,
        session_id: int,
        limit: int = 50,
    ) -> List[DemoChatMessage]:
        """
        Get demo chat history.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages

        Returns:
            List[DemoChatMessage]: Empty list (demo mode doesn't persist)
        """
        return []

    async def clear_chat_history(self, session_id: int) -> None:
        """
        Clear demo chat history (no-op in demo mode).

        Args:
            session_id: Session identifier
        """
        pass
