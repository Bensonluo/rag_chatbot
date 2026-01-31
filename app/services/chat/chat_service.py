"""
Chat orchestration service.

Integrates LLM, memory, intent detection, and retrieval to provide
intelligent chat responses with context awareness.
"""
from dataclasses import dataclass, field
from typing import Optional, List, AsyncGenerator
from enum import Enum

from app.services.llm.base import (
    LLMServiceBase,
    LLMMessage,
    LLMResponse,
)
from app.services.memory.base import MemoryStrategy, MessageContent
from app.services.intent.base import IntentDetector, Intent, IntentResult
from app.services.retrieval.vector_base import VectorSearchRequest, SearchResult
from app.core.exceptions import BaseServiceError


class ChatIntent(str, Enum):
    """Chat intent types aligned with Intent enum."""
    QUESTION = "question"
    GREETING = "greeting"
    HOW_TO = "how_to"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    RECOMMENDATION = "recommendation"
    CLARIFICATION = "clarification"
    FEEDBACK = "feedback"
    COMMAND = "command"
    CHITCHAT = "chitchat"


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
    sources: Optional[List[str]] = None
    metadata: Optional[dict] = None


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
    timestamp: Optional[str] = None


class ChatService:
    """
    Chat orchestration service.

    Coordinates intent detection, retrieval, memory, and LLM generation
    to provide intelligent chat responses.
    """

    # Intents that should trigger retrieval
    RETRIEVAL_INTENTS = {
        Intent.QUESTION,
        Intent.HOW_TO,
        Intent.COMPARISON,
        Intent.DEFINITION,
        Intent.RECOMMENDATION,
    }

    def __init__(
        self,
        llm_service: LLMServiceBase,
        memory_strategy: MemoryStrategy,
        intent_detector: IntentDetector,
        retrieval_pipeline: Optional[dict] = None,
    ) -> None:
        """
        Initialize chat service.

        Args:
            llm_service: LLM service for generation
            memory_strategy: Memory strategy for context
            intent_detector: Intent detection service
            retrieval_pipeline: Optional retrieval pipeline for RAG
        """
        self.llm_service = llm_service
        self.memory_strategy = memory_strategy
        self.intent_detector = intent_detector
        self.retrieval_pipeline = retrieval_pipeline

    async def process_message(
        self,
        session_id: int,
        message: str,
        user_id: int,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """
        Process a user message and generate response.

        Args:
            session_id: Session identifier
            message: User message
            user_id: User identifier
            max_tokens: Optional max tokens for response

        Returns:
            ChatResponse: Generated response

        Raises:
            BaseServiceError: If processing fails
        """
        try:
            # 1. Detect intent
            intent_result = await self.intent_detector.detect_with_confidence(message)
            intent = intent_result.intent

            # 2. Get conversation context from memory
            context = await self.memory_strategy.get_context(
                session_id=session_id,
                max_tokens=max_tokens,
            )

            # 3. Retrieve relevant documents if needed
            retrieved_docs = []
            sources = None

            if self._should_use_retrieval(intent) and self.retrieval_pipeline:
                retrieved_docs = await self._retrieve_documents(
                    query=message,
                    top_k=3,
                )
                sources = [doc.document_id for doc in retrieved_docs] if retrieved_docs else None

            # 4. Build messages for LLM
            messages = await self._build_messages(
                user_message=message,
                context=context,
                retrieved_docs=retrieved_docs,
            )

            # 5. Generate response
            llm_response = await self.llm_service.generate(
                messages=messages,
                max_tokens=max_tokens,
            )

            # 6. Store messages in memory
            await self._store_messages(
                session_id=session_id,
                user_message=message,
                assistant_response=llm_response.content,
            )

            # 7. Build response
            metadata = {
                "confidence": intent_result.confidence,
                "tokens_used": llm_response.usage.get("total_tokens") if llm_response.usage else None,
            }

            return ChatResponse(
                content=llm_response.content,
                session_id=session_id,
                intent=intent.value,
                sources=sources,
                metadata=metadata,
            )

        except Exception as e:
            raise BaseServiceError(
                f"Failed to process message: {str(e)}"
            ) from e

    async def process_message_stream(
        self,
        session_id: int,
        message: str,
        user_id: int,
    ) -> AsyncGenerator[str, None]:
        """
        Process message with streaming response.

        Args:
            session_id: Session identifier
            message: User message
            user_id: User identifier

        Yields:
            str: Response chunks
        """
        # 1. Detect intent
        intent_result = await self.intent_detector.detect_with_confidence(message)

        # 2. Get context (with current query for relevance filtering)
        # Check if memory strategy supports current_query parameter
        import inspect
        sig = inspect.signature(self.memory_strategy.get_context)
        if 'current_query' in sig.parameters:
            # Optimized context builder - pass query for relevance filtering
            context = await self.memory_strategy.get_context(
                session_id=session_id,
                current_query=message
            )
        else:
            # Standard memory builder - no relevance filtering
            context = await self.memory_strategy.get_context(
                session_id=session_id
            )

        # 3. Retrieve if needed
        retrieved_docs = []
        if self._should_use_retrieval(intent_result.intent) and self.retrieval_pipeline:
            retrieved_docs = await self._retrieve_documents(query=message, top_k=3)

        # 4. Build messages
        messages = await self._build_messages(
            user_message=message,
            context=context,
            retrieved_docs=retrieved_docs,
        )

        # 5. Stream response
        full_response = ""
        async for chunk in self.llm_service.generate_stream(messages):
            full_response += chunk
            yield chunk

        # 6. Store in memory
        await self._store_messages(
            session_id=session_id,
            user_message=message,
            assistant_response=full_response,
        )

    async def get_chat_history(
        self,
        session_id: int,
        limit: int = 50,
    ) -> List[ChatMessage]:
        """
        Get chat history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages

        Returns:
            List[ChatMessage]: List of chat messages
        """
        context = await self.memory_strategy.get_context(
            session_id=session_id,
        )

        return [
            ChatMessage(
                role=msg.role,
                content=msg.content,
            )
            for msg in context[:limit]
        ]

    async def clear_chat_history(self, session_id: int) -> None:
        """
        Clear chat history for a session.

        Args:
            session_id: Session identifier
        """
        await self.memory_strategy.clear_session(session_id=session_id)

    def _should_use_retrieval(self, intent: Intent) -> bool:
        """
        Determine if retrieval should be used based on intent.

        Args:
            intent: Detected intent

        Returns:
            bool: True if retrieval should be used
        """
        return intent in self.RETRIEVAL_INTENTS

    async def _retrieve_documents(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[SearchResult]:
        """
        Retrieve relevant documents using retrieval pipeline.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List[SearchResult]: Retrieved documents
        """
        if not self.retrieval_pipeline:
            return []

        try:
            # Use hybrid search
            hybrid_search = self.retrieval_pipeline.get("hybrid_search")
            if not hybrid_search:
                return []

            request = VectorSearchRequest(query=query, top_k=top_k)
            results = await hybrid_search.search(request)

            # Rerank if available
            reranker = self.retrieval_pipeline.get("reranker")
            if reranker:
                results = await reranker.rerank(results, request)

            # Enrich with metadata if available
            metadata_service = self.retrieval_pipeline.get("metadata_service")
            if metadata_service:
                results = await metadata_service.enrich_search_results(results)

            return results

        except Exception as e:
            # Log error but don't fail
            return []

    async def _build_messages(
        self,
        user_message: str,
        context: List[MessageContent],
        retrieved_docs: List[SearchResult],
    ) -> List[LLMMessage]:
        """
        Build messages for LLM generation.

        Args:
            user_message: User message
            context: Conversation context
            retrieved_docs: Retrieved documents

        Returns:
            List[LLMMessage]: Messages for LLM
        """
        messages = []

        # System prompt
        system_prompt = self._build_system_prompt(retrieved_docs)
        messages.append(LLMMessage(role="system", content=system_prompt))

        # Conversation context
        for msg in context:
            messages.append(
                LLMMessage(role=msg.role, content=msg.content)
            )

        # Current user message
        messages.append(LLMMessage(role="user", content=user_message))

        return messages

    def _build_system_prompt(self, retrieved_docs: List[SearchResult]) -> str:
        """
        Build system prompt with retrieved context.

        Args:
            retrieved_docs: Retrieved documents

        Returns:
            str: System prompt
        """
        base_prompt = (
            "You are a helpful, friendly AI assistant. "
            "Provide accurate, concise responses."
        )

        if not retrieved_docs:
            return base_prompt

        # Add retrieved context
        context_parts = ["\n\nRelevant information:"]
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"\n{i}. {doc.content}")

        return base_prompt + "".join(context_parts)

    async def _store_messages(
        self,
        session_id: int,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """
        Store messages in memory.

        Args:
            session_id: Session identifier
            user_message: User message
            assistant_response: Assistant response
        """
        # Store user message
        await self.memory_strategy.add_message(
            session_id=session_id,
            message=MessageContent(role="user", content=user_message),
        )

        # Store assistant response
        await self.memory_strategy.add_message(
            session_id=session_id,
            message=MessageContent(role="assistant", content=assistant_response),
        )

    async def _build_context_with_retrieval(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[str]:
        """
        Build context from retrieved documents.

        Args:
            query: Search query
            top_k: Number of documents

        Returns:
            List[str]: Retrieved document contents
        """
        docs = await self._retrieve_documents(query=query, top_k=top_k)
        return [doc.content for doc in docs]
