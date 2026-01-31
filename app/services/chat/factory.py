"""
Factory for creating chat service instances.

Provides simple interface for creating chat service with all required dependencies.
"""
from typing import Optional

from app.services.chat.chat_service import ChatService
from app.services.llm.base import LLMServiceBase
from app.services.memory.base import MemoryStrategy
from app.services.memory import MemoryFactory
from app.services.intent.base import IntentDetector
from app.services.intent import IntentFactory
from app.repositories.message_repository import MessageRepository
from app.repositories.session_repository import SessionRepository
from app.core.exceptions import ValidationError


class ChatServiceFactory:
    """
    Factory for creating chat service instances.

    Provides methods for creating chat service with proper dependency injection.
    """

    @staticmethod
    def create(
        llm_service: LLMServiceBase,
        memory_strategy: MemoryStrategy,
        intent_detector: IntentDetector,
        retrieval_pipeline: Optional[dict] = None,
    ) -> ChatService:
        """
        Create a chat service instance.

        Args:
            llm_service: LLM service for generation
            memory_strategy: Memory strategy for context
            intent_detector: Intent detection service
            retrieval_pipeline: Optional retrieval pipeline

        Returns:
            ChatService: Configured chat service

        Raises:
            ValidationError: If required parameters are missing
        """
        # Validate required parameters
        if llm_service is None:
            raise ValidationError("llm_service is required")

        if memory_strategy is None:
            raise ValidationError("memory_strategy is required")

        if intent_detector is None:
            raise ValidationError("intent_detector is required")

        return ChatService(
            llm_service=llm_service,
            memory_strategy=memory_strategy,
            intent_detector=intent_detector,
            retrieval_pipeline=retrieval_pipeline,
        )

    @staticmethod
    def create_with_defaults(
        llm_service: LLMServiceBase,
        message_repo: MessageRepository,
        session_repo: SessionRepository,
        memory_type: str = "optimized",
        intent_type: str = "hybrid",
        retrieval_pipeline: Optional[dict] = None,
        **memory_kwargs,
    ) -> ChatService:
        """
        Create chat service with default memory and intent strategies.

        Args:
            llm_service: LLM service for generation and intent detection
            message_repo: Message repository for memory
            session_repo: Session repository
            memory_type: Type of memory strategy ("sliding_window", "summarization", "hybrid", "optimized")
            intent_type: Type of intent detector ("rule_based", "llm_based", "hybrid")
            retrieval_pipeline: Optional retrieval pipeline
            **memory_kwargs: Additional parameters for memory strategy

        Returns:
            ChatService: Configured chat service
        """
        # Create embedding service if using optimized memory
        from app.services.embeddings import EmbeddingFactory

        embedding_service = None
        if memory_type == "optimized":
            embedding_service = EmbeddingFactory.create_from_settings()

        # Create memory strategy
        memory_strategy = MemoryFactory.create(
            memory_type=memory_type,
            message_repo=message_repo,
            llm_service=llm_service if memory_type in ["summarization", "hybrid"] else None,
            embedding_service=embedding_service,
            **memory_kwargs,
        )

        # Create intent detector
        intent_detector = IntentFactory.create(
            detector_type=intent_type,
            llm_service=llm_service if intent_type in ["llm_based", "hybrid"] else None,
        )

        return ChatService(
            llm_service=llm_service,
            memory_strategy=memory_strategy,
            intent_detector=intent_detector,
            retrieval_pipeline=retrieval_pipeline,
        )
