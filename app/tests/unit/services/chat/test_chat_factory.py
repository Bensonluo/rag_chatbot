"""Tests for chat service factory"""
import pytest
from unittest.mock import Mock


class TestChatServiceFactory:
    """Test chat service factory"""

    def test_create_chat_service(self):
        """Test creating chat service"""
        # Arrange
        from app.services.chat.factory import ChatServiceFactory

        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()
        mock_retrieval = Mock()

        # Act
        service = ChatServiceFactory.create(
            llm_service=mock_llm,
            memory_strategy=mock_memory,
            intent_detector=mock_intent,
            retrieval_pipeline=mock_retrieval,
        )

        # Assert
        assert service.llm_service == mock_llm
        assert service.memory_strategy == mock_memory
        assert service.intent_detector == mock_intent
        assert service.retrieval_pipeline == mock_retrieval

    def test_create_chat_service_without_retrieval(self):
        """Test creating chat service without retrieval"""
        # Arrange
        from app.services.chat.factory import ChatServiceFactory

        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()

        # Act
        service = ChatServiceFactory.create(
            llm_service=mock_llm,
            memory_strategy=mock_memory,
            intent_detector=mock_intent,
        )

        # Assert
        assert service.llm_service == mock_llm
        assert service.retrieval_pipeline is None

    def test_create_chat_service_with_defaults(self):
        """Test creating chat service with default configuration"""
        # Arrange
        from app.services.chat.factory import ChatServiceFactory

        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()

        # Act
        service = ChatServiceFactory.create_with_defaults(
            llm_service=mock_llm,
            memory_type="sliding_window",
            intent_type="hybrid",
        )

        # Assert
        assert service is not None
        assert service.llm_service == mock_llm

    def test_create_with_memory_factory(self):
        """Test creating chat service using memory factory"""
        # Arrange
        from app.services.chat.factory import ChatServiceFactory
        from app.services.memory import MemoryFactory

        mock_llm = Mock()
        mock_repo = Mock()
        mock_intent = Mock()

        # Create memory strategy
        memory = MemoryFactory.create(
            memory_type="sliding_window",
            message_repo=mock_repo,
            window_size=10,
        )

        # Act
        service = ChatServiceFactory.create(
            llm_service=mock_llm,
            memory_strategy=memory,
            intent_detector=mock_intent,
        )

        # Assert
        assert service.memory_strategy == memory

    def test_create_with_intent_factory(self):
        """Test creating chat service using intent factory"""
        # Arrange
        from app.services.chat.factory import ChatServiceFactory
        from app.services.intent import IntentFactory

        mock_llm = Mock()
        mock_memory = Mock()
        mock_llm_for_intent = Mock()

        # Create intent detector
        intent = IntentFactory.create(
            detector_type="hybrid",
            llm_service=mock_llm_for_intent,
        )

        # Act
        service = ChatServiceFactory.create(
            llm_service=mock_llm,
            memory_strategy=mock_memory,
            intent_detector=intent,
        )

        # Assert
        assert service.intent_detector == intent

    def test_factory_validation(self):
        """Test factory parameter validation"""
        # Arrange
        from app.services.chat.factory import ChatServiceFactory
        from app.core.exceptions import ValidationError

        # Act & Assert - Missing required parameters
        with pytest.raises(ValidationError):
            ChatServiceFactory.create(
                llm_service=None,  # Required
                memory_strategy=Mock(),
                intent_detector=Mock(),
            )
