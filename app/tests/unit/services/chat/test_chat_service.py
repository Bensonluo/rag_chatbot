"""Tests for chat service"""
import pytest
from unittest.mock import Mock, AsyncMock, patch


class TestChatService:
    """Test chat orchestration service"""

    def test_chat_service_initialization(self):
        """Test chat service initialization"""
        # Arrange
        from app.services.chat.chat_service import ChatService

        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()
        mock_retrieval = Mock()

        # Act
        service = ChatService(
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

    @pytest.mark.asyncio
    async def test_process_message_simple(self):
        """Test processing a simple message without retrieval"""
        # Arrange
        from app.services.chat.chat_service import ChatService
        from app.services.llm.base import LLMResponse

        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()
        mock_retrieval = Mock()

        # Mock intent detection
        mock_intent.detect_with_confidence = AsyncMock(
            return_value=Mock(intent="question", confidence=0.9)
        )

        # Mock memory to return empty context
        mock_memory.get_context = AsyncMock(return_value=[])

        # Mock LLM response
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="Hello! How can I help you?",
                model="gpt-4",
                usage={"total_tokens": 20}
            )
        )

        # Mock memory add_message
        mock_memory.add_message = AsyncMock()

        service = ChatService(
            llm_service=mock_llm,
            memory_strategy=mock_memory,
            intent_detector=mock_intent,
            retrieval_pipeline=mock_retrieval,
        )

        # Act
        response = await service.process_message(
            session_id=1,
            message="Hello",
            user_id=1,
        )

        # Assert
        assert response.content == "Hello! How can I help you?"
        assert response.session_id == 1
        assert response.intent == "question"
        mock_intent.detect_with_confidence.assert_called_once()
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_with_retrieval(self):
        """Test processing message with retrieval for RAG"""
        # Arrange
        from app.services.chat.chat_service import ChatService
        from app.services.llm.base import LLMResponse
        from app.services.retrieval.vector_base import SearchResult, VectorSearchRequest

        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()
        mock_retrieval = Mock()

        # Mock intent as question (needs retrieval)
        mock_intent.detect_with_confidence = AsyncMock(
            return_value=Mock(intent="question", confidence=0.9)
        )

        # Mock retrieval results
        mock_retrieval["hybrid_search"].search = AsyncMock(
            return_value=[
                SearchResult(
                    document_id="doc1",
                    content="Python is a programming language",
                    score=0.95
                )
            ]
        )

        # Mock memory
        mock_memory.get_context = AsyncMock(return_value=[])
        mock_memory.add_message = AsyncMock()

        # Mock LLM
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="Python is indeed a programming language.",
                model="gpt-4"
            )
        )

        service = ChatService(
            llm_service=mock_llm,
            memory_strategy=mock_memory,
            intent_detector=mock_intent,
            retrieval_pipeline=mock_retrieval,
        )

        # Act
        response = await service.process_message(
            session_id=1,
            message="What is Python?",
            user_id=1,
        )

        # Assert
        assert "Python" in response.content
        assert response.sources is not None
        assert len(response.sources) > 0
        mock_retrieval["hybrid_search"].search.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_with_memory(self):
        """Test processing message with conversation memory"""
        # Arrange
        from app.services.chat.chat_service import ChatService
        from app.services.llm.base import LLMResponse, LLMMessage
        from app.services.memory.base import MessageContent

        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()
        mock_retrieval = Mock()

        # Mock intent
        mock_intent.detect_with_confidence = AsyncMock(
            return_value=Mock(intent="question", confidence=0.9)
        )

        # Mock memory with conversation context
        mock_memory.get_context = AsyncMock(
            return_value=[
                MessageContent(role="user", content="My name is Alice"),
                MessageContent(role="assistant", content="Hello Alice!"),
            ]
        )
        mock_memory.add_message = AsyncMock()

        # Mock LLM
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="Nice to meet you, Alice!",
                model="gpt-4"
            )
        )

        service = ChatService(
            llm_service=mock_llm,
            memory_strategy=mock_memory,
            intent_detector=mock_intent,
            retrieval_pipeline=mock_retrieval,
        )

        # Act
        response = await service.process_message(
            session_id=1,
            message="What's my name?",
            user_id=1,
        )

        # Assert
        assert "Alice" in response.content
        # Verify LLM was called with memory context
        call_args = mock_llm.generate.call_args
        messages = call_args[0][0]  # First positional argument
        assert len(messages) > 1  # Should have context

    @pytest.mark.asyncio
    async def test_process_message_with_streaming(self):
        """Test processing message with streaming response"""
        # Arrange
        from app.services.chat.chat_service import ChatService

        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()
        mock_retrieval = Mock()

        # Mock intent
        mock_intent.detect_with_confidence = AsyncMock(
            return_value=Mock(intent="question", confidence=0.9)
        )

        # Mock memory
        mock_memory.get_context = AsyncMock(return_value=[])
        mock_memory.add_message = AsyncMock()

        # Mock streaming LLM
        async def mock_stream():
            yield "Hello"
            yield " there"
            yield "!"

        mock_llm.generate_stream = AsyncMock(return_value=mock_stream())

        service = ChatService(
            llm_service=mock_llm,
            memory_strategy=mock_memory,
            intent_detector=mock_intent,
            retrieval_pipeline=mock_retrieval,
        )

        # Act
        stream = await service.process_message_stream(
            session_id=1,
            message="Hello",
            user_id=1,
        )

        # Collect stream chunks
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        # Assert
        assert chunks == ["Hello", " there", "!"]
        mock_llm.generate_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_chat_history(self):
        """Test getting chat history"""
        # Arrange
        from app.services.chat.chat_service import ChatService
        from app.services.memory.base import MessageContent

        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()
        mock_retrieval = Mock()

        # Mock memory
        mock_memory.get_context = AsyncMock(
            return_value=[
                MessageContent(role="user", content="Hello"),
                MessageContent(role="assistant", content="Hi there!"),
                MessageContent(role="user", content="How are you?"),
            ]
        )

        service = ChatService(
            llm_service=mock_llm,
            memory_strategy=mock_memory,
            intent_detector=mock_intent,
            retrieval_pipeline=mock_retrieval,
        )

        # Act
        history = await service.get_chat_history(session_id=1)

        # Assert
        assert len(history) == 3
        assert history[0].role == "user"
        assert history[0].content == "Hello"
        assert history[1].role == "assistant"
        assert history[1].content == "Hi there!"

    @pytest.mark.asyncio
    async def test_clear_chat_history(self):
        """Test clearing chat history"""
        # Arrange
        from app.services.chat.chat_service import ChatService

        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()
        mock_retrieval = Mock()

        mock_memory.clear_session = AsyncMock()

        service = ChatService(
            llm_service=mock_llm,
            memory_strategy=mock_memory,
            intent_detector=mock_intent,
            retrieval_pipeline=mock_retrieval,
        )

        # Act
        await service.clear_chat_history(session_id=1)

        # Assert
        mock_memory.clear_session.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_process_message_error_handling(self):
        """Test error handling in message processing"""
        # Arrange
        from app.services.chat.chat_service import ChatService
        from app.core.exceptions import BaseServiceError

        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()
        mock_retrieval = Mock()

        # Mock intent to raise error
        mock_intent.detect_with_confidence = AsyncMock(
            side_effect=Exception("Intent detection failed")
        )

        service = ChatService(
            llm_service=mock_llm,
            memory_strategy=mock_memory,
            intent_detector=mock_intent,
            retrieval_pipeline=mock_retrieval,
        )

        # Act & Assert
        with pytest.raises(BaseServiceError):
            await service.process_message(
                session_id=1,
                message="Hello",
                user_id=1,
            )

    @pytest.mark.asyncio
    async def test_build_context_with_retrieval(self):
        """Test building context with retrieved documents"""
        # Arrange
        from app.services.chat.chat_service import ChatService
        from app.services.retrieval.vector_base import SearchResult

        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()
        mock_retrieval = Mock()

        # Mock retrieval
        mock_retrieval["hybrid_search"].search = AsyncMock(
            return_value=[
                SearchResult(
                    document_id="doc1",
                    content="Python documentation",
                    score=0.95
                ),
                SearchResult(
                    document_id="doc2",
                    content="Python tutorial",
                    score=0.87
                ),
            ]
        )

        service = ChatService(
            llm_service=mock_llm,
            memory_strategy=mock_memory,
            intent_detector=mock_intent,
            retrieval_pipeline=mock_retrieval,
        )

        # Act
        context = await service._build_context_with_retrieval(
            query="What is Python?",
            top_k=2
        )

        # Assert
        assert len(context) >= 2
        assert "Python documentation" in context
        assert "Python tutorial" in context

    @pytest.mark.asyncio
    async def test_should_use_retrieval(self):
        """Test decision logic for when to use retrieval"""
        # Arrange
        from app.services.chat.chat_service import ChatService
        from app.services.intent.base import Intent

        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()
        mock_retrieval = Mock()

        service = ChatService(
            llm_service=mock_llm,
            memory_strategy=mock_memory,
            intent_detector=mock_intent,
            retrieval_pipeline=mock_retrieval,
        )

        # Act & Assert - Question intent should use retrieval
        result1 = service._should_use_retrieval(Intent.QUESTION)
        assert result1 is True

        # Greeting intent should not use retrieval
        result2 = service._should_use_retrieval(Intent.GREETING)
        assert result2 is False

        # How_to intent should use retrieval
        result3 = service._should_use_retrieval(Intent.HOW_TO)
        assert result3 is True


class TestChatResponse:
    """Test ChatResponse dataclass"""

    def test_chat_response_creation(self):
        """Test creating chat response"""
        # Arrange
        from app.services.chat.chat_service import ChatResponse

        # Act
        response = ChatResponse(
            content="Hello!",
            session_id=1,
            intent="question",
            sources=["doc1", "doc2"],
            metadata={"tokens": 50}
        )

        # Assert
        assert response.content == "Hello!"
        assert response.session_id == 1
        assert response.intent == "question"
        assert len(response.sources) == 2
        assert response.metadata["tokens"] == 50

    def test_chat_response_without_sources(self):
        """Test chat response without sources"""
        # Arrange
        from app.services.chat.chat_service import ChatResponse

        # Act
        response = ChatResponse(
            content="Hi!",
            session_id=1,
            intent="greeting"
        )

        # Assert
        assert response.content == "Hi"
        assert response.sources is None
