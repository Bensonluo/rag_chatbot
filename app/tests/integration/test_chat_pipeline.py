"""Integration tests for chat pipeline"""
import pytest
from unittest.mock import Mock, AsyncMock


class TestChatPipelineIntegration:
    """Test complete chat pipeline integration"""

    @pytest.mark.asyncio
    async def test_full_chat_pipeline_with_retrieval(self):
        """Test complete pipeline: intent -> retrieval -> memory -> generation"""
        # Arrange
        from app.services.chat.chat_service import ChatService
        from app.services.llm.base import LLMResponse, LLMMessage
        from app.services.memory.base import MessageContent
        from app.services.intent.base import Intent, IntentResult
        from app.services.retrieval.vector_base import SearchResult, VectorSearchRequest

        # Mock LLM
        mock_llm = Mock()
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="Python is a high-level programming language.",
                model="gpt-4",
                usage={"total_tokens": 30}
            )
        )

        # Mock memory
        mock_memory = Mock()
        mock_memory.get_context = AsyncMock(return_value=[])
        mock_memory.add_message = AsyncMock()

        # Mock intent detector
        mock_intent = Mock()
        mock_intent.detect_with_confidence = AsyncMock(
            return_value=IntentResult(
                intent=Intent.QUESTION,
                confidence=0.95,
                metadata={}
            )
        )

        # Mock retrieval pipeline
        mock_retrieval = {
            "hybrid_search": Mock(
                search=AsyncMock(
                    return_value=[
                        SearchResult(
                            document_id="doc1",
                            content="Python is a programming language created by Guido van Rossum.",
                            score=0.95
                        )
                    ]
                )
            ),
            "reranker": Mock(
                rerank=AsyncMock(
                    return_value=[
                        SearchResult(
                            document_id="doc1",
                            content="Python is a programming language created by Guido van Rossum.",
                            score=0.95
                        )
                    ]
                )
            ),
            "metadata_service": Mock(
                enrich_search_results=AsyncMock(
                    return_value=[
                        SearchResult(
                            document_id="doc1",
                            content="Python is a programming language created by Guido van Rossum.",
                            score=0.95,
                            metadata={"title": "Python Documentation"}
                        )
                    ]
                )
            ),
        }

        # Create service
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

        # Assert - Complete pipeline executed
        assert response.content is not None
        assert "Python" in response.content
        assert response.intent == "question"
        assert response.sources is not None
        assert len(response.sources) == 1
        assert response.sources[0] == "doc1"

        # Verify all components were called
        mock_intent.detect_with_confidence.assert_called_once_with("What is Python?")
        mock_memory.get_context.assert_called_once()
        mock_retrieval["hybrid_search"].search.assert_called_once()
        mock_retrieval["reranker"].rerank.assert_called_once()
        mock_retrieval["metadata_service"].enrich_search_results.assert_called_once()
        mock_llm.generate.assert_called_once()
        assert mock_memory.add_message.call_count == 2  # User + Assistant

    @pytest.mark.asyncio
    async def test_chat_pipeline_with_conversation_memory(self):
        """Test pipeline with conversation context from memory"""
        # Arrange
        from app.services.chat.chat_service import ChatService
        from app.services.llm.base import LLMResponse
        from app.services.memory.base import MessageContent
        from app.services.intent.base import Intent, IntentResult

        mock_llm = Mock()
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="You said your name is Alice.",
                model="gpt-4"
            )
        )

        mock_memory = Mock()
        # Simulate conversation history
        mock_memory.get_context = AsyncMock(
            return_value=[
                MessageContent(role="user", content="My name is Alice"),
                MessageContent(role="assistant", content="Nice to meet you, Alice!"),
            ]
        )
        mock_memory.add_message = AsyncMock()

        mock_intent = Mock()
        mock_intent.detect_with_confidence = AsyncMock(
            return_value=IntentResult(intent=Intent.QUESTION, confidence=0.9, metadata={})
        )

        service = ChatService(
            llm_service=mock_llm,
            memory_strategy=mock_memory,
            intent_detector=mock_intent,
            retrieval_pipeline=None,
        )

        # Act
        response = await service.process_message(
            session_id=1,
            message="What's my name?",
            user_id=1,
        )

        # Assert
        assert "Alice" in response.content

        # Verify LLM was called with context
        call_args = mock_llm.generate.call_args
        messages = call_args[0][0]  # First positional arg
        assert len(messages) >= 3  # System + 2 context messages + current

    @pytest.mark.asyncio
    async def test_chat_pipeline_streaming(self):
        """Test complete pipeline with streaming response"""
        # Arrange
        from app.services.chat.chat_service import ChatService
        from app.services.intent.base import Intent, IntentResult

        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()

        # Mock streaming response
        async def mock_stream():
            yield "Hello"
            yield " Alice"
            yield "!"

        mock_llm.generate_stream = AsyncMock(return_value=mock_stream())
        mock_memory.get_context = AsyncMock(return_value=[])
        mock_memory.add_message = AsyncMock()
        mock_intent.detect_with_confidence = AsyncMock(
            return_value=IntentResult(intent=Intent.GREETING, confidence=0.95, metadata={})
        )

        service = ChatService(
            llm_service=mock_llm,
            memory_strategy=mock_memory,
            intent_detector=mock_intent,
            retrieval_pipeline=None,
        )

        # Act
        chunks = []
        async for chunk in service.process_message_stream(
            session_id=1,
            message="Hello",
            user_id=1,
        ):
            chunks.append(chunk)

        # Assert
        assert chunks == ["Hello", " Alice", "!"]

    @pytest.mark.asyncio
    async def test_chat_pipeline_error_recovery(self):
        """Test pipeline handles errors gracefully"""
        # Arrange
        from app.services.chat.chat_service import ChatService
        from app.core.exceptions import BaseServiceError

        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()
        mock_retrieval = {
            "hybrid_search": Mock(search=AsyncMock(side_effect=Exception("Search failed")))
        }

        mock_llm.generate = AsyncMock(
            return_value=Mock(content="Fallback response", model="gpt-4")
        )
        mock_memory.get_context = AsyncMock(return_value=[])
        mock_memory.add_message = AsyncMock()
        mock_intent.detect_with_confidence = AsyncMock(
            return_value=Mock(intent=Intent.QUESTION, confidence=0.9)
        )

        service = ChatService(
            llm_service=mock_llm,
            memory_strategy=mock_memory,
            intent_detector=mock_intent,
            retrieval_pipeline=mock_retrieval,
        )

        # Act - Should not fail even if retrieval fails
        response = await service.process_message(
            session_id=1,
            message="Test message",
            user_id=1,
        )

        # Assert - Should have response (without sources)
        assert response.content is not None
        assert response.sources is None  # No sources due to retrieval failure

    @pytest.mark.asyncio
    async def test_chat_pipeline_with_factory(self):
        """Test creating complete pipeline with factory"""
        # Arrange
        from app.services.chat.factory import ChatServiceFactory
        from app.services.memory import MemoryFactory
        from app.services.intent import IntentFactory

        mock_llm = Mock()
        mock_message_repo = Mock()
        mock_session_repo = Mock()

        # Mock methods
        mock_llm.generate = AsyncMock(
            return_value=Mock(content="Response", model="gpt-4", usage={})
        )
        mock_message_repo.get_recent_messages = AsyncMock(return_value=[])
        mock_message_repo.create = AsyncMock()
        mock_message_repo.count_messages = AsyncMock(return_value=0)

        # Create service with factory
        service = ChatServiceFactory.create_with_defaults(
            llm_service=mock_llm,
            message_repo=mock_message_repo,
            session_repo=mock_session_repo,
            memory_type="sliding_window",
            intent_type="rule_based",
        )

        # Act
        response = await service.process_message(
            session_id=1,
            message="Hello",
            user_id=1,
        )

        # Assert
        assert response.content == "Response"
        assert service is not None


class TestChatEndToEndScenarios:
    """Test end-to-end chat scenarios"""

    @pytest.mark.asyncio
    async def test_question_answering_scenario(self):
        """Test QA scenario with retrieval"""
        # Arrange
        from app.services.chat.chat_service import ChatService
        from app.services.llm.base import LLMResponse
        from app.services.intent.base import Intent, IntentResult
        from app.services.retrieval.vector_base import SearchResult

        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()
        mock_retrieval = {
            "hybrid_search": Mock(
                search=AsyncMock(
                    return_value=[
                        SearchResult(
                            document_id="doc1",
                            content="FastAPI is a modern web framework for building APIs with Python.",
                            score=0.95
                        )
                    ]
                )
            ),
            "reranker": Mock(rerank=AsyncMock(lambda x, r: x)),
            "metadata_service": Mock(enrich_search_results=AsyncMock(lambda x: x)),
        }

        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="FastAPI is a modern web framework for building APIs with Python.",
                model="gpt-4"
            )
        )
        mock_memory.get_context = AsyncMock(return_value=[])
        mock_memory.add_message = AsyncMock()
        mock_intent.detect_with_confidence = AsyncMock(
            return_value=IntentResult(intent=Intent.QUESTION, confidence=0.95, metadata={})
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
            message="What is FastAPI?",
            user_id=1,
        )

        # Assert
        assert "FastAPI" in response.content
        assert response.sources is not None
        assert len(response.sources) == 1

    @pytest.mark.asyncio
    async def test_greeting_scenario(self):
        """Test greeting scenario without retrieval"""
        # Arrange
        from app.services.chat.chat_service import ChatService
        from app.services.llm.base import LLMResponse
        from app.services.intent.base import Intent, IntentResult

        mock_llm = Mock()
        mock_memory = Mock()
        mock_intent = Mock()

        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="Hello! How can I help you today?",
                model="gpt-4"
            )
        )
        mock_memory.get_context = AsyncMock(return_value=[])
        mock_memory.add_message = AsyncMock()
        mock_intent.detect_with_confidence = AsyncMock(
            return_value=IntentResult(intent=Intent.GREETING, confidence=0.95, metadata={})
        )

        service = ChatService(
            llm_service=mock_llm,
            memory_strategy=mock_memory,
            intent_detector=mock_intent,
            retrieval_pipeline=None,
        )

        # Act
        response = await service.process_message(
            session_id=1,
            message="Hello",
            user_id=1,
        )

        # Assert
        assert "Hello" in response.content or "help" in response.content
        assert response.sources is None  # No retrieval for greetings
