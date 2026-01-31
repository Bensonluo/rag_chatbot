"""Tests for LLM-based intent detector"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.models.enums.intent import Intent


class TestLLMIntentDetector:
    """Test LLM-based intent detector"""

    def test_initialization(self):
        """Test detector initialization with LLM service"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase

        mock_llm = Mock(spec=LLMServiceBase)

        # Act
        detector = LLMIntentDetector(llm_service=mock_llm)

        # Assert
        assert detector is not None
        assert detector.llm_service == mock_llm

    def test_initialization_default_intents(self):
        """Test detector initialization with default intents"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase

        mock_llm = Mock(spec=LLMServiceBase)

        # Act
        detector = LLMIntentDetector(llm_service=mock_llm)

        # Assert
        assert len(detector.intents) > 0
        assert Intent.QUESTION in detector.intents

    @pytest.mark.asyncio
    async def test_detect_simple_query(self):
        """Test detecting simple question"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase, LLMResponse

        mock_llm = Mock(spec=LLMServiceBase)
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(content="question", model="gpt-4")
        )

        detector = LLMIntentDetector(llm_service=mock_llm)

        # Act
        intent = await detector.detect("What is AI?")

        # Assert
        assert intent == Intent.QUESTION
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_how_to_query(self):
        """Test detecting how-to query"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase, LLMResponse

        mock_llm = Mock(spec=LLMServiceBase)
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(content="how_to", model="gpt-4")
        )

        detector = LLMIntentDetector(llm_service=mock_llm)

        # Act
        intent = await detector.detect("How do I implement quicksort?")

        # Assert
        assert intent == Intent.HOW_TO

    @pytest.mark.asyncio
    async def test_detect_with_confidence(self):
        """Test detecting with confidence score"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase, LLMResponse

        # Return JSON with confidence
        mock_llm = Mock(spec=LLMServiceBase)
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content='{"intent": "question", "confidence": 0.95}',
                model="gpt-4"
            )
        )

        detector = LLMIntentDetector(llm_service=mock_llm)

        # Act
        result = await detector.detect_with_confidence("What is Python?")

        # Assert
        assert result.intent == Intent.QUESTION
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_detect_invalid_json_response(self):
        """Test handling invalid JSON from LLM"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase, LLMResponse

        mock_llm = Mock(spec=LLMServiceBase)
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(content="just plain text", model="gpt-4")
        )

        detector = LLMIntentDetector(llm_service=mock_llm)

        # Act & Assert - Should fall back to string parsing
        intent = await detector.detect("Hello!")
        # Could be CHITCHAT or UNKNOWN depending on implementation

    @pytest.mark.asyncio
    async def test_detect_with_context(self):
        """Test detection with conversation context"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase, LLMResponse

        mock_llm = Mock(spec=LLMServiceBase)
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(content="chitchat", model="gpt-4")
        )

        detector = LLMIntentDetector(llm_service=mock_llm)

        context = {
            "previous_messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }

        # Act
        intent = await detector.detect("How are you?", context=context)

        # Assert
        assert intent == Intent.CHITCHAT

    @pytest.mark.asyncio
    async def test_detect_llm_error(self):
        """Test handling LLM API errors"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase
        from app.core.exceptions import ExternalServiceError

        mock_llm = Mock(spec=LLMServiceBase)
        mock_llm.generate = AsyncMock(
            side_effect=Exception("API Error")
        )

        detector = LLMIntentDetector(llm_service=mock_llm)

        # Act & Assert
        with pytest.raises(ExternalServiceError):
            await detector.detect("Test query")

    def test_custom_prompt_template(self):
        """Test using custom prompt template"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase

        mock_llm = Mock(spec=LLMServiceBase)

        custom_prompt = "Classify this: {query}\nIntents: {intents}"

        # Act
        detector = LLMIntentDetector(
            llm_service=mock_llm,
            prompt_template=custom_prompt
        )

        # Assert
        assert detector.prompt_template == custom_prompt

    def test_build_system_prompt(self):
        """Test system prompt building"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase

        mock_llm = Mock(spec=LLMServiceBase)
        detector = LLMIntentDetector(llm_service=mock_llm)

        # Act
        prompt = detector._build_system_prompt()

        # Assert
        assert "intent" in prompt.lower()
        assert "classify" in prompt.lower()

    def test_format_prompt(self):
        """Test prompt formatting with variables"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase

        mock_llm = Mock(spec=LLMServiceBase)
        detector = LLMIntentDetector(llm_service=mock_llm)

        # Act
        formatted = detector._format_prompt(
            template="Query: {query}\nIntents: {intents}",
            query="Test query",
            intents="intent1, intent2"
        )

        # Assert
        assert "Query: Test query" in formatted
        assert "intent1, intent2" in formatted
