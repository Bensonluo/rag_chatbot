"""Tests for Anthropic LLM client"""
import pytest
from unittest.mock import Mock, AsyncMock


class TestAnthropicClient:
    """Test Anthropic LLM client implementation"""

    def test_initialization(self):
        """Test client initialization"""
        # Arrange & Act
        from app.services.llm.anthropic_client import AnthropicClient
        client = AnthropicClient(
            api_key="test-key",
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.7
        )

        # Assert
        assert client.api_key == "test-key"
        assert client.model == "claude-3-opus-20240229"
        assert client.max_tokens == 1000
        assert client.temperature == 0.7

    def test_initialization_defaults(self):
        """Test client initialization with defaults"""
        # Arrange & Act
        from app.services.llm.anthropic_client import AnthropicClient
        client = AnthropicClient(
            api_key="test-key",
            model="claude-3-opus-20240229"
        )

        # Assert
        assert client.max_tokens is None
        assert client.temperature is None

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful completion generation"""
        # Arrange
        from app.services.llm.anthropic_client import AnthropicClient
        from app.services.llm.base import LLMMessage

        # Mock Anthropic API response
        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Hi there!")]
        mock_response.stop_reason = "end_turn"
        mock_response.model = "claude-3-opus-20240229"
        mock_response.usage = Mock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client = Mock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        client = AnthropicClient(
            api_key="test-key", model="claude-3-opus-20240229", client=mock_client
        )
        messages = [LLMMessage(role="user", content="Hello!")]

        # Act
        response = await client.generate(messages=messages)

        # Assert
        assert response.content == "Hi there!"
        assert response.model == "claude-3-opus-20240229"
        assert response.finish_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_generate_with_system_message(self):
        """Test generation with system message"""
        # Arrange
        from app.services.llm.anthropic_client import AnthropicClient
        from app.services.llm.base import LLMMessage

        mock_response = Mock()
        mock_response.content = [Mock(type="text", text="Hi there!")]
        mock_response.stop_reason = "end_turn"
        mock_response.model = "claude-3-opus-20240229"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client = Mock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        client = AnthropicClient(
            api_key="test-key", model="claude-3-opus-20240229", client=mock_client
        )
        messages = [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Hello!"),
        ]

        # Act
        response = await client.generate(messages=messages, temperature=0.0)

        # Assert
        assert response.content == "Hi there!"
        # Verify system message was passed separately
        call_args = mock_client.messages.create.call_args
        assert "system" in call_args.kwargs
        assert call_args.kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        """Test streaming completion generation"""
        # Arrange
        from app.services.llm.anthropic_client import AnthropicClient
        from app.services.llm.base import LLMMessage

        # Mock streaming response
        async def mock_stream():
            text_chunks = ["Hi", " there", "!"]
            for chunk in text_chunks:
                mock_event = Mock()
                mock_event.type = "content_block_delta"
                mock_event.delta = Mock()
                mock_event.delta.text = chunk
                yield mock_event

        stream_context = Mock()
        stream_context.__aenter__ = AsyncMock(return_value=mock_stream())
        stream_context.__aexit__ = AsyncMock(return_value=False)
        mock_client = Mock()
        mock_client.messages.stream = Mock(return_value=stream_context)
        client = AnthropicClient(
            api_key="test-key", model="claude-3-opus-20240229", client=mock_client
        )
        messages = [LLMMessage(role="user", content="Hello!")]

        # Act
        chunks = []
        async for chunk in client.generate_stream(messages=messages):
            chunks.append(chunk)

        # Assert
        assert chunks == ["Hi", " there", "!"]

    @pytest.mark.asyncio
    async def test_generate_api_error(self):
        """Test handling API errors"""
        # Arrange
        from app.services.llm.anthropic_client import AnthropicClient
        from app.services.llm.base import LLMMessage
        from app.core.exceptions import ExternalServiceError

        mock_client = Mock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API Error"))
        client = AnthropicClient(
            api_key="test-key", model="claude-3-opus-20240229", client=mock_client
        )
        messages = [LLMMessage(role="user", content="Hello!")]

        # Act & Assert
        with pytest.raises(ExternalServiceError):
            await client.generate(messages=messages)

    def test_estimate_tokens(self):
        """Test token estimation"""
        # Arrange
        from app.services.llm.anthropic_client import AnthropicClient

        client = AnthropicClient(api_key="test-key", model="claude-3-opus-20240229")

        # Act
        count = client.estimate_tokens("Hello, world!")

        # Assert
        assert count > 0

    @pytest.mark.asyncio
    async def test_count_tokens(self):
        """Test token counting"""
        # Arrange
        from app.services.llm.anthropic_client import AnthropicClient
        from app.services.llm.base import LLMMessage

        client = AnthropicClient(api_key="test-key", model="claude-3-opus-20240229")
        messages = [
            LLMMessage(role="user", content="Hello!"),
            LLMMessage(role="assistant", content="Hi there!"),
        ]

        # Act
        count = await client.count_tokens(messages)

        # Assert
        assert count > 0

    def test_validate_model(self):
        """Test model validation"""
        # Arrange
        from app.services.llm.anthropic_client import AnthropicClient

        # Act & Assert - Valid models
        client1 = AnthropicClient(api_key="test-key", model="claude-3-opus-20240229")
        assert client1.model == "claude-3-opus-20240229"

        client2 = AnthropicClient(api_key="test-key", model="claude-3-sonnet-20240229")
        assert client2.model == "claude-3-sonnet-20240229"
