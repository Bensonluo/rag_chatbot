"""Tests for OpenAI LLM client"""

from unittest.mock import AsyncMock, Mock

import pytest


class TestOpenAIClient:
    """Test OpenAI LLM client implementation"""

    def test_initialization(self):
        """Test client initialization"""
        # Arrange & Act
        from app.services.llm.openai_client import OpenAIClient

        client = OpenAIClient(api_key="test-key", model="gpt-4", max_tokens=1000, temperature=0.7)

        # Assert
        assert client.api_key == "test-key"
        assert client.model == "gpt-4"
        assert client.max_tokens == 1000
        assert client.temperature == 0.7

    def test_initialization_defaults(self):
        """Test client initialization with defaults"""
        # Arrange & Act
        from app.services.llm.openai_client import OpenAIClient

        client = OpenAIClient(api_key="test-key", model="gpt-4")

        # Assert
        assert client.max_tokens is None
        assert client.temperature is None

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful completion generation"""
        # Arrange
        from app.services.llm.base import LLMMessage
        from app.services.llm.openai_client import OpenAIClient

        # Mock OpenAI API
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hi there!"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "gpt-4"
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        client = OpenAIClient(api_key="test-key", model="gpt-4", client=mock_client)
        messages = [LLMMessage(role="user", content="Hello!")]

        # Act
        response = await client.generate(messages=messages)

        # Assert
        assert response.content == "Hi there!"
        assert response.model == "gpt-4"
        assert response.finish_reason == "stop"
        assert response.usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_generate_with_overrides(self):
        """Test generation with parameter overrides"""
        # Arrange
        from app.services.llm.base import LLMMessage
        from app.services.llm.openai_client import OpenAIClient

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.model = "gpt-4"
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        client = OpenAIClient(
            api_key="test-key",
            model="gpt-4",
            max_tokens=500,
            temperature=0.5,
            client=mock_client,
        )
        messages = [LLMMessage(role="user", content="Hello!")]

        # Act - Override parameters
        response = await client.generate(messages=messages, max_tokens=1000, temperature=0.0)

        # Assert
        assert response.content == "Response"
        # Verify the override was passed to API
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["max_tokens"] == 1000
        assert call_args.kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        """Test streaming completion generation"""
        # Arrange
        from app.services.llm.base import LLMMessage
        from app.services.llm.openai_client import OpenAIClient

        # Mock streaming response
        async def mock_stream():
            chunks = ["Hi", " there", "!"]
            for chunk in chunks:
                mock_chunk = Mock()
                mock_chunk.choices = [Mock()]
                mock_chunk.choices[0].delta.content = chunk
                yield mock_chunk

        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        client = OpenAIClient(api_key="test-key", model="gpt-4", client=mock_client)
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
        from app.core.exceptions import ExternalServiceError
        from app.services.llm.base import LLMMessage
        from app.services.llm.openai_client import OpenAIClient

        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        client = OpenAIClient(api_key="test-key", model="gpt-4", client=mock_client)
        messages = [LLMMessage(role="user", content="Hello!")]

        # Act & Assert
        with pytest.raises(ExternalServiceError):
            await client.generate(messages=messages)

    def test_estimate_tokens(self):
        """Test token estimation"""
        # Arrange
        from app.services.llm.openai_client import OpenAIClient

        client = OpenAIClient(api_key="test-key", model="gpt-4")

        # Act
        count = client.estimate_tokens("Hello, world!")

        # Assert
        assert count > 0

    @pytest.mark.asyncio
    async def test_count_tokens(self):
        """Test token counting"""
        # Arrange
        from app.services.llm.base import LLMMessage
        from app.services.llm.openai_client import OpenAIClient

        client = OpenAIClient(api_key="test-key", model="gpt-4")
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
        from app.services.llm.openai_client import OpenAIClient

        # Act & Assert - Valid models
        client1 = OpenAIClient(api_key="test-key", model="gpt-4")
        assert client1.model == "gpt-4"

        client2 = OpenAIClient(api_key="test-key", model="gpt-3.5-turbo")
        assert client2.model == "gpt-3.5-turbo"


class TestOpenAICompatibleBaseUrl:
    """base_url support routes the SDK at any OpenAI-compatible provider."""

    def test_base_url_passed_to_sdk_client(self):
        # Arrange
        from unittest.mock import patch

        from app.services.llm.openai_client import OpenAIClient

        # Act
        with patch("app.services.llm.openai_client.AsyncOpenAI") as mock_sdk:
            OpenAIClient(api_key="k", base_url="https://api.minimaxi.com/v1")

        # Assert
        mock_sdk.assert_called_once_with(api_key="k", base_url="https://api.minimaxi.com/v1")

    def test_default_base_url_is_none(self):
        from unittest.mock import patch

        from app.services.llm.openai_client import OpenAIClient

        with patch("app.services.llm.openai_client.AsyncOpenAI") as mock_sdk:
            OpenAIClient(api_key="k")

        mock_sdk.assert_called_once_with(api_key="k", base_url=None)

    def test_injected_client_skips_sdk_construction(self):
        from unittest.mock import patch

        from app.services.llm.openai_client import OpenAIClient

        with patch("app.services.llm.openai_client.AsyncOpenAI") as mock_sdk:
            OpenAIClient(api_key="k", client=object(), base_url="https://x")

        mock_sdk.assert_not_called()
