"""Tests for LLM base interface and service"""

import pytest

from app.services.llm.base import LLMMessage, LLMServiceBase


class StubService(LLMServiceBase):
    """Minimal concrete subclass so the ABC's own method bodies can be
    exercised (the class itself can no longer be instantiated)."""

    async def generate(self, messages, max_tokens=None, temperature=None, **kwargs):
        return await super().generate(
            messages, max_tokens=max_tokens, temperature=temperature, **kwargs
        )

    async def generate_stream(self, messages, max_tokens=None, temperature=None, **kwargs):
        async for chunk in super().generate_stream(
            messages, max_tokens=max_tokens, temperature=temperature, **kwargs
        ):
            yield chunk

    def estimate_tokens(self, text):
        return super().estimate_tokens(text)

    async def count_tokens(self, messages):
        return await super().count_tokens(messages)


class TestLLMMessage:
    """Test LLM message data structure"""

    def test_create_user_message(self):
        """Test creating a user message"""
        # Arrange & Act

        message = LLMMessage(role="user", content="Hello, world!")

        # Assert
        assert message.role == "user"
        assert message.content == "Hello, world!"

    def test_create_system_message(self):
        """Test creating a system message"""
        # Arrange & Act

        message = LLMMessage(role="system", content="You are a helpful assistant.")

        # Assert
        assert message.role == "system"
        assert message.content == "You are a helpful assistant."

    def test_create_assistant_message(self):
        """Test creating an assistant message"""
        # Arrange & Act

        message = LLMMessage(role="assistant", content="Hi there!")

        # Assert
        assert message.role == "assistant"
        assert message.content == "Hi there!"

    def test_message_to_dict(self):
        """Test converting message to dictionary"""
        # Arrange

        message = LLMMessage(role="user", content="Test")

        # Act
        msg_dict = message.to_dict()

        # Assert
        assert msg_dict == {"role": "user", "content": "Test"}

    def test_message_from_dict(self):
        """Test creating message from dictionary"""
        # Arrange

        msg_dict = {"role": "user", "content": "Test"}

        # Act
        message = LLMMessage.from_dict(msg_dict)

        # Assert
        assert message.role == "user"
        assert message.content == "Test"


class TestLLMResponse:
    """Test LLM response data structure"""

    def test_create_response(self):
        """Test creating an LLM response"""
        # Arrange & Act
        from app.services.llm.base import LLMResponse

        response = LLMResponse(
            content="Hello!",
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        # Assert
        assert response.content == "Hello!"
        assert response.model == "gpt-4"
        assert response.usage["total_tokens"] == 15

    def test_response_with_finish_reason(self):
        """Test response with finish reason"""
        # Arrange & Act
        from app.services.llm.base import LLMResponse

        response = LLMResponse(content="Test", model="gpt-4", finish_reason="stop")

        # Assert
        assert response.finish_reason == "stop"


class TestLLMServiceBase:
    """Test LLM service base class via a minimal concrete subclass."""

    def test_abc_cannot_be_instantiated_directly(self):
        """The abstract class itself cannot be instantiated"""
        from app.services.llm.base import LLMServiceBase

        with pytest.raises(TypeError):
            LLMServiceBase(api_key="k", model="m")  # type: ignore[abstract]

    def test_stub_initialization_stores_config(self):
        """__init__ stores key/model and normalizes token/temperature fields"""

        service = StubService(api_key="k", model="gpt-4", max_tokens=100, temperature=0.5)

        assert service.api_key == "k"
        assert service.model == "gpt-4"
        assert service.max_tokens == 100
        assert service.temperature == 0.5
        # to_dict contract on the message model used across the suite
        assert LLMMessage(role="user", content="x").to_dict() == {
            "role": "user",
            "content": "x",
        }

    @pytest.mark.asyncio
    async def test_generate_not_implemented(self):
        """The ABC body for generate raises NotImplementedError"""

        service = StubService(api_key="k", model="m")
        messages = [LLMMessage(role="user", content="Test")]

        with pytest.raises(NotImplementedError):
            await service.generate(messages=messages)

    @pytest.mark.asyncio
    async def test_generate_stream_not_implemented(self):
        """The ABC body for generate_stream raises NotImplementedError"""

        service = StubService(api_key="k", model="m")
        messages = [LLMMessage(role="user", content="Test")]

        with pytest.raises(NotImplementedError):
            async for _ in service.generate_stream(messages=messages):
                pass

    def test_estimate_tokens_not_implemented(self):
        """The ABC body for estimate_tokens raises NotImplementedError"""
        service = StubService(api_key="k", model="m")

        with pytest.raises(NotImplementedError):
            service.estimate_tokens("Sample text")

    @pytest.mark.asyncio
    async def test_count_tokens_not_implemented(self):
        """The ABC body for count_tokens raises NotImplementedError"""

        service = StubService(api_key="k", model="m")
        messages = [LLMMessage(role="user", content="Test")]

        with pytest.raises(NotImplementedError):
            await service.count_tokens(messages)

    def test_validate_max_tokens(self):
        """max_tokens validation accepts positive ints and None"""
        service = StubService(api_key="k", model="m")

        assert service._validate_max_tokens(100) == 100
        assert service._validate_max_tokens(1) == 1
        assert service._validate_max_tokens(None) is None

    def test_validate_max_tokens_invalid(self):
        """max_tokens validation rejects non-positive values"""
        service = StubService(api_key="k", model="m")

        with pytest.raises(ValueError):
            service._validate_max_tokens(0)

        with pytest.raises(ValueError):
            service._validate_max_tokens(-1)

    def test_validate_temperature(self):
        """temperature validation accepts [0.0, 1.0] and None"""
        service = StubService(api_key="k", model="m")

        assert service._validate_temperature(0.5) == 0.5
        assert service._validate_temperature(0.0) == 0.0
        assert service._validate_temperature(1.0) == 1.0
        assert service._validate_temperature(None) is None

    def test_validate_temperature_invalid(self):
        """temperature validation rejects values outside [0.0, 1.0]"""
        service = StubService(api_key="k", model="m")

        with pytest.raises(ValueError):
            service._validate_temperature(-0.1)

        with pytest.raises(ValueError):
            service._validate_temperature(1.1)
