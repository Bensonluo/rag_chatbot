"""
Base LLM service interface and data structures.

Provides abstract interfaces for LLM providers to implement.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMMessage:
    """
    Represents a message in a conversation with an LLM.

    Attributes:
        role: Message role ("user", "assistant", "system", or "tool")
        content: Message content
        tool_calls: Tool invocations the model requested (assistant
            messages only, function-calling wire format: list of
            ``{"id", "type", "function": {"name", "arguments"}}``)
        tool_call_id: Links a "tool" message to the assistant
            tool_call it answers
    """

    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert message to dictionary format.

        Returns:
            dict: Message as dictionary with role and content, plus
            the function-calling fields when present
        """
        data: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_calls is not None:
            data["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "LLMMessage":
        """
        Create message from dictionary.

        Args:
            data: Dictionary with role and content

        Returns:
            LLMMessage: Message instance
        """
        return cls(role=data["role"], content=data["content"])


@dataclass
class LLMResponse:
    """
    Represents a response from an LLM.

    Attributes:
        content: Generated text content
        model: Model name/identifier used
        finish_reason: Reason the generation finished
        usage: Token usage information
        tool_calls: Tool invocations the model requested, when the
            request carried tool schemas and the model chose to call
            one (each item: ``{"id", "name", "arguments"}`` with
            arguments as a raw JSON string). None on plain responses.
    """

    content: str
    model: str
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None


class LLMServiceBase(ABC):
    """
    Abstract base class for LLM service implementations.

    All LLM provider implementations (OpenAI, Anthropic, etc.) must
    inherit from this class and implement the required methods.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        """
        Initialize the LLM service.

        Args:
            api_key: API key for the LLM provider
            model: Model name/identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = self._validate_max_tokens(max_tokens)
        self.temperature = self._validate_temperature(temperature)

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            messages: List of conversation messages
            max_tokens: Override max tokens for this request
            temperature: Override temperature for this request
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse: The generated response

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("generate() must be implemented by subclass")

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming completion from the LLM.

        Implementations are async generator functions (contain ``yield``),
        so calling this returns an async iterator directly — no ``await``.

        Args:
            messages: List of conversation messages
            max_tokens: Override max tokens for this request
            temperature: Override temperature for this request
            **kwargs: Additional provider-specific parameters

        Yields:
            str: Chunks of generated content

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("generate_stream() must be implemented by subclass")
        yield  # unreachable — marks this method as an async generator function

    async def generate_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion with function-calling tool schemas.

        Providers whose API supports the OpenAI-compatible ``tools``
        wire format override this; the base implementation raises so
        callers can probe capability (and fall back to a plain
        pipeline) via NotImplementedError.

        Args:
            messages: Conversation so far, including "tool" result
                messages from earlier iterations
            tools: Tool schemas in OpenAI format (``{"type":
                "function", "function": {...}}``)
            max_tokens: Override max tokens for this request
            temperature: Override temperature for this request
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse: Response whose ``tool_calls`` is populated
            when the model requested a tool invocation

        Raises:
            NotImplementedError: Provider does not support function
                calling
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support function calling"
        )

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in text.

        Args:
            text: Text to estimate tokens for

        Returns:
            int: Estimated token count

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("estimate_tokens() must be implemented by subclass")

    @abstractmethod
    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        """
        Count the actual number of tokens in messages.

        Args:
            messages: List of messages to count tokens for

        Returns:
            int: Total token count

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("count_tokens() must be implemented by subclass")

    def _validate_max_tokens(self, max_tokens: int | None) -> int | None:
        """
        Validate max_tokens parameter.

        Args:
            max_tokens: Max tokens value to validate

        Returns:
            Optional[int]: Validated max_tokens or None

        Raises:
            ValueError: If max_tokens is invalid
        """
        if max_tokens is None:
            return None

        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")

        return max_tokens

    def _validate_temperature(self, temperature: float | None) -> float | None:
        """
        Validate temperature parameter.

        Args:
            temperature: Temperature value to validate

        Returns:
            Optional[float]: Validated temperature or None

        Raises:
            ValueError: If temperature is invalid
        """
        if temperature is None:
            return None

        if not (0.0 <= temperature <= 1.0):
            raise ValueError(f"temperature must be between 0.0 and 1.0, got {temperature}")

        return temperature

    def _format_messages(self, messages: list[LLMMessage]) -> list[dict[str, Any]]:
        """
        Format messages for API request.

        Args:
            messages: List of LLM messages

        Returns:
            List[dict]: Formatted messages
        """
        return [msg.to_dict() for msg in messages]
