"""
OpenAI LLM client implementation.

Provides integration with OpenAI's GPT models.
"""

from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from app.core.exceptions import ExternalServiceError
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase
from app.services.llm.token_counter import TokenCounter


class OpenAIClient(LLMServiceBase):
    """
    OpenAI LLM service implementation.

    Supports GPT-3.5, GPT-4, and other OpenAI models.
    """

    # Available OpenAI models
    MODELS = [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
    ]

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        max_tokens: int | None = None,
        temperature: float | None = None,
        client: Any | None = None,
        base_url: str | None = None,
    ) -> None:
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4-turbo-preview)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            client: Pre-built client (tests / dependency injection)
            base_url: Override for OpenAI-compatible endpoints (MiniMax,
                DeepSeek, vLLM, ...); None uses the official OpenAI API
        """
        super().__init__(api_key, model, max_tokens, temperature)

        # Initialize async OpenAI client
        self.client = (
            client if client is not None else AsyncOpenAI(api_key=api_key, base_url=base_url)
        )

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion from OpenAI.

        Args:
            messages: List of conversation messages
            max_tokens: Override max tokens
            temperature: Override temperature
            **kwargs: Additional OpenAI parameters

        Returns:
            LLMResponse: Generated response

        Raises:
            ExternalServiceError: If API call fails
        """
        try:
            # Prepare parameters
            params: dict[str, Any] = {
                "model": self.model,
                "messages": self._format_messages(messages),
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
                "temperature": (temperature if temperature is not None else self.temperature),
            }

            # Add any additional parameters
            params.update(kwargs)

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            # Call OpenAI API
            response = await self.client.chat.completions.create(**params)

            # Extract response data
            choice = response.choices[0]
            message = choice.message
            # Tool-call responses may carry no content — keep the str
            # contract instead of leaking None downstream.
            content = message.content or ""
            # The SDK returns a list of tool-call objects, or None when the
            # model produced a plain answer; isinstance keeps malformed
            # payloads (or test doubles) from breaking extraction.
            raw_calls = getattr(message, "tool_calls", None)
            tool_calls: list[dict[str, str]] | None = None
            if isinstance(raw_calls, list):
                tool_calls = [
                    {
                        "id": call.id or "",
                        "name": call.function.name if call.function else "",
                        "arguments": call.function.arguments if call.function else "{}",
                    }
                    for call in raw_calls
                ]
            finish_reason = choice.finish_reason
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return LLMResponse(
                content=content,
                model=response.model,
                finish_reason=finish_reason,
                usage=usage,
                tool_calls=tool_calls,
            )

        except Exception as e:
            raise ExternalServiceError(
                service="OpenAI",
                message=f"Failed to generate completion: {str(e)}",
            ) from e

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

        Routes through ``generate`` (the SDK accepts ``tools`` /
        ``tool_choice`` natively) so tool calls inherit the same error
        handling — and, when wrapped by ResilientLLMService, the same
        retry / failover chain as every other LLM call.

        Args:
            messages: Conversation so far (includes "tool" messages)
            tools: Tool schemas in OpenAI format
            max_tokens: Override max tokens
            temperature: Override temperature
            **kwargs: Additional OpenAI parameters

        Returns:
            LLMResponse: Response with ``tool_calls`` populated when the
            model requested a tool invocation
        """
        return await self.generate(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice="auto",
            **kwargs,
        )

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming completion from OpenAI.

        Args:
            messages: List of conversation messages
            max_tokens: Override max tokens
            temperature: Override temperature
            **kwargs: Additional OpenAI parameters

        Yields:
            str: Content chunks

        Raises:
            ExternalServiceError: If API call fails
        """
        try:
            # Prepare parameters
            params: dict[str, Any] = {
                "model": self.model,
                "messages": self._format_messages(messages),
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
                "temperature": (temperature if temperature is not None else self.temperature),
                "stream": True,
            }

            # Add any additional parameters
            params.update(kwargs)

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            # Call OpenAI API with streaming
            stream = await self.client.chat.completions.create(**params)

            # Yield chunks as they arrive
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise ExternalServiceError(
                service="OpenAI",
                message=f"Failed to generate streaming completion: {str(e)}",
            ) from e

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using character-based estimation.

        Args:
            text: Text to estimate

        Returns:
            int: Estimated token count
        """
        return TokenCounter.estimate(text)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        """
        Count actual tokens in messages.

        Args:
            messages: List of messages

        Returns:
            int: Total token count
        """
        return TokenCounter.count_messages(messages, self.model)

    def get_max_context_tokens(self) -> int:
        """
        Get the maximum context window for the current model.

        Returns:
            int: Maximum tokens in context window
        """
        context_windows = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4-turbo-preview": 128000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
        }

        return context_windows.get(self.model, 4096)
