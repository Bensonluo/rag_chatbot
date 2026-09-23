"""
GLM (Zhipu AI) LLM client implementation.

Provides integration with GLM models from Zhipu AI.
"""

import time
from collections.abc import AsyncGenerator
from types import TracebackType
from typing import Any

import httpx
import jwt

from app.core.exceptions import ExternalServiceError
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase
from app.services.llm.token_counter import TokenCounter


def _generate_token(api_key: str, exp_seconds: int = 3600) -> str:
    """Generate JWT token for Zhipu AI API authentication.

    Zhipu AI API keys are in '{id}.{secret}' format.
    A JWT must be generated and signed with the secret.
    """
    try:
        api_id, api_secret = api_key.split(".")
    except ValueError:
        return api_key

    now = int(time.time())
    payload = {
        "api_key": api_id,
        "exp": now + exp_seconds,
        "timestamp": now,
    }
    return jwt.encode(
        payload,
        api_secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )


def _parse_tool_calls(raw: Any) -> list[dict[str, Any]] | None:
    """Normalize wire-format tool_calls to flat dicts.

    Accepts ``[{"id", "type", "function": {"name", "arguments"}}]``
    (OpenAI/GLM wire format) and returns ``[{"id", "name",
    "arguments"}]`` with arguments kept as the raw JSON string — the
    caller parses it against the tool's schema. Returns None when the
    model did not request any tool.
    """
    if not raw:
        return None
    parsed: list[dict[str, Any]] = []
    for call in raw:
        function = call.get("function") or {}
        parsed.append(
            {
                "id": call.get("id", ""),
                "name": function.get("name", ""),
                "arguments": function.get("arguments", "{}"),
            }
        )
    return parsed


class GLMClient(LLMServiceBase):
    """
    GLM (Zhipu AI) LLM service implementation.

    Supports GLM-5.3-Flash, GLM-5.2, GLM-4, and other Zhipu AI models.
    """

    # Available GLM models
    MODELS = [
        "glm-5.3-flash",
        "glm-5.2",
        "glm-5.1",
        "glm-4",
    ]

    # GLM API base URL
    API_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4/"

    def __init__(
        self,
        api_key: str,
        model: str = "glm-5.3-flash",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        """
        Initialize GLM client.

        Args:
            api_key: Zhipu AI API key (format: {id}.{secret})
            model: Model name (default: glm-5.3-flash)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        super().__init__(api_key, model, max_tokens, temperature)

        self.client = httpx.AsyncClient(
            base_url=self.API_BASE_URL,
            headers={"Content-Type": "application/json"},
            timeout=120.0,
        )

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": _generate_token(self.api_key)}

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion from GLM.

        Args:
            messages: List of conversation messages
            max_tokens: Override max tokens
            temperature: Override temperature
            **kwargs: Additional GLM parameters

        Returns:
            LLMResponse: Generated response

        Raises:
            ExternalServiceError: If API call fails
        """
        try:
            # Prepare parameters
            params = {
                "model": self.model,
                "messages": self._format_messages(messages),
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
            }

            # Add any additional parameters
            params.update(kwargs)

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            # Call GLM API
            response = await self.client.post(
                "chat/completions", json=params, headers=self._auth_headers()
            )
            response.raise_for_status()

            data = response.json()

            # Extract response data
            choice = data["choices"][0]
            message = choice["message"]
            # Tool-call responses may carry no content at all — keep the
            # str contract instead of leaking None downstream.
            content = message.get("content") or ""
            tool_calls = _parse_tool_calls(message.get("tool_calls"))
            finish_reason = choice.get("finish_reason")
            usage = {
                "prompt_tokens": data["usage"]["prompt_tokens"],
                "completion_tokens": data["usage"]["completion_tokens"],
                "total_tokens": data["usage"]["total_tokens"],
            }

            return LLMResponse(
                content=content,
                model=data["model"],
                finish_reason=finish_reason,
                usage=usage,
                tool_calls=tool_calls,
            )

        except httpx.HTTPStatusError as e:
            raise ExternalServiceError(
                service="GLM",
                message=f"HTTP error occurred: {e.response.status_code}",
                status_code=e.response.status_code,
            ) from e
        except Exception as e:
            raise ExternalServiceError(
                service="GLM",
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

        GLM speaks the OpenAI-compatible wire format (verified against
        docs.bigmodel.cn / docs.z.ai): tools + tool_choice in the
        request, ``message.tool_calls`` in the response. Routes through
        ``generate`` so tool calls inherit the same error handling —
        and, when wrapped by ResilientLLMService, the same retry /
        failover chain as every other LLM call.

        Args:
            messages: Conversation so far (includes "tool" messages)
            tools: Tool schemas in OpenAI format
            max_tokens: Override max tokens
            temperature: Override temperature
            **kwargs: Additional GLM parameters

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
        Generate a streaming completion from GLM.

        Args:
            messages: List of conversation messages
            max_tokens: Override max tokens
            temperature: Override temperature
            **kwargs: Additional GLM parameters

        Yields:
            str: Content chunks

        Raises:
            ExternalServiceError: If API call fails
        """
        try:
            # Prepare parameters
            params = {
                "model": self.model,
                "messages": self._format_messages(messages),
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
                "stream": True,
            }

            # Add any additional parameters
            params.update(kwargs)

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            # Call GLM API with streaming
            async with self.client.stream(
                "POST", "chat/completions", json=params, headers=self._auth_headers()
            ) as response:
                response.raise_for_status()

                # Parse server-sent events
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        # Skip [DONE] marker
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            import json

                            data = json.loads(data_str)

                            # Extract content from delta
                            if data.get("choices") and data["choices"][0].get("delta"):
                                delta = data["choices"][0]["delta"]
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPStatusError as e:
            raise ExternalServiceError(
                service="GLM",
                message=f"HTTP error occurred during streaming: {e.response.status_code}",
                status_code=e.response.status_code,
            ) from e
        except Exception as e:
            raise ExternalServiceError(
                service="GLM",
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
        # GLM uses similar tokenization to GPT models
        return TokenCounter.count_messages(messages, "gpt-4")

    def get_max_context_tokens(self) -> int:
        """
        Get the maximum context window for the current model.

        Returns:
            int: Maximum tokens in context window
        """
        context_windows = {
            "glm-5.3-flash": 128000,
            "glm-5.2": 128000,
            "glm-5.1": 128000,
            "glm-4": 128000,
        }

        return context_windows.get(self.model, 128000)

    async def close(self) -> None:
        """
        Close the HTTP client.

        Should be called when the client is no longer needed.
        """
        await self.client.aclose()

    async def __aenter__(self) -> "GLMClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.close()
