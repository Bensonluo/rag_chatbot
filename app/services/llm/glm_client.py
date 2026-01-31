"""
GLM (Zhipu AI) LLM client implementation.

Provides integration with GLM models from Zhipu AI.
"""
from typing import AsyncGenerator, Optional, List, Dict, Any

import httpx

from app.services.llm.base import LLMServiceBase, LLMMessage, LLMResponse
from app.services.llm.token_counter import TokenCounter
from app.core.exceptions import ExternalServiceError


class GLMClient(LLMServiceBase):
    """
    GLM (Zhipu AI) LLM service implementation.

    Supports GLM-4, GLM-3-Turbo, and other Zhipu AI models.
    """

    # Available GLM models
    MODELS = [
        "glm-4-plus",
        "glm-4-0520",
        "glm-4",
        "glm-4.5-air",
        "glm-4-air",
        "glm-4-airx",
        "glm-4-flash",
        "glm-3-turbo",
    ]

    # GLM API base URL
    API_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

    def __init__(
        self,
        api_key: str,
        model: str = "glm-4.5-air",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """
        Initialize GLM client.

        Args:
            api_key: Zhipu AI API key
            model: Model name (default: glm-4-plus)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        super().__init__(api_key, model, max_tokens, temperature)

        # Initialize async HTTP client with custom headers for JWT authentication
        self.client = httpx.AsyncClient(
            base_url=self.API_BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    async def generate(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
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
            response = await self.client.post("chat/completions", json=params)
            response.raise_for_status()

            data = response.json()

            # Extract response data
            choice = data["choices"][0]
            content = choice["message"]["content"]
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
            )

        except httpx.HTTPStatusError as e:
            raise ExternalServiceError(
                service="GLM",
                message=f"HTTP error occurred: {e.response.status_code} - {e.response.text}",
            ) from e
        except Exception as e:
            raise ExternalServiceError(
                service="GLM",
                message=f"Failed to generate completion: {str(e)}",
            ) from e

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
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
            async with self.client.stream("POST", "chat/completions", json=params) as response:
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

    async def count_tokens(self, messages: List[LLMMessage]) -> int:
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
            "glm-4-plus": 128000,
            "glm-4-0520": 128000,
            "glm-4": 128000,
            "glm-4.5-air": 128000,
            "glm-4-air": 128000,
            "glm-4-airx": 128000,
            "glm-4-flash": 128000,
            "glm-3-turbo": 128000,
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

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
