"""
GLM (Zhipu AI) LLM client implementation.

Provides integration with GLM models from Zhipu AI.
"""
import time
from typing import AsyncGenerator, Optional, List, Dict, Any

import httpx
import jwt

from app.services.llm.base import LLMServiceBase, LLMMessage, LLMResponse
from app.services.llm.token_counter import TokenCounter
from app.core.exceptions import ExternalServiceError


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


class GLMClient(LLMServiceBase):
    """
    GLM (Zhipu AI) LLM service implementation.

    Supports GLM-4, GLM-3-Turbo, and other Zhipu AI models.
    """

    # Available GLM models
    MODELS = [
        "glm-5.1",
        "glm-5.2",
        "glm-5.2",
        "glm-4",
        "glm-5.2",
        "glm-5.2",
        "glm-5.2",
        "glm-5.2",
        "glm-5.2",
    ]

    # GLM API base URL
    API_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4/"

    def __init__(
        self,
        api_key: str,
        model: str = "glm-5.1",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """
        Initialize GLM client.

        Args:
            api_key: Zhipu AI API key (format: {id}.{secret})
            model: Model name (default: glm-5.2)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        super().__init__(api_key, model, max_tokens, temperature)

        self.client = httpx.AsyncClient(
            base_url=self.API_BASE_URL,
            headers={"Content-Type": "application/json"},
            timeout=120.0,
        )

    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": _generate_token(self.api_key)}

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
            response = await self.client.post(
                "chat/completions", json=params, headers=self._auth_headers()
            )
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
            "glm-5.1": 128000,
            "glm-5.2": 128000,
            "glm-5.2": 128000,
            "glm-4": 128000,
            "glm-5.2": 128000,
            "glm-5.2": 128000,
            "glm-5.2": 128000,
            "glm-5.2": 128000,
            "glm-5.2": 128000,
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
