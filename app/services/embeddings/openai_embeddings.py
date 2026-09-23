"""
OpenAI embedding service implementation.

Provides integration with OpenAI's embedding models via the SDK.
"""

from types import TracebackType
from typing import Any

from openai import AsyncOpenAI

from app.core.exceptions import ExternalServiceError
from app.services.embeddings.base import EmbeddingResult, EmbeddingServiceBase


class OpenAIEmbeddingService(EmbeddingServiceBase):
    """
    OpenAI embedding service implementation.

    Supports the text-embedding-3 model family.
    """

    # Available models and their default dimensions
    MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        client: Any | None = None,
    ) -> None:
        """
        Initialize OpenAI embedding client.

        Args:
            api_key: OpenAI API key
            model: Model name (default: text-embedding-3-small)
            dimensions: Override dimensions (model default if not provided)
            client: Pre-built client (tests / dependency injection)
        """
        if dimensions is None:
            dimensions = self.MODELS.get(model, 1536)

        super().__init__(model=model, dimensions=dimensions)

        self.api_key = api_key
        self.client = client if client is not None else AsyncOpenAI(api_key=api_key)

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts using the OpenAI API.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult: Generated embeddings with metadata

        Raises:
            ExternalServiceError: If API call fails
        """
        try:
            if not texts:
                return EmbeddingResult(
                    embeddings=[], model=self.model, dimensions=self.dimensions, tokens_used=0
                )

            response = await self.client.embeddings.create(
                input=texts,
                model=self.model,
            )

            # Sort by index — the API does not guarantee response order.
            data = sorted(response.data, key=lambda item: item.index)
            embeddings = [list(item.embedding) for item in data]

            usage = response.usage
            tokens_used = usage.total_tokens if usage is not None else 0

            return EmbeddingResult(
                embeddings=embeddings,
                model=self.model,
                dimensions=self.dimensions,
                tokens_used=tokens_used,
            )

        except Exception as e:
            raise ExternalServiceError(
                service="OpenAI Embeddings", message=f"Failed to generate embeddings: {str(e)}"
            ) from e

    async def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            List[float]: Embedding vector

        Raises:
            ExternalServiceError: If API call fails
        """
        result = await self.embed([text])
        return result.embeddings[0] if result.embeddings else []

    async def close(self) -> None:
        """
        Close the underlying HTTP client.

        Should be called when the service is no longer needed.
        """
        await self.client.close()

    async def __aenter__(self) -> "OpenAIEmbeddingService":
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
