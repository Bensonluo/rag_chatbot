"""
GLM (Zhipu AI) embedding service implementation.

Provides integration with GLM embedding models via API.
"""
import time
from typing import List

import httpx
import jwt

from app.services.embeddings.base import EmbeddingServiceBase, EmbeddingResult
from app.core.exceptions import ExternalServiceError


def _generate_token(api_key: str, exp_seconds: int = 3600) -> str:
    """Generate JWT token for Zhipu AI API authentication."""
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


class GLMEmbeddingService(EmbeddingServiceBase):
    """
    GLM embedding service implementation.

    Supports GLM embedding models from Zhipu AI.
    """

    # Available GLM embedding models
    MODELS = {
        "embedding-2": 1024,  # GLM embedding v2 (recommended)
        "embedding-3": 1024,  # GLM embedding v3 (if available)
    }

    # GLM API base URL
    API_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4/"

    def __init__(
        self,
        api_key: str,
        model: str = "embedding-2",
        dimensions: int = None,
    ) -> None:
        """
        Initialize GLM embedding client.

        Args:
            api_key: Zhipu AI API key (format: {id}.{secret})
            model: Model name (default: embedding-2)
            dimensions: Override dimensions (auto-detected if not provided)
        """
        if dimensions is None:
            if model in self.MODELS:
                dimensions = self.MODELS[model]
            else:
                dimensions = 1024

        super().__init__(model=model, dimensions=dimensions)

        self.api_key = api_key

        self.client = httpx.AsyncClient(
            base_url=self.API_BASE_URL,
            headers={"Content-Type": "application/json"},
            timeout=60.0,
        )

    def _auth_headers(self) -> dict:
        return {"Authorization": _generate_token(self.api_key)}

    async def embed(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts using GLM API.

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
                    embeddings=[],
                    model=self.model,
                    dimensions=self.dimensions,
                    tokens_used=0
                )

            # Prepare request payload
            payload = {
                "model": self.model,
                "input": texts
            }

            # Call GLM embedding API
            response = await self.client.post(
                "embeddings", json=payload, headers=self._auth_headers()
            )
            response.raise_for_status()

            data = response.json()

            # Extract embeddings from response
            # GLM API format: {"object": "list", "data": [{"embedding": [...], "index": 0}, ...], ...}
            embeddings_data = data.get("data", [])

            # Sort by index to ensure correct order
            embeddings_data.sort(key=lambda x: x.get("index", 0))

            # Extract embedding vectors
            embeddings = [item["embedding"] for item in embeddings_data]

            # Get token usage
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)

            return EmbeddingResult(
                embeddings=embeddings,
                model=self.model,
                dimensions=self.dimensions,
                tokens_used=tokens_used
            )

        except httpx.HTTPStatusError as e:
            raise ExternalServiceError(
                service="GLM Embeddings",
                message=f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
            ) from e
        except Exception as e:
            raise ExternalServiceError(
                service="GLM Embeddings",
                message=f"Failed to generate embeddings: {str(e)}"
            ) from e

    async def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            List[float]: Embedding vector

        Raises:
            ExternalServiceError: If embedding generation fails
        """
        result = await self.embed([text])
        return result.embeddings[0] if result.embeddings else []

    async def close(self) -> None:
        """
        Close the HTTP client.

        Should be called when the client is no longer needed.
        """
        await self.client.aclose()

    async def __aenter__(self) -> "GLMEmbeddingService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
