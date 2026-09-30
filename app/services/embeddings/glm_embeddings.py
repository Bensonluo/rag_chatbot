"""
GLM (Zhipu AI) embedding service implementation.

Provides integration with GLM embedding models via API.
"""

import time
from types import TracebackType

import httpx
import jwt

from app.config.settings import settings
from app.core.exceptions import ExternalServiceError
from app.services.embeddings.base import EmbeddingResult, EmbeddingServiceBase

# Embedding calls are short and batched (<=16 chunks); a modest pool
# with a warm keepalive floor matches the traffic shape. Streaming
# LLM-scale pools would over-commit file descriptors for no benefit.
_EMBED_POOL_LIMITS = httpx.Limits(
    max_connections=64,
    max_keepalive_connections=32,
)

# GLM rejects embedding input arrays larger than 64 items (platform
# error 1214: "input数组最大不得超过64条"). The Coding Plan package
# also enforces a low concurrent-request ceiling, so oversize inputs
# are chunked at the cap and the chunks are sent sequentially —
# never fanned out concurrently.
_MAX_BATCH_ITEMS = 64


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
        "embedding-3": 2048,  # GLM embedding v3 — live-verified 2048 dims
    }

    def __init__(
        self,
        api_key: str,
        model: str = "embedding-2",
        dimensions: int | None = None,
        base_url: str | None = None,
    ) -> None:
        """
        Initialize GLM embedding client.

        Args:
            api_key: Zhipu AI API key (format: {id}.{secret})
            model: Model name (default: embedding-2)
            dimensions: Override dimensions (auto-detected if not provided)
            base_url: Override the API endpoint (defaults to GLM_BASE_URL —
                the standard PAAS API; Coding Plan keys point it at the
                coding endpoint instead)
        """
        if dimensions is None:
            dimensions = self.MODELS.get(model, 1024)

        super().__init__(model=model, dimensions=dimensions)

        self.api_key = api_key

        self.client = httpx.AsyncClient(
            base_url=base_url or settings.GLM_BASE_URL,
            headers={"Content-Type": "application/json"},
            timeout=60.0,
            limits=_EMBED_POOL_LIMITS,
        )

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": _generate_token(self.api_key)}

    async def embed(self, texts: list[str]) -> EmbeddingResult:
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
                    embeddings=[], model=self.model, dimensions=self.dimensions, tokens_used=0
                )

            embeddings: list[list[float]] = []
            tokens_used = 0

            # Sequential chunks: order-preserving and within the
            # concurrency ceiling of the Coding Plan package.
            for start in range(0, len(texts), _MAX_BATCH_ITEMS):
                batch = texts[start : start + _MAX_BATCH_ITEMS]

                response = await self.client.post(
                    "embeddings",
                    json={"model": self.model, "input": batch},
                    headers=self._auth_headers(),
                )
                response.raise_for_status()

                data = response.json()

                # Extract embeddings from response
                # GLM API format: {"object": "list", "data": [{"embedding": [...], "index": 0}, ...], ...}
                embeddings_data = data.get("data", [])

                # Sort by index to ensure correct order
                embeddings_data.sort(key=lambda x: x.get("index", 0))

                if len(embeddings_data) != len(batch):
                    # A short batch would silently misalign
                    # callers' owners<->vectors pairing downstream.
                    raise ExternalServiceError(
                        service="GLM Embeddings",
                        message=(
                            f"Batch mismatch: sent {len(batch)} inputs, "
                            f"received {len(embeddings_data)} embeddings"
                        ),
                    )

                embeddings.extend(item["embedding"] for item in embeddings_data)
                tokens_used += data.get("usage", {}).get("total_tokens", 0)

            return EmbeddingResult(
                embeddings=embeddings,
                model=self.model,
                dimensions=self.dimensions,
                tokens_used=tokens_used,
            )

        except httpx.HTTPStatusError as e:
            raise ExternalServiceError(
                service="GLM Embeddings",
                message=f"HTTP error occurred: {e.response.status_code} - {e.response.text}",
            ) from e
        except Exception as e:
            raise ExternalServiceError(
                service="GLM Embeddings", message=f"Failed to generate embeddings: {str(e)}"
            ) from e

    async def embed_single(self, text: str) -> list[float]:
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

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.close()
