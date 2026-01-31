"""
Local embedding service using sentence-transformers.

Supports BGE-M3 and other sentence-transformer models.
Runs entirely on-premises with no API calls.
"""
from typing import List
import asyncio
from functools import lru_cache

from app.services.embeddings.base import EmbeddingServiceBase, EmbeddingResult
from app.core.exceptions import ExternalServiceError


class LocalEmbeddingService(EmbeddingServiceBase):
    """
    Local embedding service using sentence-transformers.

    Supports models like:
    - BAAI/bge-m3-v2-zh (BGE-M3 Chinese-English)
    - sentence-transformers/all-MiniLM-L6-v2
    - Other HuggingFace sentence-transformer models
    """

    # Available models and their dimensions
    MODELS = {
        "bge-m3-v2-zh": 1024,  # BGE-M3 Chinese-English (recommended)
        "bge-large-zh-v1.5": 1024,  # BGE Large Chinese
        "bge-small-zh-v1.5": 512,  # BGE Small Chinese
        "multilingual-e5-large": 1024,  # E5 Large Multilingual
        "paraphrase-multilingual-MiniLM-L12-v2": 384,  # Fast multilingual
    }

    def __init__(
        self,
        model: str = "bge-m3-v2-zh",
        device: str = "cpu",
        dimensions: int = None,
    ) -> None:
        """
        Initialize local embedding service.

        Args:
            model: Model name (default: bge-m3-v2-zh)
            device: Device to use ('cpu' or 'cuda')
            dimensions: Override dimensions (auto-detected if not provided)
        """
        # Map model name to HuggingFace model ID
        model_mapping = {
            "bge-m3-v2-zh": "BAAI/bge-m3-v2-zh",
            "bge-large-zh-v1.5": "BAAI/bge-large-zh-v1.5",
            "bge-small-zh-v1.5": "BAAI/bge-small-zh-v1.5",
            "multilingual-e5-large": "intfloat/multilingual-e5-large",
            "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        }

        self.model_name = model
        self.hf_model_id = model_mapping.get(model, model)
        self.device = device

        # Auto-detect dimensions if not provided
        if dimensions is None:
            if model in self.MODELS:
                dimensions = self.MODELS[model]
            else:
                dimensions = 768  # Default for most ST models

        super().__init__(model=model, dimensions=dimensions)

        # Lazy loading: model loaded on first use
        self._model = None
        self._load_lock = asyncio.Lock()

    async def _load_model(self):
        """
        Load the sentence-transformer model (lazy loading).

        Runs in thread pool to avoid blocking event loop.
        """
        if self._model is not None:
            return

        async with self._load_lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return

            try:
                # Import here to avoid unnecessary import if not used
                from sentence_transformers import SentenceTransformer
                import torch

                # Load model in thread pool
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None,
                    lambda: SentenceTransformer(
                        self.hf_model_id,
                        device=self.device
                    )
                )

            except ImportError as e:
                raise ExternalServiceError(
                    service="LocalEmbeddings",
                    message="sentence-transformers not installed. Run: pip install sentence-transformers"
                ) from e
            except Exception as e:
                raise ExternalServiceError(
                    service="LocalEmbeddings",
                    message=f"Failed to load model {self.model_name}: {str(e)}"
                ) from e

    async def embed(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult: Generated embeddings with metadata

        Raises:
            ExternalServiceError: If embedding generation fails
        """
        try:
            # Ensure model is loaded
            await self._load_model()

            if not texts:
                return EmbeddingResult(
                    embeddings=[],
                    model=self.model,
                    dimensions=self.dimensions,
                    tokens_used=0
                )

            # Estimate tokens
            total_tokens = sum(self.estimate_tokens(text) for text in texts)

            # Run encoding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
            )

            # Convert to list of lists
            embedding_list = embeddings.tolist()

            return EmbeddingResult(
                embeddings=embedding_list,
                model=self.model,
                dimensions=self.dimensions,
                tokens_used=total_tokens
            )

        except Exception as e:
            raise ExternalServiceError(
                service="LocalEmbeddings",
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

    def get_batch_size(self) -> int:
        """
        Get the recommended batch size for embedding generation.

        Returns:
            int: Recommended batch size (larger for local models)
        """
        return 64  # Local models can handle larger batches


@lru_cache()
def get_local_embedding_service(model: str = "bge-m3-v2-zh") -> LocalEmbeddingService:
    """
    Get a cached local embedding service instance.

    Args:
        model: Model name to use

    Returns:
        LocalEmbeddingService: Cached service instance
    """
    return LocalEmbeddingService(model=model)
