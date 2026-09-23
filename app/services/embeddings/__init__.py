"""
Embedding services package.

Exports all embedding-related components including providers and factory.
"""

from app.services.embeddings.base import EmbeddingResult, EmbeddingServiceBase
from app.services.embeddings.cached_embeddings import CachedEmbeddingService
from app.services.embeddings.factory import EmbeddingFactory
from app.services.embeddings.glm_embeddings import GLMEmbeddingService
from app.services.embeddings.local_embeddings import LocalEmbeddingService
from app.services.embeddings.openai_embeddings import OpenAIEmbeddingService

__all__ = [
    # Base classes
    "EmbeddingServiceBase",
    "EmbeddingResult",
    # Providers
    "LocalEmbeddingService",
    "GLMEmbeddingService",
    "OpenAIEmbeddingService",
    # Caching
    "CachedEmbeddingService",
    # Factory
    "EmbeddingFactory",
]
