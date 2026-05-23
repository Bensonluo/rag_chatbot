"""
Retrieval services package.

Exports all retrieval components including vector clients,
hybrid search, reranking, metadata services, and factory.
"""
from app.services.retrieval.vector_base import (
    Document,
    SearchResult,
    VectorSearchRequest,
    VectorClient,
    VectorClientError,
)
from app.services.retrieval.qdrant_client import QdrantClient
from app.services.retrieval.hybrid_search import HybridSearchService, KeywordSearch
from app.services.retrieval.reranking import RerankingService, NoOpReranker
from app.services.retrieval.cross_encoder_reranker import CrossEncoderReranker
from app.services.retrieval.chained_reranker import ChainedReranker
from app.services.retrieval.document_metadata import (
    DocumentMetadata,
    DocumentMetadataService,
    DocumentMetadataRepository,
)
from app.services.retrieval.factory import RetrievalFactory

__all__ = [
    # Base classes and models
    "Document",
    "SearchResult",
    "VectorSearchRequest",
    "VectorClient",
    "VectorClientError",
    # Vector client implementations
    "QdrantClient",
    # Hybrid search
    "HybridSearchService",
    "KeywordSearch",
    # Reranking
    "RerankingService",
    "NoOpReranker",
    "CrossEncoderReranker",
    "ChainedReranker",
    # Metadata
    "DocumentMetadata",
    "DocumentMetadataService",
    "DocumentMetadataRepository",
    # Factory
    "RetrievalFactory",
]
