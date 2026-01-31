"""
Factory for creating retrieval service components.

Provides simple interface for creating vector clients, hybrid search,
reranking services, and complete retrieval pipelines.
"""
from typing import Optional

from app.services.retrieval.vector_base import VectorClient
from app.services.retrieval.qdrant_client import QdrantClient
from app.services.retrieval.hybrid_search import HybridSearchService, KeywordSearch
from app.services.retrieval.reranking import RerankingService, NoOpReranker
from app.services.retrieval.document_metadata import DocumentMetadataService, DocumentMetadataRepository
from app.services.llm.base import LLMServiceBase
from app.core.exceptions import ValidationError


class RetrievalFactory:
    """
    Factory for creating retrieval service components.

    Provides methods for creating vector clients, hybrid search services,
    rerankers, metadata services, and complete retrieval pipelines.
    """

    @staticmethod
    def create_vector_client(
        client_type: str,
        url: str,
        collection_name: str,
        api_key: Optional[str] = None,
        embedding_service=None,
        **kwargs,
    ) -> VectorClient:
        """
        Create a vector database client.

        Args:
            client_type: Type of client ("qdrant")
            url: Vector database URL
            collection_name: Collection/table name
            api_key: Optional API key
            embedding_service: Optional embedding service for generating embeddings
            **kwargs: Additional client-specific parameters

        Returns:
            VectorClient: Configured vector client

        Raises:
            ValidationError: If client_type is invalid
        """
        if client_type == "qdrant":
            return QdrantClient(
                url=url,
                collection_name=collection_name,
                api_key=api_key,
                embedding_service=embedding_service,
                **kwargs,
            )
        else:
            valid_types = ["qdrant"]
            raise ValidationError(
                f"Invalid client_type: {client_type}. "
                f"Must be one of {valid_types}"
            )

    @staticmethod
    def create_hybrid_search(
        vector_client: VectorClient,
        vector_weight: float = 0.5,
        keyword_search: Optional[KeywordSearch] = None,
    ) -> HybridSearchService:
        """
        Create a hybrid search service.

        Args:
            vector_client: Vector search client
            vector_weight: Weight for vector search (0.0-1.0)
            keyword_search: Optional keyword search instance

        Returns:
            HybridSearchService: Configured hybrid search service

        Raises:
            ValidationError: If weights are invalid
        """
        # Create keyword search if not provided
        if keyword_search is None:
            keyword_search = KeywordSearch()

        # Validate weights
        if not (0.0 <= vector_weight <= 1.0):
            raise ValidationError(
                f"vector_weight must be between 0.0 and 1.0, got {vector_weight}"
            )

        return HybridSearchService(
            vector_client=vector_client,
            keyword_search=keyword_search,
            vector_weight=vector_weight,
        )

    @staticmethod
    def create_reranker(
        reranker_type: str = "llm",
        llm_service: Optional[LLMServiceBase] = None,
        top_n: int = 5,
    ) -> object:
        """
        Create a reranking service.

        Args:
            reranker_type: Type of reranker ("llm" or "noop")
            llm_service: LLM service (required for "llm" type)
            top_n: Number of top results to return

        Returns:
            Reranking service instance

        Raises:
            ValidationError: If reranker_type is invalid or LLM not provided
        """
        if reranker_type == "noop":
            return NoOpReranker()

        elif reranker_type == "llm":
            if llm_service is None:
                raise ValidationError(
                    "llm_service is required for LLM-based reranking"
                )

            return RerankingService(
                llm_service=llm_service,
                top_n=top_n,
            )

        else:
            valid_types = ["llm", "noop"]
            raise ValidationError(
                f"Invalid reranker_type: {reranker_type}. "
                f"Must be one of {valid_types}"
            )

    @staticmethod
    def create_metadata_service(
        repository: DocumentMetadataRepository,
    ) -> DocumentMetadataService:
        """
        Create a document metadata service.

        Args:
            repository: Metadata repository

        Returns:
            DocumentMetadataService: Metadata service instance
        """
        return DocumentMetadataService(repository=repository)

    @staticmethod
    def create_pipeline(
        vector_client: VectorClient,
        llm_service: Optional[LLMServiceBase] = None,
        metadata_repository: Optional[DocumentMetadataRepository] = None,
        use_reranking: bool = True,
        use_metadata_enrichment: bool = True,
        vector_weight: float = 0.5,
        reranker_top_n: int = 5,
    ) -> dict:
        """
        Create a complete retrieval pipeline.

        Args:
            vector_client: Vector search client
            llm_service: Optional LLM service (for reranking)
            metadata_repository: Optional metadata repository
            use_reranking: Whether to use LLM reranking
            use_metadata_enrichment: Whether to use metadata enrichment
            vector_weight: Weight for vector search in hybrid (0.0-1.0)
            reranker_top_n: Number of results to return after reranking

        Returns:
            dict: Pipeline components
                - "vector_client": Vector client
                - "hybrid_search": Hybrid search service
                - "reranker": Reranking service
                - "metadata_service": Metadata service (optional)
        """
        # Create hybrid search
        hybrid_search = RetrievalFactory.create_hybrid_search(
            vector_client=vector_client,
            vector_weight=vector_weight,
        )

        # Create reranker
        if use_reranking and llm_service:
            reranker = RetrievalFactory.create_reranker(
                reranker_type="llm",
                llm_service=llm_service,
                top_n=reranker_top_n,
            )
        else:
            reranker = RetrievalFactory.create_reranker(reranker_type="noop")

        # Create metadata service
        metadata_service = None
        if use_metadata_enrichment and metadata_repository:
            metadata_service = RetrievalFactory.create_metadata_service(
                repository=metadata_repository,
            )

        return {
            "vector_client": vector_client,
            "hybrid_search": hybrid_search,
            "reranker": reranker,
            "metadata_service": metadata_service,
        }
