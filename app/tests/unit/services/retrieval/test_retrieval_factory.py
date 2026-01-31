"""Tests for retrieval service factory"""
import pytest
from unittest.mock import Mock


class TestRetrievalFactory:
    """Test retrieval service factory"""

    def test_create_qdrant_client(self):
        """Test creating Qdrant client"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        # Act
        client = RetrievalFactory.create_vector_client(
            client_type="qdrant",
            url="http://localhost:6333",
            collection_name="test_collection"
        )

        # Assert
        assert client.collection_name == "test_collection"

    def test_create_qdrant_client_with_api_key(self):
        """Test creating Qdrant client with API key"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        # Act
        client = RetrievalFactory.create_vector_client(
            client_type="qdrant",
            url="http://localhost:6333",
            collection_name="test_collection",
            api_key="test_api_key"
        )

        # Assert
        assert client.api_key == "test_api_key"

    def test_create_invalid_client_type(self):
        """Test creating invalid client type raises error"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory
        from app.core.exceptions import ValidationError

        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            RetrievalFactory.create_vector_client(
                client_type="invalid_type",
                url="http://localhost:6333",
                collection_name="test_collection"
            )
        assert "client" in str(exc_info.value).lower()

    def test_create_hybrid_search_service(self):
        """Test creating hybrid search service"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        mock_vector_client = Mock()
        mock_keyword_search = Mock()

        # Act
        service = RetrievalFactory.create_hybrid_search(
            vector_client=mock_vector_client,
            keyword_search=mock_keyword_search,
            vector_weight=0.7
        )

        # Assert
        assert service.vector_weight == 0.7
        assert service.keyword_weight == 0.3

    def test_create_hybrid_search_default_weights(self):
        """Test creating hybrid search with default weights"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        mock_vector_client = Mock()
        mock_keyword_search = Mock()

        # Act
        service = RetrievalFactory.create_hybrid_search(
            vector_client=mock_vector_client,
            keyword_search=mock_keyword_search
        )

        # Assert
        assert service.vector_weight == 0.5
        assert service.keyword_weight == 0.5

    def test_create_reranking_service(self):
        """Test creating reranking service"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        mock_llm = Mock()

        # Act
        reranker = RetrievalFactory.create_reranker(
            llm_service=mock_llm,
            top_n=10
        )

        # Assert
        assert reranker.top_n == 10

    def test_create_reranking_service_default_top_n(self):
        """Test creating reranker with default top_n"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        mock_llm = Mock()

        # Act
        reranker = RetrievalFactory.create_reranker(
            llm_service=mock_llm
        )

        # Assert
        assert reranker.top_n == 5

    def test_create_noop_reranker(self):
        """Test creating no-op reranker"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        # Act
        reranker = RetrievalFactory.create_reranker(
            reranker_type="noop"
        )

        # Assert
        from app.services.retrieval.reranking import NoOpReranker
        assert isinstance(reranker, NoOpReranker)

    def test_create_document_metadata_service(self):
        """Test creating document metadata service"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        mock_repo = Mock()

        # Act
        service = RetrievalFactory.create_metadata_service(
            repository=mock_repo
        )

        # Assert
        assert service.repository == mock_repo

    def test_create_retrieval_pipeline(self):
        """Test creating complete retrieval pipeline"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        mock_vector_client = Mock()
        mock_llm = Mock()
        mock_metadata_repo = Mock()

        # Act
        pipeline = RetrievalFactory.create_pipeline(
            vector_client=mock_vector_client,
            llm_service=mock_llm,
            metadata_repository=mock_metadata_repo,
            use_reranking=True,
            use_metadata_enrichment=True
        )

        # Assert
        assert pipeline["vector_client"] == mock_vector_client
        assert pipeline["reranker"] is not None
        assert pipeline["metadata_service"] is not None

    def test_create_retrieval_pipeline_without_reranking(self):
        """Test creating pipeline without reranking"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        mock_vector_client = Mock()
        mock_llm = Mock()

        # Act
        pipeline = RetrievalFactory.create_pipeline(
            vector_client=mock_vector_client,
            llm_service=mock_llm,
            use_reranking=False
        )

        # Assert
        from app.services.retrieval.reranking import NoOpReranker
        assert isinstance(pipeline["reranker"], NoOpReranker)

    def test_create_retrieval_pipeline_without_metadata(self):
        """Test creating pipeline without metadata enrichment"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        mock_vector_client = Mock()
        mock_llm = Mock()

        # Act
        pipeline = RetrievalFactory.create_pipeline(
            vector_client=mock_vector_client,
            llm_service=mock_llm,
            use_metadata_enrichment=False
        )

        # Assert
        assert pipeline["metadata_service"] is None

    def test_factory_config_validation(self):
        """Test factory configuration validation"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory
        from app.core.exceptions import ValidationError

        # Act & Assert - Invalid vector weight
        with pytest.raises(ValidationError):
            RetrievalFactory.create_hybrid_search(
                vector_client=Mock(),
                keyword_search=Mock(),
                vector_weight=1.5  # Invalid
            )

    def test_create_with_custom_reranker_config(self):
        """Test creating reranker with custom configuration"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        mock_llm = Mock()

        # Act
        reranker = RetrievalFactory.create_reranker(
            llm_service=mock_llm,
            reranker_type="llm",
            top_n=15
        )

        # Assert
        assert reranker.top_n == 15
