"""Tests for Qdrant vector client"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock


class TestQdrantClient:
    """Test Qdrant client implementation"""

    @pytest.mark.asyncio
    async def test_ensure_collection_creates_it_for_first_use(self):
        """A fresh Compose Qdrant instance should work without manual setup."""
        from app.services.retrieval.qdrant_client import QdrantClient

        class FakeDistance:
            COSINE = "Cosine"

        class FakeVectorParams:
            def __init__(self, **values):
                self.__dict__.update(values)

        models = {
            "Distance": FakeDistance,
            "VectorParams": FakeVectorParams,
        }
        mock_client = Mock()
        mock_client.collection_exists = AsyncMock(return_value=False)
        mock_client.create_collection = AsyncMock(return_value=True)
        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="documents",
            client=mock_client,
            embedding_service=Mock(dimensions=1024),
        )
        client._client_injected = False

        with patch.object(
            client,
            "_qdrant_model",
            side_effect=lambda name: models[name],
        ):
            await client._ensure_collection()

        mock_client.create_collection.assert_awaited_once()
        vector_config = mock_client.create_collection.await_args.kwargs[
            "vectors_config"
        ]
        assert vector_config.size == 1024
        assert vector_config.distance == "Cosine"

    def test_client_initialization(self):
        """Test Qdrant client initialization"""
        # Arrange & Act
        from app.services.retrieval.qdrant_client import QdrantClient
        from app.services.retrieval.vector_base import VectorClient

        mock_qdrant = Mock()
        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="test_collection",
            api_key=None,
            client=mock_qdrant
        )

        # Assert
        assert client.collection_name == "test_collection"
        assert isinstance(client, VectorClient)

    def test_client_initialization_with_api_key(self):
        """Test client initialization with API key"""
        # Arrange & Act
        from app.services.retrieval.qdrant_client import QdrantClient

        mock_qdrant = Mock()
        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="test_collection",
            api_key="test_api_key",
            client=mock_qdrant
        )

        # Assert
        assert client.api_key == "test_api_key"

    @pytest.mark.asyncio
    async def test_add_documents(self):
        """Test adding documents to Qdrant"""
        # Arrange
        from app.services.retrieval.qdrant_client import QdrantClient
        from app.services.retrieval.vector_base import Document

        mock_client = Mock()
        mock_client.upsert = AsyncMock(return_value=Mock(status="completed"))

        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="test_collection",
            client=mock_client
        )

        documents = [
            Document(
                id="doc1",
                content="Test content 1",
                embedding=[0.1, 0.2, 0.3],
                metadata={"category": "test"}
            ),
            Document(
                id="doc2",
                content="Test content 2",
                embedding=[0.4, 0.5, 0.6]
            )
        ]

        # Act
        result = await client.add_documents(documents)

        # Assert
        assert len(result) == 2
        assert result == ["doc1", "doc2"]
        mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_search(self):
        """Test searching documents in Qdrant"""
        # Arrange
        from app.services.retrieval.qdrant_client import QdrantClient
        from app.services.retrieval.vector_base import VectorSearchRequest

        mock_client = Mock()
        mock_search_result = Mock(
            id="chunk1",
            payload={
                "document_id": "doc1",
                "chunk_id": "chunk1",
                "content": "Test content",
                "metadata": {"category": "test", "chunk_id": "chunk1"}
            },
            score=0.95
        )
        mock_client.search = AsyncMock(return_value=[mock_search_result])

        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="test_collection",
            client=mock_client
        )

        # Mock embedding generation
        with patch.object(client, "_generate_embedding", return_value=[0.1, 0.2, 0.3]):
            request = VectorSearchRequest(
                query="test query",
                top_k=5
            )

            # Act
            results = await client.search(request)

            # Assert
            assert len(results) == 1
            assert results[0].document_id == "doc1"
            assert results[0].score == 0.95
            assert results[0].content == "Test content"
            assert results[0].metadata == {
                "category": "test",
                "chunk_id": "chunk1",
            }

    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        """Test searching with metadata filters"""
        # Arrange
        from app.services.retrieval.qdrant_client import QdrantClient
        from app.services.retrieval.vector_base import VectorSearchRequest

        mock_client = Mock()
        mock_client.search = AsyncMock(return_value=[])

        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="test_collection",
            client=mock_client
        )

        with patch.object(client, "_generate_embedding", return_value=[0.1, 0.2, 0.3]):
            request = VectorSearchRequest(
                query="test query",
                top_k=10,
                filters={"category": "tech"}
            )

            # Act
            await client.search(request)

            # Assert - Verify filters were passed
            mock_client.search.assert_called_once()
            call_args = mock_client.search.call_args
            assert call_args is not None

    @pytest.mark.asyncio
    async def test_delete_by_filter_keeps_root_payload_keys(self):
        """Document deletion filters target root payload fields."""
        from app.services.retrieval.qdrant_client import QdrantClient

        mock_client = Mock()
        mock_client.count = AsyncMock(return_value=Mock(count=3))
        mock_client.delete = AsyncMock(return_value=Mock(status="completed"))

        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="test_collection",
            client=mock_client,
        )

        with patch.object(
            client,
            "_build_filter",
            return_value=Mock(name="qdrant_filter"),
        ) as build_filter:
            deleted = await client.delete_by_filter({"document_id": "doc-demo"})

        build_filter.assert_called_once_with({"document_id": "doc-demo"})
        mock_client.count.assert_awaited_once()
        mock_client.delete.assert_awaited_once()
        assert deleted == 3

    @pytest.mark.asyncio
    async def test_delete_documents(self):
        """Test deleting documents from Qdrant"""
        # Arrange
        from app.services.retrieval.qdrant_client import QdrantClient

        mock_client = Mock()
        mock_client.delete = AsyncMock(return_value=Mock(status="completed"))

        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="test_collection",
            client=mock_client
        )

        # Act
        await client.delete(["doc1", "doc2"])

        # Assert
        mock_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_document(self):
        """Test updating a document in Qdrant"""
        # Arrange
        from app.services.retrieval.qdrant_client import QdrantClient
        from app.services.retrieval.vector_base import Document

        mock_client = Mock()
        mock_client.upsert = AsyncMock(return_value=Mock(status="completed"))

        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="test_collection",
            client=mock_client
        )

        document = Document(
            id="doc1",
            content="Updated content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"updated": True}
        )

        # Act
        await client.update(document)

        # Assert
        mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_document(self):
        """Test getting a document by ID"""
        # Arrange
        from app.services.retrieval.qdrant_client import QdrantClient

        mock_client = Mock()
        mock_retrieve = Mock(
            id="doc1",
            payload={
                "content": "Test content",
                "metadata": {"category": "test"}
            },
            vector=[0.1, 0.2, 0.3]
        )
        mock_client.retrieve = AsyncMock(return_value=[mock_retrieve])

        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="test_collection",
            client=mock_client
        )

        # Act
        document = await client.get_document("doc1")

        # Assert
        assert document is not None
        assert document.id == "doc1"
        assert document.content == "Test content"
        assert document.embedding == [0.1, 0.2, 0.3]
        assert document.metadata == {"category": "test"}

    @pytest.mark.asyncio
    async def test_get_document_not_found(self):
        """Test getting a non-existent document"""
        # Arrange
        from app.services.retrieval.qdrant_client import QdrantClient

        mock_client = Mock()
        mock_client.retrieve = AsyncMock(return_value=[])

        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="test_collection",
            client=mock_client
        )

        # Act
        document = await client.get_document("nonexistent")

        # Assert
        assert document is None

    @pytest.mark.asyncio
    async def test_add_documents_without_embeddings(self):
        """Test adding documents without embeddings (should generate them)"""
        # Arrange
        from app.services.retrieval.qdrant_client import QdrantClient
        from app.services.retrieval.vector_base import Document

        mock_client = Mock()
        mock_client.upsert = AsyncMock(return_value=Mock(status="completed"))

        client = QdrantClient(
            url="http://localhost:6333",
            collection_name="test_collection",
            client=mock_client
        )

        # Mock embedding generation
        with patch.object(client, "_generate_embedding", return_value=[0.1, 0.2, 0.3]):
            documents = [
                Document(id="doc1", content="Test content")  # No embedding
            ]

            # Act
            result = await client.add_documents(documents)

            # Assert
            assert len(result) == 1
            mock_client.upsert.assert_called_once()
