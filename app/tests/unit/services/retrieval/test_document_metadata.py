"""Tests for document metadata service"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest


class TestDocumentMetadata:
    """Test DocumentMetadata dataclass"""

    def test_metadata_creation(self):
        """Test creating document metadata"""
        # Arrange
        from app.services.retrieval.document_metadata import DocumentMetadata

        # Act
        metadata = DocumentMetadata(
            document_id="doc1",
            title="Test Document",
            author="Test Author",
            category="test",
            tags=["python", "tutorial"],
            created_at=datetime.now(),
        )

        # Assert
        assert metadata.document_id == "doc1"
        assert metadata.title == "Test Document"
        assert metadata.author == "Test Author"
        assert metadata.category == "test"
        assert metadata.tags == ["python", "tutorial"]

    def test_metadata_with_optional_fields(self):
        """Test metadata with optional fields"""
        # Arrange
        from app.services.retrieval.document_metadata import DocumentMetadata

        # Act
        metadata = DocumentMetadata(document_id="doc2", title="Another Document")

        # Assert
        assert metadata.document_id == "doc2"
        assert metadata.title == "Another Document"
        assert metadata.author is None
        assert metadata.category is None
        assert metadata.tags is None


class TestDocumentMetadataService:
    """Test document metadata service"""

    def test_service_initialization(self):
        """Test metadata service initialization"""
        # Arrange
        from app.services.retrieval.document_metadata import DocumentMetadataService

        mock_repo = Mock()

        # Act
        service = DocumentMetadataService(repository=mock_repo)

        # Assert
        assert service.repository == mock_repo

    @pytest.mark.asyncio
    async def test_add_metadata(self):
        """Test adding metadata"""
        # Arrange
        from app.services.retrieval.document_metadata import (
            DocumentMetadata,
            DocumentMetadataService,
        )

        mock_repo = Mock()
        mock_repo.create = AsyncMock(return_value=Mock(id=1))

        service = DocumentMetadataService(repository=mock_repo)

        metadata = DocumentMetadata(document_id="doc1", title="Test", category="tech")

        # Act
        await service.add_metadata(metadata)

        # Assert
        mock_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_metadata(self):
        """Test retrieving metadata"""
        # Arrange
        from app.services.retrieval.document_metadata import (
            DocumentMetadata,
            DocumentMetadataService,
        )

        mock_repo = Mock()
        mock_metadata = DocumentMetadata(
            id=1, document_id="doc1", title="Test Document", category="tech"
        )
        mock_repo.get_by_document_id = AsyncMock(return_value=mock_metadata)

        service = DocumentMetadataService(repository=mock_repo)

        # Act
        result = await service.get_metadata("doc1")

        # Assert
        assert result is not None
        assert result.document_id == "doc1"
        assert result.title == "Test Document"
        assert result.category == "tech"

    @pytest.mark.asyncio
    async def test_get_metadata_not_found(self):
        """Test getting metadata for non-existent document"""
        # Arrange
        from app.services.retrieval.document_metadata import DocumentMetadataService

        mock_repo = Mock()
        mock_repo.get_by_document_id = AsyncMock(return_value=None)

        service = DocumentMetadataService(repository=mock_repo)

        # Act
        result = await service.get_metadata("nonexistent")

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_update_metadata(self):
        """Test updating metadata"""
        # Arrange
        from app.services.retrieval.document_metadata import (
            DocumentMetadata,
            DocumentMetadataService,
        )

        mock_repo = Mock()
        mock_existing = DocumentMetadata(
            id=1, document_id="doc1", title="Old Title", category="old"
        )
        mock_repo.get_by_document_id = AsyncMock(return_value=mock_existing)
        mock_repo.update = AsyncMock(return_value=mock_existing)

        service = DocumentMetadataService(repository=mock_repo)

        updated_metadata = DocumentMetadata(
            id=1, document_id="doc1", title="New Title", category="new"
        )

        # Act
        await service.update_metadata(updated_metadata)

        # Assert
        mock_repo.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_metadata(self):
        """Test deleting metadata"""
        # Arrange
        from app.services.retrieval.document_metadata import DocumentMetadataService

        mock_repo = Mock()
        mock_repo.delete_by_document_id = AsyncMock(return_value=True)

        service = DocumentMetadataService(repository=mock_repo)

        # Act
        await service.delete_metadata("doc1")

        # Assert
        mock_repo.delete_by_document_id.assert_called_once_with("doc1")

    @pytest.mark.asyncio
    async def test_search_by_category(self):
        """Test searching metadata by category"""
        # Arrange
        from app.services.retrieval.document_metadata import (
            DocumentMetadata,
            DocumentMetadataService,
        )

        mock_repo = Mock()
        mock_results = [
            DocumentMetadata(id=1, document_id="doc1", title="Doc 1", category="tech"),
            DocumentMetadata(id=2, document_id="doc2", title="Doc 2", category="tech"),
        ]
        mock_repo.get_by_category = AsyncMock(return_value=mock_results)

        service = DocumentMetadataService(repository=mock_repo)

        # Act
        results = await service.search_by_category("tech")

        # Assert
        assert len(results) == 2
        assert results[0].category == "tech"
        assert results[1].category == "tech"

    @pytest.mark.asyncio
    async def test_search_by_tags(self):
        """Test searching metadata by tags"""
        # Arrange
        from app.services.retrieval.document_metadata import (
            DocumentMetadata,
            DocumentMetadataService,
        )

        mock_repo = Mock()
        mock_results = [
            DocumentMetadata(id=1, document_id="doc1", title="Doc 1", tags=["python", "api"]),
            DocumentMetadata(id=2, document_id="doc2", title="Doc 2", tags=["python", "tutorial"]),
        ]
        mock_repo.get_by_tags = AsyncMock(return_value=mock_results)

        service = DocumentMetadataService(repository=mock_repo)

        # Act
        results = await service.search_by_tags(["python"])

        # Assert
        assert len(results) == 2
        assert "python" in (results[0].tags or [])
        assert "python" in (results[1].tags or [])

    @pytest.mark.asyncio
    async def test_get_metadata_batch(self):
        """Test getting metadata for multiple documents"""
        # Arrange
        from app.services.retrieval.document_metadata import (
            DocumentMetadata,
            DocumentMetadataService,
        )

        mock_repo = Mock()
        mock_results = {
            "doc1": DocumentMetadata(id=1, document_id="doc1", title="Doc 1"),
            "doc2": DocumentMetadata(id=2, document_id="doc2", title="Doc 2"),
        }
        mock_repo.get_by_document_ids = AsyncMock(return_value=mock_results)

        service = DocumentMetadataService(repository=mock_repo)

        # Act
        results = await service.get_metadata_batch(["doc1", "doc2"])

        # Assert
        assert len(results) == 2
        assert "doc1" in results
        assert "doc2" in results

    @pytest.mark.asyncio
    async def test_enrich_search_results(self):
        """Test enriching search results with metadata"""
        # Arrange
        from app.services.retrieval.document_metadata import (
            DocumentMetadata,
            DocumentMetadataService,
        )
        from app.services.retrieval.vector_base import SearchResult

        mock_repo = Mock()
        mock_metadata = {
            "doc1": DocumentMetadata(id=1, document_id="doc1", title="Doc 1", category="tech"),
            "doc2": DocumentMetadata(id=2, document_id="doc2", title="Doc 2", category="general"),
        }
        mock_repo.get_by_document_ids = AsyncMock(return_value=mock_metadata)

        service = DocumentMetadataService(repository=mock_repo)

        search_results = [
            SearchResult(document_id="doc1", content="Content 1", score=0.9, metadata={}),
            SearchResult(document_id="doc2", content="Content 2", score=0.8, metadata={}),
        ]

        # Act
        enriched = await service.enrich_search_results(search_results)

        # Assert
        assert len(enriched) == 2
        meta0 = enriched[0].metadata
        meta1 = enriched[1].metadata
        assert meta0 is not None
        assert meta1 is not None
        assert meta0["title"] == "Doc 1"
        assert meta0["category"] == "tech"
        assert meta1["title"] == "Doc 2"
        assert meta1["category"] == "general"

    async def test_enrich_does_not_mutate_original_metadata(self):
        """Enrichment must copy metadata dicts — results are shared
        across the retrieval pipeline and must not cross-contaminate."""
        from app.services.retrieval.document_metadata import (
            DocumentMetadata,
            DocumentMetadataService,
        )
        from app.services.retrieval.vector_base import SearchResult

        mock_repo = Mock()
        mock_repo.get_by_document_ids = AsyncMock(
            return_value={
                "doc1": DocumentMetadata(id=1, document_id="doc1", title="Doc 1", category="tech")
            }
        )
        service = DocumentMetadataService(repository=mock_repo)

        original_metadata = {"source": "kb"}
        result = SearchResult(
            document_id="doc1", content="Content", score=0.9, metadata=original_metadata
        )

        enriched = await service.enrich_search_results([result])

        # Original dict untouched; enriched copy carries old + new keys.
        assert original_metadata == {"source": "kb"}
        meta = enriched[0].metadata
        assert meta is not None
        assert meta is not original_metadata
        assert meta["source"] == "kb"
        assert meta["title"] == "Doc 1"

        # A second enrichment pass must not accumulate keys.
        twice = await service.enrich_search_results([enriched[0]])
        meta_twice = twice[0].metadata
        assert meta_twice is not None
        assert meta_twice["title"] == "Doc 1"
        assert meta_twice["source"] == "kb"
        assert meta_twice is not meta


class TestDocumentMetadataRepository:
    """Test document metadata repository"""

    @pytest.mark.asyncio
    async def test_repository_initialization(self):
        """Test repository initialization"""
        # Arrange
        from app.services.retrieval.document_metadata import DocumentMetadataRepository

        mock_session = Mock()

        # Act
        repo = DocumentMetadataRepository(session=mock_session)

        # Assert
        assert repo.session == mock_session

    @pytest.mark.asyncio
    async def test_repository_create(self):
        """Test creating metadata in repository"""
        # Arrange
        from app.services.retrieval.document_metadata import (
            DocumentMetadataModel,
            DocumentMetadataRepository,
        )

        mock_session = Mock()
        repo = DocumentMetadataRepository(session=mock_session)

        metadata_model = DocumentMetadataModel(document_id="doc1", title="Test", category="tech")

        # Act
        result = await repo.create(metadata_model)

        # Assert
        assert result == metadata_model
        stored = await repo.get_by_document_id("doc1")
        assert stored is not None
        assert stored.title == "Test"

    @pytest.mark.asyncio
    async def test_repository_get_by_document_id(self):
        """Test getting metadata by document ID"""
        # Arrange
        from app.services.retrieval.document_metadata import (
            DocumentMetadataModel,
            DocumentMetadataRepository,
        )

        mock_session = Mock()
        mock_model = DocumentMetadataModel(id=1, document_id="doc1", title="Test")
        repo = DocumentMetadataRepository(session=mock_session)
        await repo.create(mock_model)

        # Act
        result = await repo.get_by_document_id("doc1")

        # Assert
        assert result is not None
        assert result.document_id == "doc1"
