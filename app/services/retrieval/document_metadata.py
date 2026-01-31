"""
Document metadata service for managing document metadata.

Provides service and repository for storing and retrieving
document metadata (title, author, category, tags, etc.).
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from sqlalchemy import select, String, DateTime, Text
from sqlalchemy.orm import mapped_column, Mapped

from app.repositories.base import BaseRepository


@dataclass
class DocumentMetadata:
    """
    Document metadata dataclass.

    Attributes:
        id: Database ID (optional for new records)
        document_id: Unique document identifier
        title: Document title
        author: Document author
        category: Document category
        tags: List of tags
        created_at: Creation timestamp
        updated_at: Last update timestamp
        source: Source URL or reference
        language: Document language
    """
    id: Optional[int] = None
    document_id: str = ""
    title: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[list[str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    source: Optional[str] = None
    language: Optional[str] = None


class DocumentMetadataModel:
    """
    Document metadata database model.

    Note: This is a simple model for demonstration.
    In production, this would extend SQLAlchemy Base.
    """

    def __init__(
        self,
        document_id: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[list[str]] = None,
        source: Optional[str] = None,
        language: Optional[str] = None,
        id: Optional[int] = None,
    ):
        self.id = id
        self.document_id = document_id
        self.title = title
        self.author = author
        self.category = category
        self.tags = tags or []
        self.source = source
        self.language = language
        self.created_at = datetime.now()
        self.updated_at = datetime.now()


class DocumentMetadataRepository(BaseRepository):
    """
    Repository for document metadata operations.

    Note: This is a simplified implementation for demonstration.
    In production, would use actual SQLAlchemy models.
    """

    def __init__(self, session) -> None:
        """
        Initialize metadata repository.

        Args:
            session: Database session
        """
        self.session = session
        self._storage: dict[str, DocumentMetadata] = {}

    async def create(
        self,
        metadata: DocumentMetadataModel,
    ) -> DocumentMetadataModel:
        """
        Create metadata record.

        Args:
            metadata: Metadata model

        Returns:
            DocumentMetadataModel: Created metadata
        """
        # In-memory storage for demonstration
        metadata_dict = DocumentMetadata(
            id=len(self._storage) + 1,
            document_id=metadata.document_id,
            title=metadata.title,
            author=metadata.author,
            category=metadata.category,
            tags=metadata.tags,
            source=metadata.source,
            language=metadata.language,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        self._storage[metadata.document_id] = metadata_dict

        return metadata

    async def get_by_document_id(
        self,
        document_id: str,
    ) -> Optional[DocumentMetadata]:
        """
        Get metadata by document ID.

        Args:
            document_id: Document identifier

        Returns:
            DocumentMetadata | None: Metadata if found
        """
        return self._storage.get(document_id)

    async def get_by_document_ids(
        self,
        document_ids: list[str],
    ) -> dict[str, DocumentMetadata]:
        """
        Get metadata for multiple documents.

        Args:
            document_ids: List of document IDs

        Returns:
            dict[str, DocumentMetadata]: Mapping of document ID to metadata
        """
        result = {}
        for doc_id in document_ids:
            metadata = self._storage.get(doc_id)
            if metadata:
                result[doc_id] = metadata

        return result

    async def update(
        self,
        metadata: DocumentMetadata,
    ) -> DocumentMetadata:
        """
        Update metadata record.

        Args:
            metadata: Metadata to update

        Returns:
            DocumentMetadata: Updated metadata
        """
        metadata.updated_at = datetime.now()
        self._storage[metadata.document_id] = metadata
        return metadata

    async def delete_by_document_id(
        self,
        document_id: str,
    ) -> bool:
        """
        Delete metadata by document ID.

        Args:
            document_id: Document identifier

        Returns:
            bool: True if deleted, False if not found
        """
        if document_id in self._storage:
            del self._storage[document_id]
            return True
        return False

    async def get_by_category(
        self,
        category: str,
    ) -> list[DocumentMetadata]:
        """
        Get metadata by category.

        Args:
            category: Category name

        Returns:
            list[DocumentMetadata]: List of metadata with matching category
        """
        return [
            m for m in self._storage.values()
            if m.category == category
        ]

    async def get_by_tags(
        self,
        tags: list[str],
    ) -> list[DocumentMetadata]:
        """
        Get metadata by tags.

        Args:
            tags: List of tags to match

        Returns:
            list[DocumentMetadata]: List of metadata with matching tags
        """
        result = []
        for metadata in self._storage.values():
            if metadata.tags:
                # Check if any tag matches
                if any(tag in metadata.tags for tag in tags):
                    result.append(metadata)

        return result


class DocumentMetadataService:
    """
    Service for managing document metadata.

    Provides high-level operations for creating, retrieving,
    updating, and deleting document metadata.
    """

    def __init__(
        self,
        repository: DocumentMetadataRepository,
    ) -> None:
        """
        Initialize metadata service.

        Args:
            repository: Metadata repository
        """
        self.repository = repository

    async def add_metadata(
        self,
        metadata: DocumentMetadata,
    ) -> DocumentMetadata:
        """
        Add metadata for a document.

        Args:
            metadata: Metadata to add

        Returns:
            DocumentMetadata: Added metadata
        """
        model = DocumentMetadataModel(
            document_id=metadata.document_id,
            title=metadata.title,
            author=metadata.author,
            category=metadata.category,
            tags=metadata.tags,
            source=metadata.source,
            language=metadata.language,
        )

        result = await self.repository.create(model)

        return DocumentMetadata(
            id=result.id,
            document_id=result.document_id,
            title=result.title,
            author=result.author,
            category=result.category,
            tags=result.tags,
            created_at=result.created_at,
            updated_at=result.updated_at,
            source=result.source,
            language=result.language,
        )

    async def get_metadata(
        self,
        document_id: str,
    ) -> Optional[DocumentMetadata]:
        """
        Get metadata for a document.

        Args:
            document_id: Document identifier

        Returns:
            DocumentMetadata | None: Metadata if found
        """
        return await self.repository.get_by_document_id(document_id)

    async def get_metadata_batch(
        self,
        document_ids: list[str],
    ) -> dict[str, DocumentMetadata]:
        """
        Get metadata for multiple documents.

        Args:
            document_ids: List of document IDs

        Returns:
            dict[str, DocumentMetadata]: Mapping of document ID to metadata
        """
        return await self.repository.get_by_document_ids(document_ids)

    async def update_metadata(
        self,
        metadata: DocumentMetadata,
    ) -> DocumentMetadata:
        """
        Update metadata for a document.

        Args:
            metadata: Metadata to update

        Returns:
            DocumentMetadata: Updated metadata
        """
        return await self.repository.update(metadata)

    async def delete_metadata(
        self,
        document_id: str,
    ) -> None:
        """
        Delete metadata for a document.

        Args:
            document_id: Document identifier
        """
        await self.repository.delete_by_document_id(document_id)

    async def search_by_category(
        self,
        category: str,
    ) -> list[DocumentMetadata]:
        """
        Search documents by category.

        Args:
            category: Category name

        Returns:
            list[DocumentMetadata]: List of matching metadata
        """
        return await self.repository.get_by_category(category)

    async def search_by_tags(
        self,
        tags: list[str],
    ) -> list[DocumentMetadata]:
        """
        Search documents by tags.

        Args:
            tags: List of tags to match

        Returns:
            list[DocumentMetadata]: List of matching metadata
        """
        return await self.repository.get_by_tags(tags)

    async def enrich_search_results(
        self,
        results: list,
    ) -> list:
        """
        Enrich search results with metadata.

        Args:
            results: List of SearchResult objects

        Returns:
            list: Enriched search results with metadata
        """
        # Extract document IDs
        doc_ids = [r.document_id for r in results]

        # Fetch metadata
        metadata_map = await self.repository.get_by_document_ids(doc_ids)

        # Enrich results
        enriched = []
        for result in results:
            metadata = metadata_map.get(result.document_id)

            # Copy result and add metadata
            enriched_result = result

            if metadata:
                # Add metadata to result
                new_metadata = result.metadata or {}
                new_metadata.update({
                    "title": metadata.title,
                    "author": metadata.author,
                    "category": metadata.category,
                    "tags": metadata.tags,
                    "source": metadata.source,
                    "language": metadata.language,
                })

                # Create new enriched result
                from app.services.retrieval.vector_base import SearchResult
                enriched_result = SearchResult(
                    document_id=result.document_id,
                    content=result.content,
                    score=result.score,
                    metadata=new_metadata,
                )

            enriched.append(enriched_result)

        return enriched
