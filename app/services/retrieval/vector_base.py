"""
Vector service base interface and data models.

Provides abstract interface for vector database operations and common data models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Document:
    """
    Document for vector storage.

    Attributes:
        id: Unique document identifier
        content: Document text content
        embedding: Optional vector embedding
        metadata: Optional document metadata
    """

    id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict | None = None


@dataclass
class SearchResult:
    """
    Result from vector similarity search.

    Attributes:
        document_id: Document identifier
        content: Document content
        score: Similarity score (0-1, higher is better)
        metadata: Optional document metadata
    """

    document_id: str
    content: str
    score: float
    metadata: dict | None = None


@dataclass
class VectorSearchRequest:
    """
    Request for vector similarity search.

    Attributes:
        query: Search query text
        top_k: Number of results to return (default: 5)
        filters: Optional metadata filters
    """

    query: str
    top_k: int = 5
    filters: dict | None = None


class VectorClient(ABC):
    """
    Abstract base class for vector database clients.

    Defines the interface for vector storage and retrieval operations.
    """

    @abstractmethod
    async def add_documents(
        self,
        documents: list[Document],
    ) -> list[str]:
        """
        Add documents to vector database.

        Args:
            documents: List of documents to add

        Returns:
            List[str]: List of document IDs

        Raises:
            VectorClientError: If operation fails
        """
        pass

    @abstractmethod
    async def search(
        self,
        request: VectorSearchRequest,
    ) -> list[SearchResult]:
        """
        Search for similar documents.

        Args:
            request: Search request with query and parameters

        Returns:
            List[SearchResult]: List of search results sorted by score

        Raises:
            VectorClientError: If operation fails
        """
        pass

    @abstractmethod
    async def delete(
        self,
        document_ids: list[str],
    ) -> None:
        """
        Delete documents from vector database.

        Args:
            document_ids: List of document IDs to delete

        Raises:
            VectorClientError: If operation fails
        """
        pass

    @abstractmethod
    async def update(
        self,
        document: Document,
    ) -> None:
        """
        Update a document in vector database.

        Args:
            document: Document with updated content

        Raises:
            VectorClientError: If operation fails
        """
        pass

    @abstractmethod
    async def get_document(
        self,
        document_id: str,
    ) -> Document | None:
        """
        Get a document by ID.

        Args:
            document_id: Document identifier

        Returns:
            Document | None: Document if found, None otherwise

        Raises:
            VectorClientError: If operation fails
        """
        pass


class VectorClientError(Exception):
    """Base exception for vector client errors."""

    def __init__(self, message: str, details: dict | None = None):
        """
        Initialize vector client error.

        Args:
            message: Error message
            details: Optional error details
        """
        self.message = message
        self.details = details
        super().__init__(self.message)
