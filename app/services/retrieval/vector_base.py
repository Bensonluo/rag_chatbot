"""
Vector service base interface and data models.

Provides abstract interface for vector database operations and common data models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any


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
    metadata: dict[str, Any] | None = None


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
    metadata: dict[str, Any] | None = None


@dataclass
class VectorSearchRequest:
    """
    Request for vector similarity search.

    Attributes:
        query: Search query text
        top_k: Number of results to return (default: 5)
        filters: Optional metadata filters
        acl_filters: Server-injected retrieval ACL scope (review
            2026-09-26 #1; docs/design/per-user-architecture.md D5).
            A separate channel from business ``filters``: built only
            from server-side context by ``retrieval_acl_scope()``,
            never from the request body or slot extraction, and never
            narrowed by the FILTERABLE_METADATA_KEYS whitelist (that
            whitelist is the business-filter contract).
    """

    query: str
    top_k: int = 5
    filters: dict[str, Any] | None = None
    acl_filters: dict[str, Any] | None = None

    def with_merged_filters(self) -> VectorSearchRequest:
        """Copy with the ACL scope folded into ``filters`` (ACL wins).

        The single enforcement point: HybridSearchService.search calls
        this before dispatching to the vector and keyword legs, so no
        leg can be asked to search without the caller's scope applied —
        a colliding business key (even a forged one) cannot loosen it.
        Identity when there is no scope, so today's all-public KB
        produces byte-identical requests.
        """
        if not self.acl_filters:
            return self
        return replace(
            self,
            filters={**(self.filters or {}), **self.acl_filters},
            acl_filters=None,
        )


# Chunk-metadata filter contract: payload keys under "metadata." that
# search-time filters may reference. Slot types outside this set (order_id,
# reason, issue, ...) have no payload counterpart — filtering on them would
# match an empty set and silently zero out retrieval.
FILTERABLE_METADATA_KEYS = frozenset(
    {"product", "category", "doc_type", "source", "language", "tags"}
)


def intersect_metadata_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    """Keep only filters whose keys are part of the chunk-metadata contract."""
    return {k: v for k, v in (filters or {}).items() if k in FILTERABLE_METADATA_KEYS}


def retrieval_acl_scope() -> dict[str, Any] | None:
    """The calling user's retrieval ACL scope, server-side only (review
    2026-09-26 #1; docs/design/per-user-architecture.md D5).

    The return value must be constructed exclusively from server-side
    identity context — never from the request body or slot extraction —
    and travels on ``VectorSearchRequest.acl_filters``, a channel the
    business-filter machinery (whitelist, filter-miss fallback) cannot
    strip: the fallback drops business filters but carries this channel
    unchanged, and both hybrid legs enforce the merged scope.

    Today every KB document is public (no ownership columns exist), so
    every caller's scope is the empty constraint. Private documents
    (design P2) replace this constant with a per-caller expression
    (visible = public OR owned-by-caller); the invariants pinned in
    app/tests/unit/services/dialogue/test_acl_seam.py do not change.
    """
    return None


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

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        """
        Initialize vector client error.

        Args:
            message: Error message
            details: Optional error details
        """
        self.message = message
        self.details = details
        super().__init__(self.message)
