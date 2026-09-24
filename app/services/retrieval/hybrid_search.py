"""
Hybrid search service combining vector and keyword search.

Implements Reciprocal Rank Fusion (RRF) to combine results from
vector similarity search and keyword-based search.
"""

import logging
import re
from collections import defaultdict
from typing import Any

from app.core.exceptions import ValidationError
from app.services.retrieval.vector_base import (
    SearchResult,
    VectorClient,
    VectorClientError,
    VectorSearchRequest,
)

logger = logging.getLogger(__name__)


def _matches_metadata_filters(doc: dict[str, Any], filters: dict[str, Any]) -> bool:
    """
    Check a keyword-index doc against metadata filters.

    A doc matches when every filter key is present in its metadata and
    equals the filter value; list-valued metadata (e.g. tags) matches on
    membership. Docs without metadata never match a filtered request.
    """
    metadata = doc.get("metadata") or {}
    for key, value in filters.items():
        field = metadata.get(key)
        if isinstance(field, list):
            if value not in field:
                return False
        elif field != value:
            return False
    return True


class KeywordSearch:
    """
    Simple keyword-based search implementation.

    Uses BM25-style ranking based on term frequency.
    """

    def __init__(self) -> None:
        """Initialize keyword search."""
        self.documents: dict[str, dict[str, Any]] = {}
        self.document_terms: dict[str, set[str]] = {}

    async def add_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> None:
        """
        Add documents to keyword search index.

        Args:
            documents: List of dicts with 'id' and 'content' keys
        """
        for doc in documents:
            doc_id = doc["id"]
            content = doc["content"]
            self.documents[doc_id] = doc
            self.document_terms[doc_id] = self._extract_terms(content)

    async def rebuild(
        self,
        documents: list[dict[str, Any]],
    ) -> None:
        """
        Replace the whole index state from a document snapshot.

        Unlike add_documents (upsert only), rebuild drops ids absent
        from the snapshot — the periodic refresh relies on this to
        expire deleted chunks. State is built fully before being
        assigned, so a concurrent search never sees a partial index.

        Args:
            documents: Full corpus snapshot with 'id' and 'content'
        """
        new_documents = {doc["id"]: doc for doc in documents}
        new_terms = {
            doc_id: self._extract_terms(doc["content"]) for doc_id, doc in new_documents.items()
        }
        self.documents = new_documents
        self.document_terms = new_terms

    async def search(
        self,
        request: VectorSearchRequest,
    ) -> list[SearchResult]:
        """
        Search documents by keyword matching.

        Args:
            request: Search request

        Returns:
            List[SearchResult]: Ranked search results
        """
        # Metadata filters apply before ranking: a doc outside the filter
        # set must not compete for top_k slots even when it matches terms.
        doc_ids: list[str] = list(self.documents)
        if request.filters:
            doc_ids = [
                doc_id
                for doc_id in doc_ids
                if _matches_metadata_filters(self.documents[doc_id], request.filters)
            ]

        if not request.query.strip():
            # Return all (filtered) documents with zero score for empty query
            return [
                SearchResult(
                    document_id=doc_id,
                    content=self.documents[doc_id]["content"],
                    score=0.0,
                )
                for doc_id in doc_ids
            ]

        # Extract query terms
        query_terms = self._extract_terms(request.query)

        # Score each document
        scores = []
        for doc_id in doc_ids:
            score = self._compute_bm25_score(query_terms, self.document_terms[doc_id])
            if score > 0:
                scores.append(
                    SearchResult(
                        document_id=doc_id,
                        content=self.documents[doc_id]["content"],
                        score=score,
                    )
                )

        # Sort by score (descending)
        scores.sort(key=lambda x: x.score, reverse=True)

        # Apply top_k limit
        return scores[: request.top_k]

    def _extract_terms(self, text: str) -> set[str]:
        """
        Extract search terms from text.

        Args:
            text: Input text

        Returns:
            set[str]: Set of lowercase terms
        """
        # Convert to lowercase and extract alphanumeric terms
        terms = re.findall(r"\b\w+\b", text.lower())
        return set(terms)

    def _compute_bm25_score(
        self,
        query_terms: set[str],
        doc_terms: set[str],
    ) -> float:
        """
        Compute BM25-style score.

        Args:
            query_terms: Query term set
            doc_terms: Document term set

        Returns:
            float: BM25 score
        """
        if not query_terms or not doc_terms:
            return 0.0

        # Count matching terms
        matches = query_terms.intersection(doc_terms)

        if not matches:
            return 0.0

        # Simple score: number of matching terms / query length
        # This is a simplified BM25 (without IDF and document length normalization)
        return len(matches) / len(query_terms)


class HybridSearchService:
    """
    Hybrid search service combining vector and keyword search.

    Uses Reciprocal Rank Fusion (RRF) to combine rankings from
    vector similarity search and keyword-based search.
    """

    def __init__(
        self,
        vector_client: VectorClient,
        keyword_search: KeywordSearch,
        vector_weight: float = 0.5,
    ) -> None:
        """
        Initialize hybrid search service.

        Args:
            vector_client: Vector search client
            keyword_search: Keyword search instance
            vector_weight: Weight for vector search (0.0-1.0)

        Raises:
            ValidationError: If weights don't sum to 1.0
        """
        keyword_weight = 1.0 - vector_weight

        if not (0.0 <= vector_weight <= 1.0):
            raise ValidationError(f"vector_weight must be between 0.0 and 1.0, got {vector_weight}")

        self.vector_client = vector_client
        self.keyword_search = keyword_search
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    async def search(
        self,
        request: VectorSearchRequest,
    ) -> list[SearchResult]:
        """
        Perform hybrid search combining vector and keyword results.

        Args:
            request: Search request

        Returns:
            List[SearchResult]: Combined and reranked results

        Raises:
            VectorClientError: If both search methods fail
        """
        vector_results = []
        keyword_results = []

        # Try vector search; log and continue with keyword-only
        try:
            vector_results = await self.vector_client.search(request)
        except Exception:
            logger.warning("Vector search leg failed", exc_info=True)
            vector_results = []

        # Try keyword search; log and continue with vector-only
        try:
            keyword_results = await self.keyword_search.search(request)
        except Exception:
            logger.warning("Keyword search leg failed", exc_info=True)
            keyword_results = []

        # If both failed, raise error
        if not vector_results and not keyword_results:
            raise VectorClientError("Both vector and keyword search failed")

        # Combine results using RRF
        combined = self._reciprocal_rank_fusion(
            vector_results,
            keyword_results,
        )

        # Apply top_k limit
        return combined[: request.top_k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[SearchResult],
        keyword_results: list[SearchResult],
        k: int = 60,
    ) -> list[SearchResult]:
        """
        Combine rankings using Reciprocal Rank Fusion (RRF).

        RRF formula: score = sum(1 / (k + rank))

        Args:
            vector_results: Vector search results
            keyword_results: Keyword search results
            k: RRF constant (default: 60)

        Returns:
            List[SearchResult]: Fused and reranked results
        """
        # Accumulate RRF scores
        scores: dict[str, float] = defaultdict(float)
        doc_data: dict[str, dict[str, Any]] = {}

        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result.document_id
            rrf_score = 1.0 / (k + rank)
            scores[doc_id] += self.vector_weight * rrf_score

            # Store document data
            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    "content": result.content,
                    "metadata": result.metadata or {},
                    "vector_score": result.score,
                    "keyword_score": 0.0,
                }

        # Process keyword results
        for rank, result in enumerate(keyword_results, start=1):
            doc_id = result.document_id
            rrf_score = 1.0 / (k + rank)
            scores[doc_id] += self.keyword_weight * rrf_score

            # Store or update document data
            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    "content": result.content,
                    "metadata": result.metadata or {},
                    "vector_score": 0.0,
                    "keyword_score": result.score,
                }
            else:
                doc_data[doc_id]["keyword_score"] = result.score

        # Build final results
        fused_results = []
        for doc_id, score in scores.items():
            data = doc_data[doc_id]

            # Store individual scores in metadata
            metadata = data["metadata"].copy() if data["metadata"] else {}
            metadata["vector_score"] = data["vector_score"]
            metadata["keyword_score"] = data["keyword_score"]
            metadata["rrf_score"] = score

            result = SearchResult(
                document_id=doc_id,
                content=data["content"],
                score=score,
                metadata=metadata,
            )
            fused_results.append(result)

        # Sort by RRF score (descending)
        fused_results.sort(key=lambda x: x.score, reverse=True)

        return fused_results
