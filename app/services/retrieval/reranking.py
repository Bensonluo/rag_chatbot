"""
Result reranking service for improving search results.

Provides LLM-based and no-op reranking strategies to optimize
search result ordering based on query relevance and diversity.
"""

import json

from app.services.llm.base import LLMMessage, LLMServiceBase
from app.services.retrieval.vector_base import (
    SearchResult,
    VectorSearchRequest,
)


def _result_id(result: SearchResult) -> str:
    """Chunk identity for ranking; the parent document ID remains source provenance."""
    return str((result.metadata or {}).get("chunk_id") or result.document_id)


class NoOpReranker:
    """
    No-op reranker that passes results through unchanged.

    Useful for disabling reranking while maintaining the same interface.
    """

    async def rerank(
        self,
        results: list[SearchResult],
        request: VectorSearchRequest,  # noqa: ARG002  # Reranker interface conformance (no-op)
    ) -> list[SearchResult]:
        """
        Return results unchanged (no reranking).

        Args:
            results: Search results to rerank
            request: Original search request

        Returns:
            List[SearchResult]: Same results (unchanged)
        """
        return results


class RerankingService:
    """
    LLM-based reranking service.

    Uses LLM to rerank search results based on query relevance
    and result diversity.
    """

    def __init__(
        self,
        llm_service: LLMServiceBase,
        top_n: int = 5,
    ) -> None:
        """
        Initialize reranking service.

        Args:
            llm_service: LLM service for reranking
            top_n: Number of top results to return
        """
        self.llm_service = llm_service
        self.top_n = top_n

    async def rerank(
        self,
        results: list[SearchResult],
        request: VectorSearchRequest,
    ) -> list[SearchResult]:
        """
        Rerank search results using LLM.

        Args:
            results: Search results to rerank
            request: Original search request

        Returns:
            List[SearchResult]: Reranked results (limited to top_n)
        """
        if not results:
            return []

        # Build reranking prompt
        prompt = self._build_reranking_prompt(
            query=request.query,
            results=results,
        )

        # Call LLM
        response = await self.llm_service.generate(
            messages=[
                LLMMessage(
                    role="system",
                    content="You are a search result reranking system. "
                    "Return JSON with reranked result IDs and optionally scores.",
                ),
                LLMMessage(role="user", content=prompt),
            ],
            max_tokens=500,
            temperature=0.0,  # Deterministic
        )

        # Parse LLM response
        return self._parse_reranking_response(
            response.content,
            original_results=results,
        )

    def _build_reranking_prompt(
        self,
        query: str,
        results: list[SearchResult],
    ) -> str:
        """
        Build prompt for LLM reranking.

        Args:
            query: Search query
            results: Search results

        Returns:
            str: Reranking prompt
        """
        # Format results for prompt
        result_docs = []
        for i, result in enumerate(results, start=1):
            result_docs.append(
                f"{i}. Result ID: {_result_id(result)}\n"
                f"   Source document ID: {result.document_id}\n"
                f"   Content: {result.content}\n"
                f"   Score: {result.score:.3f}\n"
            )

        results_text = "\n".join(result_docs)

        prompt = f"""Given this search query:

"{query}"

And these search results (ranked by initial similarity score):

{results_text}

Please rerank the results based on:
1. Relevance to the query
2. Information quality
3. Diversity of content

Use the Result IDs above (each identifies a separate chunk). Return JSON with this format:
{{"reranked": ["result_id1", "result_id2", "result_id3", ...]}}

Or if you want to provide new relevance scores:
{{"results": [{{"id": "result_id1", "score": 0.95}}, {{"id": "result_id2", "score": 0.85}}, ...]}}

Only rerank the top {self.top_n} results."""

        return prompt

    def _parse_reranking_response(
        self,
        response: str,
        original_results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Parse LLM reranking response.

        Args:
            response: LLM response text
            original_results: Original search results

        Returns:
            List[SearchResult]: Reranked results
        """
        try:
            data = json.loads(response.strip())
            if not isinstance(data, dict):
                return original_results

            result_map = {_result_id(result): result for result in original_results}
            selections: list[tuple[str, float | None]]
            if "reranked" in data:
                selections = [(identifier, None) for identifier in data["reranked"]]
            elif "results" in data:
                selections = [
                    (item["id"], float(item["score"]) if "score" in item else None)
                    for item in data["results"]
                ]
            else:
                return original_results

            reranked: list[SearchResult] = []
            seen: set[str] = set()
            for identifier, score in selections:
                if (
                    not isinstance(identifier, str)
                    or identifier not in result_map
                    or identifier in seen
                ):
                    continue
                seen.add(identifier)
                result = result_map[identifier]
                metadata = {**(result.metadata or {}), "reranked": True}
                if score is not None:
                    metadata["reranking_score"] = score
                reranked.append(
                    SearchResult(
                        document_id=result.document_id,
                        content=result.content,
                        score=result.score if score is None else score,
                        metadata=metadata,
                    )
                )

            # Both response formats may return a subset. Keep other evidence in
            # its original order without collapsing sibling chunks or duplicating IDs.
            for result in original_results:
                identifier = _result_id(result)
                if identifier not in seen:
                    reranked.append(result)
                    seen.add(identifier)
            return reranked[: self.top_n]
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return original_results
