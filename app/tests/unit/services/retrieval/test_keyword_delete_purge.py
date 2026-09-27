"""Deleting a document must purge the keyword leg synchronously (review #7).

The vector deletion and the epoch flip are immediate, but the BM25 leg
only expired via the periodic rebuild — for up to one refresh interval
a deleted document stayed recallable by keyword, and an answer built
from that stale hit was cached under the NEW epoch, so its influence
outlived the index window (L0 TTL is measured in hours). The fix:
``KeywordSearch.remove_document`` plus a fail-open module registry the
ingestion path calls right after the epoch bump, so both retrieval
legs are clean before ``delete_document`` returns.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock

import app.services.retrieval.keyword_refresh as keyword_refresh_module
from app.services.retrieval.hybrid_search import HybridSearchService, KeywordSearch
from app.services.retrieval.vector_base import VectorSearchRequest

# Chunk dicts in the exact list_chunks shape: id / content / metadata.
_CHUNK_A1: dict[str, Any] = {
    "id": "a1",
    "content": "zxqalpha 退货政策第一条",
    "metadata": {"document_id": "docA"},
}
_CHUNK_A2: dict[str, Any] = {
    "id": "a2",
    "content": "zxqalpha 退货政策第二条",
    "metadata": {"document_id": "docA"},
}
_CHUNK_B1: dict[str, Any] = {
    "id": "b1",
    "content": "yxwbeta 运费说明",
    "metadata": {"document_id": "docB"},
}


def _seeded_leg() -> KeywordSearch:
    leg = KeywordSearch()
    leg.documents = {
        "a1": _CHUNK_A1,
        "a2": _CHUNK_A2,
        "b1": _CHUNK_B1,
    }
    leg.document_terms = {
        "a1": leg._extract_terms(_CHUNK_A1["content"]),
        "a2": leg._extract_terms(_CHUNK_A2["content"]),
        "b1": leg._extract_terms(_CHUNK_B1["content"]),
    }
    return leg


class TestKeywordLegPurge:
    async def test_remove_document_drops_every_chunk_of_that_document(self) -> None:
        """remove_document deletes all chunks of the document — the
        cross-store guarantee: vector gone AND keyword gone."""
        leg = _seeded_leg()

        removed = await leg.remove_document("docA")

        assert removed == 2
        hits = await leg.search(VectorSearchRequest(query="zxqalpha", top_k=5))
        assert hits == []
        # Other documents untouched.
        survivor = await leg.search(VectorSearchRequest(query="yxwbeta", top_k=5))
        assert [h.document_id for h in survivor] == ["b1"]

    async def test_remove_document_is_noop_for_unknown_id(self) -> None:
        leg = _seeded_leg()

        removed = await leg.remove_document("docMissing")

        assert removed == 0
        assert set(leg.documents) == {"a1", "a2", "b1"}


class TestPurgeHelper:
    async def test_purge_without_registration_is_silent_noop(self, monkeypatch) -> None:
        from app.services.retrieval.keyword_refresh import purge_keyword_document

        monkeypatch.setattr(keyword_refresh_module, "_keyword_hybrid", None)

        await purge_keyword_document("docA")  # must not raise

    async def test_purge_removes_chunks_from_registered_index(self, monkeypatch) -> None:
        from app.services.retrieval.keyword_refresh import purge_keyword_document

        leg = _seeded_leg()
        monkeypatch.setattr(
            keyword_refresh_module,
            "_keyword_hybrid",
            HybridSearchService(vector_client=Mock(), keyword_search=leg),
        )

        await purge_keyword_document("docA")

        hits = await leg.search(VectorSearchRequest(query="zxqalpha", top_k=5))
        assert hits == []

    async def test_purge_failure_is_fail_open(self, monkeypatch) -> None:
        """A purge failure must not fail the delete API — the periodic
        rebuild remains the backstop."""
        from app.services.retrieval.keyword_refresh import purge_keyword_document

        broken = Mock()
        broken.keyword_search.remove_document = AsyncMock(side_effect=RuntimeError("index locked"))
        monkeypatch.setattr(keyword_refresh_module, "_keyword_hybrid", broken)

        await purge_keyword_document("docA")  # must not raise


class TestIngestionWiring:
    async def test_delete_document_purges_keyword_index(self, monkeypatch) -> None:
        """The full chain: vectors deleted → epoch bumped → keyword leg
        purged, all before delete_document returns (single-instance
        guarantee; replicas converge on their next rebuild)."""
        import app.services.documents.ingestion as ingestion_module
        from app.services.documents.ingestion import DocumentIngestionService

        leg = _seeded_leg()
        hybrid = HybridSearchService(vector_client=Mock(), keyword_search=leg)
        monkeypatch.setattr(keyword_refresh_module, "_keyword_hybrid", hybrid)

        qdrant = Mock()
        qdrant.delete_by_filter = AsyncMock(return_value=2)
        bump_epoch = AsyncMock()
        monkeypatch.setattr(ingestion_module, "bump_kb_epoch", bump_epoch)
        service = DocumentIngestionService(
            qdrant_client=qdrant,
            embedding_provider="local",
            chunking_strategy="semantic",
        )

        result = await service.delete_document("docA")

        assert result["deleted_chunks"] == 2
        bump_epoch.assert_awaited_once()
        # The stale-recall window is closed before the call returns.
        hits = await leg.search(VectorSearchRequest(query="zxqalpha", top_k=5))
        assert hits == []

    async def test_delete_document_skips_purge_on_empty_delete(self, monkeypatch) -> None:
        """An empty delete (unknown id) is not a KB mutation: no epoch
        bump, no purge."""
        import app.services.documents.ingestion as ingestion_module
        from app.services.documents.ingestion import DocumentIngestionService

        leg = _seeded_leg()
        hybrid = HybridSearchService(vector_client=Mock(), keyword_search=leg)
        monkeypatch.setattr(keyword_refresh_module, "_keyword_hybrid", hybrid)

        qdrant = Mock()
        qdrant.delete_by_filter = AsyncMock(return_value=0)
        bump_epoch = AsyncMock()
        monkeypatch.setattr(ingestion_module, "bump_kb_epoch", bump_epoch)
        service = DocumentIngestionService(
            qdrant_client=qdrant,
            embedding_provider="local",
            chunking_strategy="semantic",
        )

        await service.delete_document("docA")

        bump_epoch.assert_not_awaited()
        assert set(leg.documents) == {"a1", "a2", "b1"}
