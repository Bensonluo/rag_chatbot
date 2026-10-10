"""Fusion retains chunk identity and graph source provenance."""

import hashlib

from app.services.graph.base import GraphSearchResult
from app.services.graph.retrieval.multi_path_fusion import MultiPathRetrievalFusion
from app.services.retrieval.vector_base import SearchResult


def test_fusion_preserves_multiple_chunks_from_the_same_source_document() -> None:
    originals = [
        SearchResult("policy", "Refunds allowed.", 0.9, {"chunk_id": "refund", "page_start": 1}),
        SearchResult(
            "policy", "Opened items excluded.", 0.8, {"chunk_id": "exception", "page_start": 2}
        ),
    ]
    metadata = [dict(result.metadata) for result in originals]

    fused = MultiPathRetrievalFusion().fuse(originals, [], top_k=5)

    assert len(fused) == 2
    assert [result.document_id for result in fused] == ["policy", "policy"]
    assert {result.metadata["chunk_id"] for result in fused} == {"refund", "exception"}
    assert [result.metadata for result in originals] == metadata


def test_graph_keys_cover_full_content_and_preserve_source_metadata() -> None:
    prefix = "Shared policy context. " * 20
    graph_results = [
        GraphSearchResult(
            content=prefix + suffix,
            entities=[],
            relations=[],
            score=0.8,
            source_type="graph_embedding",
            metadata={"source_document_ids": ["policy"], "source_chunk_ids": [suffix]},
        )
        for suffix in ("allowed", "excluded")
    ]
    metadata = [dict(result.metadata) for result in graph_results]

    fused = MultiPathRetrievalFusion().fuse([], graph_results, top_k=5)

    assert len(fused) == 2
    assert {result.document_id for result in fused} == {
        "graph_" + hashlib.sha256(result.content.encode("utf-8")).hexdigest()
        for result in graph_results
    }
    assert all(result.metadata["source_document_ids"] == ["policy"] for result in fused)
    assert {tuple(result.metadata["source_chunk_ids"]) for result in fused} == {
        ("allowed",),
        ("excluded",),
    }
    assert [result.metadata for result in graph_results] == metadata
