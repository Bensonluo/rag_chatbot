"""Graph evidence must surface its source documents in response citations."""

from app.services.dialogue.nodes import _extract_sources, _search_result_to_dict
from app.services.graph.base import GraphSearchResult
from app.services.retrieval.vector_base import SearchResult


def test_graph_result_preserves_all_source_document_citations() -> None:
    metadata = {"source_document_ids": ["policy", "exceptions"], "source_chunk_ids": ["chunk"]}
    graph = GraphSearchResult("Refund rule", [], [], 0.9, "graph_embedding", metadata)

    converted = _search_result_to_dict(graph)

    assert converted["document_id"] == "policy"
    assert _extract_sources([converted]) == ["policy", "exceptions"]
    assert "source_type" not in metadata


def test_fused_graph_citations_use_source_documents_instead_of_synthetic_result_id() -> None:
    graph = SearchResult(
        "graph_digest",
        "Rule",
        0.9,
        {"source_type": "graph_embedding", "source_document_ids": ["policy", "exceptions"]},
    )
    vector = SearchResult("policy", "Policy", 0.8, {"chunk_id": "chunk"})

    assert _extract_sources([_search_result_to_dict(graph), _search_result_to_dict(vector)]) == [
        "policy",
        "exceptions",
    ]


def test_single_graph_source_and_legacy_document_citations_are_retained() -> None:
    docs = [
        {"metadata": {"source_document_id": "policy"}},
        {"document_id": "legacy", "metadata": None},
        {"metadata": {"source_document_ids": "policy"}},
    ]

    assert _extract_sources(docs) == ["policy", "legacy"]
