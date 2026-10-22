"""Graph retrieval must preserve known source sets without inventing mappings."""

from unittest.mock import AsyncMock, Mock

import pytest

from app.services.chat.citations import citations_from_documents
from app.services.dialogue.nodes import _extract_sources, _graph_doc_to_dict, _search_result_to_dict
from app.services.graph.base import GraphEntity, GraphRelation, GraphSearchResult
from app.services.graph.community.global_search import GlobalSearchService
from app.services.graph.provenance import collect_source_provenance
from app.services.graph.retrieval.graph_embedding_search import GraphEmbeddingSearch
from app.services.graph.retrieval.graph_retrieval_service import GraphRetrievalService
from app.services.graph.retrieval.multi_path_fusion import MultiPathRetrievalFusion
from app.services.graph.retrieval.text_to_cypher import TextToCypherService


async def test_embedding_neighborhood_and_vector_fusion_keep_document_revisions() -> None:
    entity = GraphEntity(
        id="entity",
        name="Warranty",
        type="Policy",
        description="Coverage policy.",
        properties={
            "source_document_ids": ["policy-a", "policy-b"],
            "source_version_ids": ["v1"],
            "source_chunk_ids": ["chunk-a"],
            "internal_token": "private",
        },
    )
    relation = GraphRelation(
        id="relation",
        source_entity_id="entity",
        target_entity_id="other",
        relation_type="COVERS",
        properties={
            "source_document_id": "policy-c",
            "source_version_id": "v2",
            "source_chunk_ids": ["chunk-c"],
        },
    )
    hit = GraphSearchResult(
        content="Coverage",
        entities=[entity],
        relations=[],
        score=0.9,
        source_type="graph_embedding",
    )
    neighbor = GraphSearchResult(
        content="Covers replacement parts.",
        entities=[],
        relations=[relation],
        score=0.8,
        source_type="graph_embedding",
    )
    graph = Mock(
        search_entities_by_embedding=AsyncMock(return_value=[hit]),
        get_entity_neighborhood=AsyncMock(return_value=[neighbor]),
    )
    results = await GraphEmbeddingSearch(
        graph, Mock(embed_single=AsyncMock(return_value=[1.0]))
    ).search("coverage")
    for doc in [
        _graph_doc_to_dict(results[0]),
        _search_result_to_dict(MultiPathRetrievalFusion().fuse([], results)[0]),
    ]:
        citation = citations_from_documents([doc])[0]
        assert citation["source_type"] == "graph"
        assert citation["source_document_ids"] == ["policy-a", "policy-b", "policy-c"]
        assert citation["source_version_ids"] == ["v1", "v2"]
        assert citation["source_chunk_ids"] == ["chunk-a", "chunk-c"]
        assert "document_id" not in citation
        assert "version_id" not in citation
        assert "private" not in str(citation)


def test_cypher_rows_preserve_nested_properties_and_explicit_source_columns() -> None:
    service = TextToCypherService(Mock(), Mock())
    results = service._to_search_results(
        [
            {
                "entity": {
                    "name": "Policy",
                    "description": "Coverage",
                    "properties": {
                        "source_document_ids": ["a", "b"],
                        "source_version_ids": ["v-a"],
                        "source_chunk_ids": ["chunk-a"],
                    },
                },
                "source_document_id": "c",
                "source_version_id": "v-c",
            }
        ],
        "private user query",
    )
    citation = citations_from_documents([_graph_doc_to_dict(results[0])])[0]
    assert citation["source_document_ids"] == ["c", "a", "b"]
    assert citation["source_version_ids"] == ["v-c", "v-a"]
    assert citation["source_chunk_ids"] == ["chunk-a"]
    assert "private user query" not in str(citation)


async def test_identical_graph_content_fuses_all_known_sources() -> None:
    cypher = GraphSearchResult(
        content="Shared fact",
        entities=[],
        relations=[],
        score=1,
        source_type="text_to_cypher",
        metadata={"source_document_id": "a", "source_version_id": "v-a"},
    )
    embedding = GraphSearchResult(
        content="Shared fact",
        entities=[],
        relations=[],
        score=1,
        source_type="graph_embedding",
        metadata={"source_document_ids": ["b", "c"], "source_version_ids": ["v-b"]},
    )
    retrieval = GraphRetrievalService(
        text_to_cypher=Mock(query=AsyncMock(return_value=[cypher])),
        graph_embedding_search=Mock(search=AsyncMock(return_value=[embedding])),
    )
    results = await retrieval.search("Shared fact")
    assert len(results) == 1
    citation = citations_from_documents([_graph_doc_to_dict(results[0])])[0]
    assert citation["source_document_ids"] == ["a", "b", "c"]
    assert citation["source_version_ids"] == ["v-a", "v-b"]


@pytest.mark.parametrize("fallback", [False, True])
async def test_community_source_fields_survive_both_query_modes(fallback: bool) -> None:
    row = {
        "cid": "community",
        "title": "Coverage",
        "summary": "Community synthesis.",
        "source_document_ids": ["policy-a", "policy-b"],
        "source_version_ids": ["v1"],
        "source_chunk_ids": ["chunk-a"],
    }
    graph = Mock(
        execute_cypher=AsyncMock(
            side_effect=[RuntimeError("index unavailable"), [row]] if fallback else [[row]]
        )
    )
    results = await GlobalSearchService(
        graph, Mock(embed_single=AsyncMock(return_value=[1.0]))
    ).search("Coverage")
    citation = citations_from_documents([_graph_doc_to_dict(results[0])])[0]
    assert citation["source_document_ids"] == row["source_document_ids"]
    assert citation["source_version_ids"] == ["v1"]
    assert citation["source_chunk_ids"] == ["chunk-a"]
    assert "source_version_ids" in graph.execute_cypher.await_args.args[0]
    assert "document_id" not in citation and "version_id" not in citation


async def test_unversioned_community_does_not_borrow_current_entity_sources() -> None:
    graph = Mock(
        execute_cypher=AsyncMock(
            return_value=[{"cid": "old", "summary": "Old summary", "entities": ["Policy"]}]
        )
    )
    results = await GlobalSearchService(
        graph, Mock(embed_single=AsyncMock(return_value=[1.0]))
    ).search("Policy")
    citation = citations_from_documents([_graph_doc_to_dict(results[0])])[0]
    assert citation["source_type"] == "graph"
    assert citation["source_document_ids"] == []
    assert citation["source_version_ids"] == []
    graph.execute_cypher.assert_awaited_once()
    fused = MultiPathRetrievalFusion().fuse([], results)
    # A graph result hash is not a real source ID, even on the legacy surface.
    assert _extract_sources([_search_result_to_dict(fused[0])]) == []


def test_provenance_extraction_is_bounded_and_ignores_other_properties() -> None:
    sources = collect_source_provenance(
        {
            "source_document_ids": ["x" * 513, "known"],
            "query": "private query",
            "source_version_ids": [f"v-{i}" for i in range(100)],
            "properties": {"internal_token": "private token"},
        }
    )
    assert sources["source_document_ids"] == ["known"]
    assert len(sources["source_version_ids"]) == 50
    assert "private" not in str(sources)
