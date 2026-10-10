"""Document graph transactions preserve independent sources and provenance."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.graph.base import GraphClientError, GraphEntity, GraphRelation
from app.services.graph.neo4j_client import Neo4jClient


@pytest.fixture
def graph():
    client = Neo4jClient()
    queries = []

    async def run(query, params):
        queries.append((query, params))
        if "RETURN node.id AS id" in query:
            # A shared entity has an existing ID, not the upload's proposed ID.
            rows = [{"id": f"stored-{params['name']}"}]
        elif "RETURN r.id AS id" in query:
            rows = [{"id": "stored-relation"}]
        elif "RETURN ids" in query:
            rows = [{"ids": ["shared", "only-this-document"]}]
        elif "RETURN size(nodes) AS count" in query:
            rows = [{"count": 1}]
        elif "RETURN size(relations) AS count" in query:
            rows = [{"count": 2}]
        else:
            rows = []
        return MagicMock(data=AsyncMock(return_value=rows))

    transaction = MagicMock(run=AsyncMock(side_effect=run))
    session = AsyncMock()
    session.__aenter__.return_value = session

    async def execute_write(operation):
        return await operation(transaction)

    session.execute_write.side_effect = execute_write
    client._driver = MagicMock(session=MagicMock(return_value=session))
    return client, queries, transaction, session


@pytest.mark.asyncio
async def test_document_graph_resolves_shared_ids_and_records_authoritative_provenance(graph):
    client, queries, _, session = graph
    entities = [
        GraphEntity(
            id="incoming-a",
            name="Refund",
            type="Policy",
            description="Refunds within seven days",
            embedding=[0.1, 0.2],
            properties={
                "source_document_id": "forged",
                "source_version_id": "forged",
                "source_chunk_ids": ["chunk-1", "chunk-1", "chunk-2"],
                "details": {"days": 7},
            },
        ),
        GraphEntity(id="incoming-b", name="Return", type="Process"),
    ]
    relations = [
        GraphRelation(
            id="incoming-rel",
            source_entity_id="incoming-a",
            target_entity_id="incoming-b",
            relation_type="GUIDES",
            properties={"source_chunk_ids": ["chunk-2"]},
        )
    ]

    result = await client.replace_document_graph("doc-a", "version-2", entities, relations)

    assert result == {"entities": 2, "relations": 1}
    session.execute_write.assert_awaited_once()
    query, params = next((q, p) for q, p in queries if p.get("name") == "Refund")
    assert "MERGE (node:Entity {name: $name, type: $type})" in query
    assert "MERGE (source:DocumentEntity" in query
    assert "node +=" not in query  # Source claims must not overwrite a shared node.
    assert params["document_id"] == "doc-a"
    assert params["version_id"] == "version-2"
    assert params["chunk_ids"] == ["chunk-1", "chunk-2"]
    assert params["embedding"] == [0.1, 0.2]
    assert json.loads(params["properties_json"]) == {"details": {"days": 7}}
    relation_query, relation_params = next((q, p) for q, p in queries if "RETURN r.id AS id" in q)
    assert relation_params["source_id"] == "stored-Refund"
    assert relation_params["target_id"] == "stored-Return"
    assert relation_params["document_id"] == "doc-a"
    assert relation_params["version_id"] == "version-2"
    assert relation_params["chunk_ids"] == ["chunk-2"]
    assert "MERGE (a)-[r:`GUIDES` {source_document_id: $document_id}]->(b)" in relation_query
    assert entities[0].properties["source_document_id"] == "forged"


@pytest.mark.asyncio
async def test_delete_removes_only_document_claims_and_keeps_shared_or_legacy_entities(graph):
    client, queries, _, session = graph
    result = await client.delete_document_graph("doc-a")

    assert result == {"entities": 1, "relations": 2}
    session.execute_write.assert_awaited_once()
    edge_query, edge_params = next(
        (q, p) for q, p in queries if "RETURN size(relations) AS count" in q
    )
    assert "r.source_document_id = $document_id" in edge_query
    assert edge_params == {"document_id": "doc-a"}
    support_query, support_params = next((q, p) for q, p in queries if "RETURN ids" in q)
    assert "DocumentEntity {source_document_id: $document_id}" in support_query
    assert support_params == {"document_id": "doc-a"}
    assert "DETACH DELETE source" in support_query
    orphan_query, orphan_params = next(
        (q, p) for q, p in queries if "RETURN size(nodes) AS count" in q
    )
    assert "node.document_managed = true" in orphan_query
    assert "AND NOT (node)--()" in orphan_query
    assert "DETACH DELETE node" not in orphan_query
    assert orphan_params["ids"] == ["shared", "only-this-document"]
    refresh = next(q for q, p in queries if "SET node.source_document_ids" in q)
    assert "collect(source) AS sources" in refresh
    assert "node.original_description" in refresh
    assert "node.original_embedding" in refresh


@pytest.mark.asyncio
async def test_empty_replacement_clears_previous_version_in_same_transaction(graph):
    client, queries, _, session = graph
    assert await client.replace_document_graph("doc-a", "v3", [], []) == {
        "entities": 0,
        "relations": 0,
    }
    session.execute_write.assert_awaited_once()
    assert any("RETURN size(relations) AS count" in query for query, _ in queries)
    assert queries[-1][1] == {"document_id": "doc-a", "version_id": "v3"}


@pytest.mark.asyncio
async def test_failed_write_aborts_transaction_instead_of_reporting_partial_success(graph):
    client, queries, transaction, _ = graph
    original_run = transaction.run.side_effect

    async def fail_entity(query, params):
        if "RETURN node.id AS id" in query:
            raise RuntimeError("neo4j write failed")
        return await original_run(query, params)

    transaction.run.side_effect = fail_entity
    with pytest.raises(GraphClientError, match="neo4j write failed"):
        await client.replace_document_graph(
            "doc-a", "v2", [GraphEntity(id="e1", name="Refund", type="Policy")], []
        )
    assert not any("SET document.version_id" in query for query, _ in queries)


@pytest.mark.asyncio
async def test_missing_entity_record_is_an_error_not_false_success(graph):
    client, _, transaction, _ = graph
    transaction.run.return_value = MagicMock(data=AsyncMock(return_value=[]))
    transaction.run.side_effect = None
    with pytest.raises(GraphClientError, match="stored ID"):
        await client.replace_document_graph(
            "doc-a", "v2", [GraphEntity(id="e1", name="Refund", type="Policy")], []
        )


@pytest.mark.asyncio
async def test_cross_document_relation_is_rejected_before_mutation(graph):
    client, _, _, session = graph
    with pytest.raises(GraphClientError, match="endpoints"):
        await client.replace_document_graph(
            "doc-a",
            "v2",
            [GraphEntity(id="e1", name="Refund", type="Policy")],
            [GraphRelation("r1", "e1", "foreign-entity", "GUIDES")],
        )
    session.execute_write.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_ids", ["chunk-1", [None], [""], [1]])
async def test_invalid_chunk_provenance_is_rejected_before_mutation(graph, chunk_ids):
    client, _, _, session = graph
    with pytest.raises(GraphClientError, match="source_chunk_ids"):
        await client.replace_document_graph(
            "doc-a",
            "v2",
            [GraphEntity("e1", "Refund", "Policy", {"source_chunk_ids": chunk_ids})],
            [],
        )
    session.execute_write.assert_not_awaited()


@pytest.mark.asyncio
async def test_embedding_search_falls_back_without_gds_and_returns_source_provenance():
    client = Neo4jClient()
    client._run_query = AsyncMock(
        side_effect=[
            RuntimeError("vector index dimensions mismatch"),
            [
                {
                    "id": "e1",
                    "name": "Refund",
                    "type": "Policy",
                    "score": 0.9,
                    "source_document_ids": ["doc-a", "doc-b"],
                    "source_version_ids": ["v1", "v3"],
                    "source_chunk_ids": ["chunk-1", "chunk-2"],
                }
            ],
        ]
    )
    results = await client.search_entities_by_embedding([0.1, 0.2], entity_types=["Policy"])

    assert results[0].metadata["source_document_ids"] == ["doc-a", "doc-b"]
    assert results[0].entities[0].properties["source_chunk_ids"] == ["chunk-1", "chunk-2"]
    query, params = client._run_query.await_args.args
    assert "gds." not in query
    assert "size(e.embedding) = size($embedding)" in query
    assert "en > 0 AND qn > 0" in query
    assert params["entity_types"] == ["Policy"]


@pytest.mark.asyncio
async def test_identity_constraint_failures_are_visible():
    client = Neo4jClient()
    client._run_query = AsyncMock(side_effect=RuntimeError("permission denied"))
    with pytest.raises(GraphClientError, match="identity constraints"):
        await client._ensure_constraints()
