"""Regression tests for the document ingestion demo flow."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.services.documents.ingestion import DocumentIngestionService
from app.services.embeddings.base import EmbeddingResult


@pytest.mark.asyncio
async def test_ingest_text_chunks_embeds_stores_and_returns_summary() -> None:
    """A text upload should complete the full demo ingestion journey."""
    qdrant_client = Mock()
    qdrant_client.add = AsyncMock()

    embedding_service = Mock()
    embedding_service.embed = AsyncMock(
        return_value=EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            model="demo-embedding",
            dimensions=3,
            tokens_used=12,
        )
    )

    with patch(
        "app.services.documents.ingestion.EmbeddingFactory.create",
        return_value=embedding_service,
    ):
        service = DocumentIngestionService(
            qdrant_client=qdrant_client,
            chunking_strategy="semantic",
        )

    result = await service.ingest_text(
        text="退款政策支持七天内申请。",
        title="退款政策",
        metadata={"category": "policy"},
        document_id="doc-demo",
    )

    assert result["document_id"] == "doc-demo"
    assert result["chunks_count"] == 1
    assert result["embedding_model"] == "demo-embedding"
    qdrant_client.add.assert_awaited_once()
    payload = qdrant_client.add.await_args.kwargs["payloads"][0]
    assert payload["document_id"] == "doc-demo"
    assert payload["chunk_id"]
    assert payload["metadata"]["category"] == "policy"
    assert payload["metadata"]["chunk_id"] == payload["chunk_id"]


@pytest.mark.asyncio
async def test_delete_document_uses_root_document_id_filter() -> None:
    """Ingested payloads store document_id at the root, not under metadata."""
    qdrant_client = Mock()
    qdrant_client.delete_by_filter = AsyncMock(return_value=2)

    with patch("app.services.documents.ingestion.EmbeddingFactory.create"):
        service = DocumentIngestionService(qdrant_client=qdrant_client)

    result = await service.delete_document("doc-demo")

    qdrant_client.delete_by_filter.assert_awaited_once_with(
        {"document_id": "doc-demo"}
    )
    assert result == {"document_id": "doc-demo", "deleted_chunks": 2}


def test_ingestion_reuses_an_injected_embedding_service() -> None:
    """The API should not create another heavyweight local model per upload."""
    embedding_service = Mock()

    with patch("app.services.documents.ingestion.EmbeddingFactory.create") as create:
        service = DocumentIngestionService(
            qdrant_client=Mock(),
            embedding_service=embedding_service,
        )

    assert service.embedding_service is embedding_service
    create.assert_not_called()


@pytest.mark.asyncio
async def test_graph_relations_reference_the_entities_written_for_the_chunk() -> None:
    """Extracted relations must point at the entity IDs actually sent to Neo4j."""
    qdrant_client = Mock(add=AsyncMock())
    embedding_service = Mock(
        embed=AsyncMock(
            return_value=EmbeddingResult(
                embeddings=[[0.1, 0.2]],
                model="demo",
                dimensions=2,
                tokens_used=4,
            )
        )
    )
    extraction = Mock(
        entities=[
            {"name": "退款政策", "type": "Policy"},
            {"name": "退款申请", "type": "Process"},
        ],
        relations=[
            {
                "source": "退款政策",
                "target": "退款申请",
                "type": "GUIDES",
            }
        ],
    )
    extractor = Mock(extract=AsyncMock(return_value=extraction))
    graph_client = Mock(
        add_entities=AsyncMock(return_value=["entity-policy", "entity-process"]),
        add_relations=AsyncMock(),
    )
    service = DocumentIngestionService(
        qdrant_client=qdrant_client,
        embedding_service=embedding_service,
        graph_client=graph_client,
        entity_extractor=extractor,
    )

    result = await service.ingest_text("退款政策指导退款申请。", "退款知识")

    relation = graph_client.add_relations.await_args.args[0][0]
    assert relation.source_entity_id == "entity-policy"
    assert relation.target_entity_id == "entity-process"
    assert result["graph_entities_extracted"] == 2
    assert result["graph_relations_extracted"] == 1
