"""Regression tests for the document ingestion demo flow."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.services.documents.ingestion import DocumentIngestionService
from app.services.embeddings.base import EmbeddingResult
from app.services.graph.extraction.base import ExtractionResult


@pytest.mark.asyncio
async def test_ingest_text_chunks_embeds_stores_and_returns_summary() -> None:
    """A text upload should complete the full demo ingestion journey."""
    qdrant_client = Mock()
    qdrant_client.replace_document = AsyncMock()

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
    qdrant_client.replace_document.assert_awaited_once()
    payload = qdrant_client.replace_document.await_args.kwargs["payloads"][0]
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

    qdrant_client.delete_by_filter.assert_awaited_once_with({"document_id": "doc-demo"})
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
    qdrant_client = Mock(replace_document=AsyncMock())
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
    extraction = ExtractionResult(
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
        replace_document_graph=AsyncMock(return_value={"entities": 2, "relations": 1}),
    )

    async def embed_all(texts):
        return EmbeddingResult(
            embeddings=[[0.1, 0.2] for _ in texts], model="demo", dimensions=2, tokens_used=4
        )

    embedding_service.embed.side_effect = embed_all
    service = DocumentIngestionService(
        qdrant_client=qdrant_client,
        embedding_service=embedding_service,
        graph_client=graph_client,
        entity_extractor=extractor,
    )

    result = await service.ingest_text("退款政策指导退款申请。", "退款知识")

    _, _, entities, relations = graph_client.replace_document_graph.await_args.args
    relation = relations[0]
    assert relation.source_entity_id == entities[0].id
    assert relation.target_entity_id == entities[1].id
    assert all(entity.embedding is not None for entity in entities)
    assert result["graph_entities_extracted"] == 2
    assert result["graph_relations_extracted"] == 1


@pytest.mark.asyncio
async def test_ingest_text_batches_embedding_calls_at_16() -> None:
    """Docs with many chunks must not blow the embedding API request caps.

    GLM's embedding endpoint rejects >64 texts / ~8-10K total tokens per
    request; the whole document used to ride one unbatched call, so any
    real policy doc (>~20 chunks) failed ingestion wholesale with an
    HTTP 400. Batches are capped at 16 (≈6.4K tokens worst case) and
    re-concatenated in order.
    """
    qdrant_client = Mock()
    qdrant_client.replace_document = AsyncMock()

    batch_sizes: list[int] = []

    async def fake_embed(texts: list[str]) -> EmbeddingResult:
        batch_sizes.append(len(texts))
        return EmbeddingResult(
            embeddings=[[float(i)] for i in range(len(texts))],
            model="demo-embedding",
            dimensions=1,
            tokens_used=len(texts),
        )

    embedding_service = Mock()
    embedding_service.embed = fake_embed

    with patch(
        "app.services.documents.ingestion.EmbeddingFactory.create",
        return_value=embedding_service,
    ):
        service = DocumentIngestionService(
            qdrant_client=qdrant_client,
            chunking_strategy="fixed",
            max_chunk_size=20,
            chunk_overlap=0,
        )

    result = await service.ingest_text(
        text="退款政策条款内容持续说明。" * 40, title="大文档", document_id="doc-big"
    )

    chunks_count = result["chunks_count"]
    assert chunks_count > 16, "test premise: enough chunks to force multiple batches"
    assert len(batch_sizes) > 1
    assert all(size <= DocumentIngestionService.EMBEDDING_BATCH_SIZE for size in batch_sizes)
    assert sum(batch_sizes) == chunks_count
    # Vectors re-concatenated in order: qdrant receives one vector per chunk.
    vectors = qdrant_client.replace_document.await_args.kwargs["vectors"]
    assert len(vectors) == chunks_count
    # Tokens are summed across batches for the ingestion report.
    assert result["total_tokens"] == chunks_count


@pytest.mark.asyncio
async def test_ingest_and_delete_rotate_kb_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every successful Qdrant mutation rotates the KB epoch so
    epoch-scoped caches (L0 answer cache) invalidate wholesale."""
    from app.services.documents import ingestion as ingestion_module

    bump = AsyncMock()
    monkeypatch.setattr(ingestion_module, "bump_kb_epoch", bump)

    qdrant_client = Mock()
    qdrant_client.replace_document = AsyncMock()
    embedding_service = Mock()
    embedding_service.embed = AsyncMock(
        return_value=EmbeddingResult(
            embeddings=[[0.1]], model="demo-embedding", dimensions=1, tokens_used=1
        )
    )
    service = DocumentIngestionService(
        qdrant_client=qdrant_client, embedding_service=embedding_service
    )

    await service.ingest_text("退货政策。", "小文档", document_id="doc-a")
    assert bump.await_count == 1

    # An empty vector delete still repairs stale keyword/graph state.
    qdrant_client.delete_by_filter = AsyncMock(return_value=0)
    await service.delete_document("doc-a")
    assert bump.await_count == 2

    # A real delete is a mutation — rotate.
    qdrant_client.delete_by_filter = AsyncMock(return_value=3)
    await service.delete_document("doc-a")
    assert bump.await_count == 3
