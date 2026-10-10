"""Evidence-preservation and publication contracts across the ingestion pipeline."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from app.core.exceptions import ExternalServiceError, ValidationError
from app.services.documents import ingestion, parsing
from app.services.embeddings.base import EmbeddingResult
from app.services.graph.extraction.base import ExtractionResult
from app.services.graph.extraction.llm_extractor import LLMEntityExtractor
from app.tests.unit.services.documents.test_parsing import make_pdf, text_page


@pytest.fixture
def service(monkeypatch):
    async def embed(texts):
        return EmbeddingResult(
            embeddings=[[float(len(text)), 1.0] for text in texts],
            model="test-embedding",
            dimensions=2,
            tokens_used=len(texts),
        )

    monkeypatch.setattr(ingestion, "publish_keyword_document", AsyncMock())
    monkeypatch.setattr(ingestion, "purge_keyword_document", AsyncMock())
    monkeypatch.setattr(ingestion, "bump_kb_epoch", AsyncMock())
    return ingestion.DocumentIngestionService(
        qdrant_client=Mock(replace_document=AsyncMock()),
        embedding_service=Mock(embed=AsyncMock(side_effect=embed)),
    )


def payloads(service):
    return service.qdrant_client.replace_document.await_args.kwargs["payloads"]


@pytest.mark.asyncio
async def test_long_document_keeps_tail_evidence_and_exact_chunk_offsets(service) -> None:
    text = "A" * 100000 + "\n\nTAIL_EVIDENCE"

    result = await service.ingest_text(text, "Long policy", document_id="policy")

    assert result["_normalized_text"] == text
    stored = payloads(service)
    assert any("TAIL_EVIDENCE" in chunk["content"] for chunk in stored)
    for chunk in stored:
        metadata = chunk["metadata"]
        assert chunk["content"] == text[metadata["start_pos"] : metadata["end_pos"]]
        assert len(chunk["content"]) <= 512


@pytest.mark.asyncio
async def test_pdf_title_type_and_page_citations_survive_ingestion(service) -> None:
    data = make_pdf([text_page("First page policy"), text_page("Second page exceptions")])

    result = await service.ingest_bytes(
        data=data,
        title="Customer refund policy",
        file_type="pdf",
        document_id="policy",
        metadata={"original_filename": "refunds.pdf", "category": "policy"},
    )

    metadata = payloads(service)[0]["metadata"]
    assert result["title"] == metadata["title"] == "Customer refund policy"
    assert metadata["file_type"] == "pdf"
    assert metadata["source"] == "refunds.pdf"
    assert metadata["page_start"] == 1
    assert metadata["page_end"] == 2
    assert metadata["category"] == "policy"
    assert metadata["embedding_provider"] == "local"
    assert metadata["embedding_model"] == "test-embedding"


@pytest.mark.asyncio
async def test_blank_pdf_pages_do_not_produce_empty_search_chunks(service, monkeypatch) -> None:
    pages = [parsing.ParsedPage(number=1, text="Refund policy.")]
    pages.extend(parsing.ParsedPage(number=n, text="") for n in range(2, 400))
    pages.append(parsing.ParsedPage(number=400, text="Exceptions apply."))
    monkeypatch.setattr(
        parsing,
        "parse_document",
        AsyncMock(return_value=parsing.ParsedDocument(pages, ["Blank pages"])),
    )

    result = await service.ingest_bytes(data=b"pdf", title="Policy", file_type="pdf")

    assert result["_normalized_text"] == "Refund policy.\n\nExceptions apply."
    assert all(chunk["content"].strip() for chunk in payloads(service))
    assert payloads(service)[0]["metadata"]["page_end"] == 400
    assert result["warnings"] == ["Blank pages"]


@pytest.mark.asyncio
async def test_markdown_type_is_normalized_and_headings_remain_searchable(service) -> None:
    text = "# Refunds\n\n## Eligibility\nApply within seven days."

    await service.ingest_bytes(data=text.encode(), title="Refunds", file_type=".MD")

    stored = payloads(service)[0]
    assert stored["content"] == text
    assert stored["metadata"]["file_type"] == "md"
    assert stored["metadata"]["section_path"] == ["Refunds"]


@pytest.mark.asyncio
async def test_markdown_code_comments_are_not_section_headings(service) -> None:
    service.max_chunk_size, service.chunk_overlap = 48, 0
    text = (
        "# Policy\n\n## Examples\n\n```python\n# Fake heading\n"
        + "print('example')\n" * 8
        + "```\n"
    )

    await service.ingest_bytes(data=text.encode(), title="Policy", file_type="md")

    paths = [chunk["metadata"].get("section_path", []) for chunk in payloads(service)]
    assert all("Fake heading" not in path for path in paths)
    assert ["Policy", "Examples"] in paths


@pytest.mark.asyncio
async def test_markdown_filename_title_and_private_metadata_are_preserved_safely(
    service, tmp_path: Path
) -> None:
    source = tmp_path / "temporary-upload.md"
    source.write_text("# Refunds\n\nSeven days.")
    metadata = {
        "original_filename": "refund-policy.md",
        "uploaded_by": "private-user-id",
        "uploaded_by_email": "private@example.com",
        "file_path": "/private/temp/file.md",
    }

    result = await service.ingest_file(str(source), metadata)

    assert result["title"] == "refund-policy"
    stored_metadata = payloads(service)[0]["metadata"]
    assert stored_metadata["file_type"] == "md"
    assert stored_metadata["source"] == "refund-policy.md"
    assert not {"uploaded_by", "uploaded_by_email", "file_path"}.intersection(stored_metadata)
    assert metadata["file_path"] == "/private/temp/file.md"


@pytest.mark.asyncio
async def test_multilingual_embeddings_obey_count_and_byte_budgets(service) -> None:
    text = "退款政策支持七天内申请。" * 1500

    await service.ingest_text(text, "退款政策")

    calls = service.embedding_service.embed.await_args_list
    assert len(calls) > 1
    for call in calls:
        batch = call.args[0]
        assert len(batch) <= 16
        assert sum(len(text.encode("utf-8")) for text in batch) <= 8000
    assert sum(len(call.args[0]) for call in calls) == len(payloads(service))


@pytest.mark.asyncio
async def test_reingestion_replaces_document_with_stable_chunk_ids(service) -> None:
    text = "Refund policy. " * 100

    first = await service.ingest_text(text, "Refund policy", document_id="policy")
    first_ids = service.qdrant_client.replace_document.await_args.kwargs["ids"]
    repeated = await service.ingest_text(text, "Refund policy", document_id="policy")
    repeated_ids = service.qdrant_client.replace_document.await_args.kwargs["ids"]
    changed = await service.ingest_text(
        text + "New exclusion.", "Refund policy", document_id="policy"
    )
    changed_ids = service.qdrant_client.replace_document.await_args.kwargs["ids"]

    assert first["version_id"] == repeated["version_id"]
    assert first_ids == repeated_ids
    assert changed["version_id"] != first["version_id"]
    assert set(changed_ids).isdisjoint(first_ids)
    assert all(
        call.args[0] == "policy" for call in service.qdrant_client.replace_document.await_args_list
    )
    service.qdrant_client.add.assert_not_called()
    assert ingestion.publish_keyword_document.await_count == 3


@pytest.mark.asyncio
async def test_embedding_count_mismatch_does_not_publish_partial_document(service) -> None:
    service.embedding_service.embed = AsyncMock(return_value=EmbeddingResult([], "test", 2, 0))

    with pytest.raises(ExternalServiceError, match="count"):
        await service.ingest_text("Refunds allowed.", "Policy")

    service.qdrant_client.replace_document.assert_not_awaited()
    ingestion.publish_keyword_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_oversized_embedding_chunk_is_rejected_before_publication(service) -> None:
    service.max_chunk_size = 10000

    with pytest.raises(ValidationError, match="budget"):
        await service.ingest_text("退" * 3000, "Policy")

    service.qdrant_client.replace_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_failure_is_explicit_and_old_graph_is_removed(service) -> None:
    service.graph_enabled = True
    service.graph_client = Mock(delete_document_graph=AsyncMock())
    service.entity_extractor = Mock(extract=AsyncMock(side_effect=RuntimeError("offline")))

    result = await service.ingest_text("Refunds allowed.", "Policy", document_id="policy")

    assert result["graph_status"] == "failed"
    assert result["warnings"]
    service.qdrant_client.replace_document.assert_awaited_once()
    service.graph_client.delete_document_graph.assert_awaited_once_with("policy")


@pytest.mark.asyncio
async def test_graph_entities_have_embeddings_and_source_chunk_ids(service) -> None:
    service.graph_enabled = True
    service.graph_client = Mock(
        replace_document_graph=AsyncMock(return_value={"entities": 2, "relations": 1})
    )
    service.entity_extractor = Mock(
        extract=AsyncMock(
            return_value=ExtractionResult(
                entities=[
                    {"name": "Refund", "type": "Policy", "description": "Seven days"},
                    {"name": "Customer", "type": "Person"},
                ],
                relations=[{"source": "Refund", "target": "Customer", "type": "COVERS"}],
            )
        )
    )

    result = await service.ingest_text(
        "Refund policy applies to customers. " * 30, "Policy", document_id="policy"
    )

    doc_id, version_id, entities, relations = (
        service.graph_client.replace_document_graph.await_args.args
    )
    assert doc_id == "policy" and version_id == result["version_id"]
    assert result["graph_status"] == "indexed"
    chunk_ids = {chunk["chunk_id"] for chunk in payloads(service)}
    assert len(chunk_ids) > 1
    assert len(entities) == 2 and len(relations) == 1
    for entity in entities:
        assert entity.embedding is not None and len(entity.embedding) == 2
        assert set(entity.properties["source_chunk_ids"]) == chunk_ids
    assert set(relations[0].properties["source_chunk_ids"]) == chunk_ids
    assert {relations[0].source_entity_id, relations[0].target_entity_id} == {
        entity.id for entity in entities
    }


@pytest.mark.asyncio
async def test_replacement_clears_old_graph_when_extraction_is_disabled(service) -> None:
    events = []

    async def clear_graph(document_id):
        events.append("graph cleared")

    async def replace_vectors(*args, **kwargs):
        events.append("vectors replaced")

    service.graph_client = Mock(delete_document_graph=AsyncMock(side_effect=clear_graph))
    service.qdrant_client.replace_document.side_effect = replace_vectors

    result = await service.ingest_text("New policy.", "Policy", document_id="policy")

    assert events == ["graph cleared", "vectors replaced"]
    assert result["graph_status"] == "disabled"


@pytest.mark.asyncio
async def test_storage_enabled_without_graph_client_cannot_publish_replacement(service) -> None:
    service.graph_storage_enabled = True

    with pytest.raises(ExternalServiceError, match="Graph"):
        await service.ingest_text("New policy.", "Policy", document_id="policy")

    service.qdrant_client.replace_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_clears_old_graph_even_when_extraction_is_disabled(service) -> None:
    service.graph_client = Mock(delete_document_graph=AsyncMock())
    service.qdrant_client.delete_by_filter = AsyncMock(return_value=2)

    result = await service.delete_document("policy")

    service.graph_client.delete_document_graph.assert_awaited_once_with("policy")
    assert result["deleted_chunks"] == 2


@pytest.mark.asyncio
async def test_failed_graph_cleanup_prevents_vector_deletion(service) -> None:
    service.graph_client = Mock(
        delete_document_graph=AsyncMock(side_effect=RuntimeError("offline"))
    )
    service.qdrant_client.delete_by_filter = AsyncMock()

    with pytest.raises(ExternalServiceError, match="Graph"):
        await service.delete_document("policy")

    service.qdrant_client.delete_by_filter.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_vector_replacement_invalidates_cached_old_evidence(service) -> None:
    state = []

    async def replace(document_id, **kwargs):
        state.extend(kwargs["ids"])
        raise RuntimeError("Old points could not be retired")

    service.qdrant_client.replace_document.side_effect = replace

    with pytest.raises(RuntimeError, match="retired"):
        await service.ingest_text("New policy.", "Policy", document_id="policy")

    assert state
    ingestion.purge_keyword_document.assert_awaited_once_with("policy")
    ingestion.bump_kb_epoch.assert_awaited_once()
    ingestion.publish_keyword_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_vector_deletion_timeout_also_rotates_cache_epoch(service) -> None:
    service.qdrant_client.delete_by_filter = AsyncMock(side_effect=TimeoutError())

    with pytest.raises(TimeoutError):
        await service.delete_document("policy")

    ingestion.bump_kb_epoch.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        RuntimeError("provider offline"),
        "not JSON",
        '{"refusal": "cannot extract"}',
        '{"entities": "invalid", "relations": []}',
        '{"entities": [{"name": "Refund", "type": "UnknownType"}], "relations": []}',
    ],
)
async def test_real_extractor_failures_cannot_mark_graph_indexed(service, response) -> None:
    llm = Mock(generate=AsyncMock())
    if isinstance(response, Exception):
        llm.generate.side_effect = response
    else:
        llm.generate.return_value = Mock(content=response)
    service.graph_enabled = True
    service.graph_client = Mock(
        replace_document_graph=AsyncMock(return_value={"entities": 0, "relations": 0}),
        delete_document_graph=AsyncMock(),
    )
    service.entity_extractor = LLMEntityExtractor(llm)

    result = await service.ingest_text("Refunds allowed.", "Policy", document_id="policy")

    assert result["graph_status"] == "failed"
    assert "retried" in result["warnings"][-1]
    service.qdrant_client.replace_document.assert_awaited_once()
    service.graph_client.replace_document_graph.assert_not_awaited()
    service.graph_client.delete_document_graph.assert_awaited_once_with("policy")


@pytest.mark.asyncio
async def test_valid_empty_extraction_is_distinct_from_provider_failure(service) -> None:
    llm = Mock(generate=AsyncMock(return_value=Mock(content='{"entities": [], "relations": []}')))
    service.graph_enabled = True
    service.graph_client = Mock(
        replace_document_graph=AsyncMock(return_value={"entities": 0, "relations": 0})
    )
    service.entity_extractor = LLMEntityExtractor(llm)

    result = await service.ingest_text("General introduction.", "Policy")

    assert result["graph_status"] == "indexed"
    service.graph_client.replace_document_graph.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingestion_closes_owned_extraction_llm_after_vector_close_failure(service) -> None:
    llm = Mock(close=AsyncMock())
    service.entity_extractor = LLMEntityExtractor(llm, owns_llm=True)
    service.qdrant_client.client.close = AsyncMock(side_effect=RuntimeError("vector close failed"))
    service.embedding_service.close = AsyncMock()

    with pytest.raises(RuntimeError, match="vector close failed"):
        await service.close()

    llm.close.assert_awaited_once()
    service.embedding_service.close.assert_awaited_once()
