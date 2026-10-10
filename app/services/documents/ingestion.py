"""Lossless document ingestion with versioned chunks and explicit graph status."""

import asyncio
import hashlib
import logging
import re
import uuid
from bisect import bisect_right
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.core.exceptions import ExternalServiceError, ValidationError
from app.services.documents.base import ChunkingStrategy, Document, DocumentChunk
from app.services.documents.chunking import (
    FixedSizeChunking,
    RecursiveCharacterChunking,
    SemanticChunking,
)
from app.services.documents.metadata import PIPELINE_VERSION, public_chunk_metadata
from app.services.documents.preprocessing import DocumentPreprocessor
from app.services.embeddings import EmbeddingFactory
from app.services.embeddings.base import EmbeddingServiceBase
from app.services.graph.base import GraphClient, GraphEntity, GraphRelation
from app.services.graph.extraction.base import EntityExtractor
from app.services.retrieval.kb_epoch import bump_kb_epoch
from app.services.retrieval.keyword_refresh import publish_keyword_document, purge_keyword_document
from app.services.retrieval.qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


def _markdown_sections(text: str) -> list[tuple[int, list[str]]]:
    """Build heading ancestry once, excluding fenced code and preserving source offsets."""
    sections: list[tuple[int, list[str]]] = []
    ancestry: list[tuple[int, str]] = []
    fence: tuple[str, int] | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if marker:
            delimiter = marker[1]
            if fence is None:
                fence = (delimiter[0], len(delimiter))
            elif delimiter[0] == fence[0] and len(delimiter) >= fence[1] and not marker[2].strip():
                fence = None
        elif fence is None:
            heading = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+)$", line.rstrip("\r\n"))
            if heading:
                level = len(heading[1])
                title = re.sub(r"[ \t]+#+[ \t]*$", "", heading[2]).strip()
                ancestry = [(depth, name) for depth, name in ancestry if depth < level]
                ancestry.append((level, title))
                sections.append((offset, [name for _, name in ancestry]))
        offset += len(line)
    return sections


class DocumentIngestionService:
    CHUNKING_STRATEGIES: dict[str, Callable[[], ChunkingStrategy]] = {
        "fixed": FixedSizeChunking,
        "semantic": SemanticChunking,
        "recursive": RecursiveCharacterChunking,
    }
    EMBEDDING_BATCH_SIZE = 16
    # UTF-8 bytes are a conservative upper bound for byte-tokenizer input,
    # unlike chars/4 (which badly undercounts Chinese policy text).
    EMBEDDING_BATCH_BYTES = 8_000

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_provider: str = "local",
        chunking_strategy: str = "semantic",
        max_chunk_size: int = 512,
        chunk_overlap: int = 50,
        graph_client: GraphClient | None = None,
        entity_extractor: EntityExtractor | None = None,
        embedding_service: EmbeddingServiceBase | None = None,
        graph_enabled: bool | None = None,
        ocr_languages: str = "eng+chi_sim",
        graph_storage_enabled: bool | None = None,
    ) -> None:
        if chunking_strategy not in self.CHUNKING_STRATEGIES:
            raise ValidationError(f"Unknown chunking strategy: {chunking_strategy}")
        if type(max_chunk_size) is not int or max_chunk_size <= 0:
            raise ValidationError("Chunk size must be a positive integer")
        if type(chunk_overlap) is not int or not 0 <= chunk_overlap < max_chunk_size:
            raise ValidationError("Chunk overlap must be nonnegative and smaller than chunk size")
        self.qdrant_client = qdrant_client
        self.embedding_provider = embedding_provider
        self.chunking_strategy = chunking_strategy
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.graph_client = graph_client
        self.entity_extractor = entity_extractor
        self.graph_enabled = graph_client is not None if graph_enabled is None else graph_enabled
        self.graph_storage_enabled = bool(
            graph_storage_enabled or graph_client is not None or self.graph_enabled
        )
        self.ocr_languages = ocr_languages
        self.embedding_service = embedding_service or EmbeddingFactory.create(
            provider=embedding_provider
        )
        self.chunking = self.CHUNKING_STRATEGIES[chunking_strategy]()
        self.preprocessor = DocumentPreprocessor()

    async def ingest_text(
        self,
        text: str,
        title: str,
        metadata: dict[str, Any] | None = None,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        return await self._ingest_pages([(None, text)], title, "txt", metadata, document_id, [])

    async def ingest_bytes(
        self,
        *,
        data: bytes,
        title: str,
        file_type: str,
        metadata: dict[str, Any] | None = None,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        from app.services.documents.parsing import parse_document

        file_type = file_type.lower().lstrip(".")
        parsed = await parse_document(data, file_type, ocr_languages=self.ocr_languages)
        return await self._ingest_pages(
            [(page.number, page.text) for page in parsed.pages],
            title,
            file_type,
            metadata,
            document_id,
            parsed.warnings,
        )

    async def ingest_file(
        self,
        file_path: str,
        metadata: dict[str, Any] | None = None,
        document_id: str | None = None,
    ) -> dict[str, Any]:
        path = Path(file_path)
        if not path.is_file():
            raise ValidationError(f"File not found: {path.name}")
        metadata = dict(metadata or {})
        filename = str(metadata.get("original_filename") or path.name)
        metadata["original_filename"] = filename
        title = str(metadata.get("title") or Path(filename).stem)
        return await self.ingest_bytes(
            data=await asyncio.to_thread(path.read_bytes),
            title=title,
            file_type=path.suffix.lower().lstrip("."),
            metadata=metadata,
            document_id=document_id,
        )

    async def _embed(self, texts: list[str]) -> tuple[list[list[float]], int, str]:
        vectors: list[list[float]] = []
        tokens = 0
        model = ""
        batch: list[str] = []
        size = 0

        async def flush() -> None:
            nonlocal tokens, model
            result = await self.embedding_service.embed(batch)
            if len(result.embeddings) != len(batch):
                raise ExternalServiceError(
                    "Embeddings", "Embedding count does not match chunk count"
                )
            vectors.extend(result.embeddings)
            tokens += result.tokens_used
            model = result.model

        for text in texts:
            length = len(text.encode("utf-8"))
            if length > self.EMBEDDING_BATCH_BYTES:
                raise ValidationError("Chunk exceeds the embedding input budget; reduce chunk size")
            if batch and (
                len(batch) >= self.EMBEDDING_BATCH_SIZE
                or size + length > self.EMBEDDING_BATCH_BYTES
            ):
                await flush()
                batch, size = [], 0
            batch.append(text)
            size += length
        if batch:
            await flush()
        return vectors, tokens, model

    async def _ingest_pages(
        self,
        pages: list[tuple[int | None, str]],
        title: str,
        file_type: str,
        metadata: dict[str, Any] | None,
        document_id: str | None,
        warnings: list[str],
    ) -> dict[str, Any]:
        if not title.strip():
            raise ValidationError("Document title is required")
        metadata = public_chunk_metadata(dict(metadata or {}))
        pieces: list[str] = []
        page_ranges: list[tuple[int, int, int]] = []
        position = 0
        for number, raw in pages:
            processed, _ = await self.preprocessor.process(raw)
            if number is not None and not processed.strip():
                continue
            if pieces:
                position += 2
            start = position
            pieces.append(processed)
            position += len(processed)
            if number is not None:
                page_ranges.append((start, position, number))
        text = "\n\n".join(pieces)
        if not text.strip():
            raise ValidationError("Document text cannot be empty")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        document_id = document_id or str(uuid.uuid5(uuid.NAMESPACE_URL, title + file_type + digest))
        version_id = str(
            metadata.get("version_id")
            or uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{document_id}:{digest}:{PIPELINE_VERSION}:{self.chunking_strategy}:{self.max_chunk_size}:{self.chunk_overlap}",
            )
        )
        metadata.update(
            title=title.strip(),
            version_id=version_id,
            file_type=file_type,
            pipeline_version=PIPELINE_VERSION,
        )
        metadata["source"] = metadata.get("original_filename") or title.strip()
        document = Document(
            document_id=document_id,
            title=title.strip(),
            content=text,
            file_type=file_type,
            metadata=metadata,
        )
        chunks = await self.chunking.chunk(document, self.max_chunk_size, self.chunk_overlap)
        if not chunks:
            raise ValidationError("Document chunking produced no chunks")
        sections = _markdown_sections(text) if file_type == "md" else []
        section_positions = [position for position, _ in sections]
        for chunk in chunks:
            start, end = chunk.metadata["start_pos"], chunk.metadata["end_pos"]
            page_numbers = [n for a, b, n in page_ranges if a < end and b > start]
            if page_numbers:
                chunk.metadata.update(page_start=page_numbers[0], page_end=page_numbers[-1])
            if sections:
                section_index = bisect_right(section_positions, start) - 1
                if section_index < 0 and section_positions[0] < end:
                    section_index = 0
                if section_index >= 0:
                    chunk.metadata["section_path"] = sections[section_index][1]
        vectors, total_tokens, embedding_model = await self._embed([c.content for c in chunks])
        for chunk in chunks:
            chunk.metadata.update(
                embedding_provider=self.embedding_provider, embedding_model=embedding_model
            )
        payloads = [
            {
                "chunk_id": c.chunk_id,
                "document_id": document_id,
                "content": c.content,
                "metadata": {
                    **c.metadata,
                    "chunk_id": c.chunk_id,
                    "document_id": document_id,
                    "version_id": version_id,
                    "index": c.index,
                },
            }
            for c in chunks
        ]
        # Callers serialize writes to one document (the durable worker uses a
        # database row lock). Publish new points before retiring the old revision.
        if not self.graph_enabled and (self.graph_storage_enabled or self.graph_client is not None):
            # Turning extraction off must retire assertions from older revisions.
            # Refuse publication if that cleanup cannot be completed.
            try:
                await self._clear_document_graph(document_id)
            finally:
                await bump_kb_epoch()
        try:
            await self.qdrant_client.replace_document(
                document_id, ids=[c.chunk_id for c in chunks], vectors=vectors, payloads=payloads
            )
        except Exception:
            # An upsert or retirement timeout can leave a changed vector corpus.
            # Evict local keyword evidence and rotate caches before a safe retry.
            try:
                await purge_keyword_document(document_id)
            finally:
                await bump_kb_epoch()
            raise
        await publish_keyword_document(document_id, payloads)
        await bump_kb_epoch()
        graph_status = "disabled"
        graph_counts = {"entities": 0, "relations": 0}
        if self.graph_enabled:
            graph_status = "failed"
            try:
                if self.graph_client is None or self.entity_extractor is None:
                    raise ExternalServiceError("Graph", "Graph extraction is unavailable")
                graph_counts = await self._write_graph(document_id, version_id, chunks)
                graph_status = "indexed"
            except Exception:
                logger.exception("Graph indexing failed for document %s", document_id)
                # Remove the superseded graph so it cannot answer with old policy
                # assertions. Cleanup failure is surfaced to operators in the job.
                try:
                    await self._clear_document_graph(document_id)
                except Exception:
                    logger.exception("Graph cleanup failed for document %s", document_id)
                    warnings.append("Graph cleanup failed; graph indexing must be retried")
                warnings.append("Vector index is ready; graph indexing failed and can be retried")
            finally:
                await bump_kb_epoch()
        return {
            "document_id": document_id,
            "version_id": version_id,
            "title": title.strip(),
            "chunks_count": len(chunks),
            "total_tokens": total_tokens,
            "embedding_model": embedding_model,
            "chunking_strategy": self.chunking_strategy,
            "graph_status": graph_status,
            "graph_entities_extracted": graph_counts["entities"],
            "graph_relations_extracted": graph_counts["relations"],
            "warnings": warnings,
            "_normalized_text": text,
        }

    async def _write_graph(
        self, document_id: str, version_id: str, chunks: list[DocumentChunk]
    ) -> dict[str, int]:
        assert self.entity_extractor is not None and self.graph_client is not None
        entities: dict[tuple[str, str], GraphEntity] = {}
        relations: dict[tuple[str, str, str], GraphRelation] = {}
        for chunk in chunks:
            extraction = await self.entity_extractor.extract(chunk.content)
            error = getattr(extraction, "error", None)
            if isinstance(error, str) and error:
                raise ExternalServiceError("Graph extraction", error)
            names: dict[str, str] = {}
            for raw in extraction.entities:
                entity_key = (raw["name"], raw["type"])
                if entity_key not in entities:
                    entities[entity_key] = GraphEntity(
                        id=str(
                            uuid.uuid5(
                                uuid.NAMESPACE_URL, f"{document_id}:{version_id}:{entity_key}"
                            )
                        ),
                        name=raw["name"],
                        type=raw["type"],
                        description=raw.get("description"),
                        properties={**raw.get("properties", {}), "source_chunk_ids": []},
                    )
                entity = entities[entity_key]
                if chunk.chunk_id not in entity.properties["source_chunk_ids"]:
                    entity.properties["source_chunk_ids"].append(chunk.chunk_id)
                names[entity.name] = entity.id
            for raw in extraction.relations:
                if raw["source"] not in names or raw["target"] not in names:
                    continue
                relation_key = (names[raw["source"]], names[raw["target"]], raw["type"])
                if relation_key not in relations:
                    relations[relation_key] = GraphRelation(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(relation_key))),
                        source_entity_id=relation_key[0],
                        target_entity_id=relation_key[1],
                        relation_type=relation_key[2],
                        properties={**raw.get("properties", {}), "source_chunk_ids": []},
                    )
                relations[relation_key].properties["source_chunk_ids"].append(chunk.chunk_id)
        entity_list = list(entities.values())
        if entity_list:
            vectors, _, _ = await self._embed(
                [
                    (e.name + ": " + (e.description or ""))[: self.max_chunk_size]
                    for e in entity_list
                ]
            )
            for entity, vector in zip(entity_list, vectors, strict=True):
                entity.embedding = vector
        return await self.graph_client.replace_document_graph(
            document_id, version_id, entity_list, list(relations.values())
        )

    async def _clear_document_graph(self, document_id: str) -> None:
        if self.graph_client is None:
            raise ExternalServiceError(
                "Graph", "Cannot clear graph assertions while graph service is unavailable"
            )
        try:
            await self.graph_client.delete_document_graph(document_id)
        except Exception as exc:
            raise ExternalServiceError(
                "Graph", "Failed to clear the document's graph assertions"
            ) from exc

    async def delete_document(self, document_id: str) -> dict[str, Any]:
        try:
            if self.graph_storage_enabled or self.graph_client is not None or self.graph_enabled:
                await self._clear_document_graph(document_id)
            deleted = await self.qdrant_client.delete_by_filter({"document_id": document_id})
            await purge_keyword_document(document_id)
        finally:
            # Also rotates on an idempotent retry after a partial deletion.
            await bump_kb_epoch()
        return {"document_id": document_id, "deleted_chunks": deleted}

    async def close(self) -> None:
        """Release request/worker-owned clients; the shared graph stays connected."""
        from app.services.embeddings.cached_embeddings import CachedEmbeddingService

        closers = [self.qdrant_client.client.close]
        service = self.embedding_service
        if isinstance(service, CachedEmbeddingService):
            closers.append(service.close)
            service = service.embedding_service
        close = getattr(service, "close", None)
        if close is not None:
            closers.append(close)
        if self.entity_extractor is not None:
            closers.append(self.entity_extractor.close)
        errors: list[Exception] = []
        for close in closers:
            try:
                await close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
