"""Authorize new knowledge evidence against one committed registry statement.

Physical index writes are not publication. The active document pointer is the
durable publication fact; a version's current attempt status may change during
retry without revoking an earlier publication. Unregistered raw corpora must be
imported through the durable jobs before they can serve new customer answers.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from app.models.database.document import Document, DocumentVersion
from app.services.graph.base import GraphSearchResult
from app.services.graph.provenance import collect_source_provenance
from app.services.retrieval.vector_base import SearchResult

_GRAPH_TYPES = frozenset({"graph", "graph_embedding", "text_to_cypher", "community_summary"})
_SOURCE_LIMIT = 50


class PublicationUnavailableError(RuntimeError):
    """The committed publication registry could not be read within its budget."""


@dataclass(frozen=True)
class _Evidence:
    documents: frozenset[str]
    versions: frozenset[str]
    faq_id: str | None = None


def _identifier(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() and len(value) <= 512 else None


def _identifiers(value: object) -> frozenset[str] | None:
    if not isinstance(value, list) or len(value) >= _SOURCE_LIMIT:
        return None
    if any(_identifier(item) is None for item in value):
        return None
    return frozenset(value)


def _is_graph_metadata(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    source_type = value.get("source_type")
    return isinstance(source_type, str) and source_type in _GRAPH_TYPES


def _graph_evidence(*values: object) -> _Evidence | None:
    provenance = collect_source_provenance(*values)
    if provenance.get("provenance_incomplete"):
        return None
    documents = _identifiers(provenance.get("source_document_ids"))
    versions = _identifiers(provenance.get("source_version_ids"))
    chunks = provenance.get("source_chunk_ids", [])
    if not documents or not versions or _identifiers(chunks) is None:
        return None
    return _Evidence(documents, versions)


def _document_evidence(
    document_id: object, metadata: object, version_id: object = None
) -> _Evidence | None:
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        return None
    if metadata.get("provenance_incomplete"):
        return None
    meta_document = metadata.get("document_id")
    meta_version = metadata.get("version_id")
    if document_id is not None and _identifier(document_id) is None:
        return None
    if meta_document is not None and _identifier(meta_document) is None:
        return None
    if document_id is not None and meta_document is not None and document_id != meta_document:
        return None
    if version_id is not None and _identifier(version_id) is None:
        return None
    if meta_version is not None and _identifier(meta_version) is None:
        return None
    if version_id is not None and meta_version is not None and version_id != meta_version:
        return None
    canonical_document = _identifier(document_id) or _identifier(meta_document)
    canonical_version = _identifier(version_id) or _identifier(meta_version)
    if not canonical_document or not canonical_version:
        return None
    # A chunk cannot silently carry another document/revision's provenance.
    for key, expected in (
        ("source_document_ids", canonical_document),
        ("source_version_ids", canonical_version),
    ):
        if key in metadata:
            values = _identifiers(metadata[key])
            if values is None or (values and values != {expected}):
                return None
    for key, expected in (
        ("source_document_id", canonical_document),
        ("source_version_id", canonical_version),
    ):
        if metadata.get(key) is not None and metadata[key] != expected:
            return None
    return _Evidence(frozenset({canonical_document}), frozenset({canonical_version}))


def _dictionary_evidence(document: object) -> _Evidence | None:
    if not isinstance(document, dict):
        return None
    metadata = document.get("metadata")
    if _is_graph_metadata(metadata):
        return _graph_evidence(metadata)
    return _document_evidence(document.get("document_id"), metadata, document.get("version_id"))


def _citation_evidence(citation: object) -> _Evidence | None:
    if not isinstance(citation, dict):
        return None
    source_type = citation.get("source_type")
    if source_type == "faq":
        faq_id = _identifier(citation.get("faq_id"))
        version = citation.get("version_id")
        if (
            not faq_id
            or not isinstance(version, str)
            or not version.startswith("sha256:")
            or len(version) != 71
            or any(character not in "0123456789abcdef" for character in version[7:])
            or citation.get("document_id")
            or citation.get("source_document_ids")
            or citation.get("source_version_ids")
        ):
            return None
        return _Evidence(frozenset(), frozenset(), faq_id=faq_id)
    if source_type == "graph":
        provenance = dict(citation)
        if citation.get("document_id") is not None:
            provenance["source_document_id"] = citation["document_id"]
        if citation.get("version_id") is not None:
            provenance["source_version_id"] = citation["version_id"]
        return _graph_evidence(provenance)
    if source_type == "document":
        return _document_evidence(citation.get("document_id"), citation)
    return None


class PublicationGuard:
    """Fail closed on unproved evidence; preserve input objects and their order."""

    def __init__(self, session_maker: Callable[[], Any], timeout_seconds: float = 3.0) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("Publication registry timeout must be finite and positive")
        self._session_maker = session_maker
        self._timeout_seconds = timeout_seconds

    async def _published(self, evidence: Sequence[_Evidence | None]) -> dict[str, str]:
        versions = {version for item in evidence if item is not None for version in item.versions}
        if not versions:
            return {}
        # All referenced version ownership and document activation facts are
        # read in one statement. No independent read-committed SELECTs are
        # misrepresented as an atomic snapshot, and no per-hit queries run.
        statement = (
            select(DocumentVersion.id, DocumentVersion.document_id)
            .join(Document, Document.external_doc_id == DocumentVersion.document_id)
            .where(
                DocumentVersion.id.in_(versions),
                Document.is_active.is_(True),
                Document.active_version_id == DocumentVersion.id,
                DocumentVersion.status != "deleted",
            )
        )
        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with self._session_maker() as session:
                    rows = (await session.execute(statement)).all()
                    return dict(rows)
        except Exception as exc:
            raise PublicationUnavailableError(
                "Knowledge publication registry is unavailable"
            ) from exc

    @staticmethod
    def _allowed(evidence: _Evidence | None, published: dict[str, str]) -> bool:
        if evidence is None:
            return False
        if evidence.faq_id is not None:
            return True
        if not evidence.documents or not evidence.versions:
            return False
        # Independent graph sets are resolved using actual version ownership,
        # never zipped. Every claimed document and every version must agree.
        owners = {published.get(version) for version in evidence.versions}
        return None not in owners and owners == evidence.documents

    async def filter_results(self, results: Sequence[SearchResult]) -> list[SearchResult]:
        evidence = [
            _graph_evidence(result.metadata)
            if _is_graph_metadata(result.metadata)
            else _document_evidence(result.document_id, result.metadata)
            for result in results
        ]
        published = await self._published(evidence)
        return [
            result
            for result, item in zip(results, evidence, strict=True)
            if self._allowed(item, published)
        ]

    async def filter_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        evidence = [_dictionary_evidence(document) for document in documents]
        published = await self._published(evidence)
        return [
            document
            for document, item in zip(documents, evidence, strict=True)
            if self._allowed(item, published)
        ]

    async def filter_graph_results(
        self, results: Sequence[GraphSearchResult]
    ) -> list[GraphSearchResult]:
        evidence = [
            _graph_evidence(
                result.metadata,
                [entity.properties for entity in result.entities],
                [relation.properties for relation in result.relations],
            )
            for result in results
        ]
        published = await self._published(evidence)
        return [
            result
            for result, item in zip(results, evidence, strict=True)
            if self._allowed(item, published)
        ]

    async def answer_allowed(self, sources: list[str], citations: list[dict[str, Any]]) -> bool:
        if not sources or not citations or any(_identifier(source) is None for source in sources):
            return False
        evidence = [_citation_evidence(citation) for citation in citations]
        if any(item is None for item in evidence):
            return False
        covered = {
            source
            for item in evidence
            if item is not None
            for source in (item.documents if item.faq_id is None else {f"faq:{item.faq_id}"})
        }
        if covered != set(sources):
            return False
        published = await self._published(evidence)
        return all(self._allowed(item, published) for item in evidence)
