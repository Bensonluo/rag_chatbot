"""Build public citation snapshots without exposing arbitrary retrieval metadata."""

import hashlib
import json
from typing import Any

from pydantic import ValidationError

from app.models.schemas.citation import SourceCitation

_GRAPH_TYPES = {"graph_embedding", "text_to_cypher", "community_summary", "graph"}
MAX_CITATIONS = 20


def normalize_citations(value: object) -> list[dict[str, Any]]:
    """Validate persisted/cache references; absent or malformed entries add no evidence."""
    if not isinstance(value, list):
        return []
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value[:MAX_CITATIONS]:
        if not isinstance(item, (dict, SourceCitation)):
            continue
        try:
            citation = SourceCitation.model_validate(item).model_dump(
                mode="json", exclude_none=True
            )
        except ValidationError:
            continue
        key = json.dumps(citation, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            citations.append(citation)
    return citations


def _text(value: object, limit: int = 512) -> str | None:
    return value[:limit] if isinstance(value, str) and value.strip() else None


def _identifier(value: object) -> str | None:
    # Truncating a revision/chunk/document identity creates a different source.
    return value if isinstance(value, str) and value.strip() and len(value) <= 512 else None


def _strings(value: object, limit: int = 50, *, display: bool = False) -> list[str]:
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list):
        return []
    strings = [(_text(v) if display else _identifier(v)) for v in values]
    return list(dict.fromkeys(v for v in strings if v is not None))[:limit]


def citations_from_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Snapshot only supplied evidence. Missing revision/location stays unknown."""
    citations = []
    for document in documents[:MAX_CITATIONS]:
        metadata = document.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        source_type = metadata.get("source_type")
        graph = isinstance(source_type, str) and source_type in _GRAPH_TYPES
        document_id = _identifier(document.get("document_id")) or _identifier(
            metadata.get("document_id")
        )
        # Fusion's graph_<digest> identifies a synthesized result, not a document.
        if graph:
            document_id = _identifier(metadata.get("source_document_id"))
        citation: dict[str, Any] = {
            "source_type": "graph" if graph else "document",
            "document_id": document_id,
            "version_id": _identifier(metadata.get("version_id"))
            or _identifier(metadata.get("source_version_id")),
            "chunk_id": _identifier(metadata.get("chunk_id")),
            "title": _text(metadata.get("title"), 500),
            "excerpt": _text(document.get("content"), 800) or "",
            "section_path": _strings(metadata.get("section_path"), 20, display=True),
            "source_document_ids": _strings(metadata.get("source_document_ids")),
            "source_version_ids": _strings(metadata.get("source_version_ids")),
            "source_chunk_ids": _strings(metadata.get("source_chunk_ids")),
        }
        for key in ("page_start", "page_end", "start_pos", "end_pos"):
            value = metadata.get(key)
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= (1 if key.startswith("page") else 0)
            ):
                citation[key] = value
        citations.append(citation)
    return normalize_citations(citations)


def faq_citations(faq_id: str, question: str, answer: str) -> list[dict[str, Any]]:
    """Version a curated answer by content; no document-revision claim is implied."""
    digest = hashlib.sha256(
        json.dumps([faq_id, question, answer], ensure_ascii=False).encode()
    ).hexdigest()
    return normalize_citations(
        [
            {
                "source_type": "faq",
                "faq_id": _identifier(faq_id),
                "title": question[:500] or None,
                "version_id": f"sha256:{digest}",
                "excerpt": answer[:800],
            }
        ]
    )


def sources_from_citations(citations: list[dict[str, Any]]) -> list[str]:
    """Legacy document/FAQ identifiers derived only from real source identities."""
    sources: list[str] = []
    for citation in normalize_citations(citations):
        candidates = [citation.get("document_id"), *citation.get("source_document_ids", [])]
        if citation.get("source_type") == "faq" and citation.get("faq_id"):
            candidates.append(f"faq:{citation['faq_id']}")
        for value in candidates:
            if isinstance(value, str) and value and value not in sources:
                sources.append(value)
    return sources
