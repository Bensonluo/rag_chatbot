"""Public policy scope and system-owned chunk provenance."""

import json
from typing import Any

from app.core.exceptions import ValidationError

PIPELINE_VERSION = "support-documents-v2"
RESERVED_FIELDS = frozenset(
    {
        "document_id",
        "version_id",
        "chunk_id",
        "index",
        "title",
        "file_type",
        "file_name",
        "file_path",
        "original_filename",
        "start_pos",
        "end_pos",
        "page",
        "page_start",
        "page_end",
        "section_path",
        "chunk_size",
        "chunking_strategy",
        "content_hash",
        "pipeline_version",
        "uploaded_by",
        "uploaded_by_email",
        "source",
        "source_url",
        "source_type",
        "source_document_id",
        "source_document_ids",
        "source_version_id",
        "source_version_ids",
        "source_chunk_ids",
        "embedding_provider",
        "embedding_model",
    }
)


def validate_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Accept bounded JSON metadata while protecting system-owned fields."""
    metadata = dict(metadata or {})
    protected = RESERVED_FIELDS.intersection(metadata)
    if protected:
        raise ValidationError(f"Reserved metadata fields: {', '.join(sorted(protected))}")
    try:
        serialized = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValidationError("Metadata must contain valid JSON values") from exc
    if len(serialized.encode("utf-8")) > 16_384:
        raise ValidationError("Metadata exceeds 16 KiB")
    for field in ("lang", "topic", "product", "region", "category"):
        if field in metadata and (
            not isinstance(metadata[field], str)
            or not metadata[field].strip()
            or len(metadata[field]) > 200
        ):
            raise ValidationError(
                f"Metadata {field} must be a non-empty string up to 200 characters"
            )
    return metadata


def public_chunk_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Uploader identity belongs in the registry, never in search payloads."""
    return {
        k: v
        for k, v in metadata.items()
        if k not in {"uploaded_by", "uploaded_by_email", "file_path"}
    }
