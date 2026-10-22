"""Collect independent graph source sets without leaking arbitrary properties."""

from typing import Any

_SOURCE_KEYS = {
    "source_document_ids": ("source_document_ids", "source_document_id"),
    "source_version_ids": ("source_version_ids", "source_version_id"),
    "source_chunk_ids": ("source_chunk_ids", "source_chunk_id"),
}


def collect_source_provenance(*values: object) -> dict[str, Any]:
    """Retain existing IDs from graph rows/properties; never pair flattened sets."""
    collected: dict[str, list[str]] = {key: [] for key in _SOURCE_KEYS}

    def visit(value: object, depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(value, dict):
            for destination, keys in _SOURCE_KEYS.items():
                for key in keys:
                    raw = value.get(key)
                    ids = [raw] if isinstance(raw, str) else raw
                    if isinstance(ids, list):
                        for item in ids[:50]:
                            if (
                                isinstance(item, str)
                                and item
                                and len(item) <= 512
                                and item not in collected[destination]
                                and len(collected[destination]) < 50
                            ):
                                collected[destination].append(item)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child, depth + 1)
        elif isinstance(value, list):
            for child in value[:100]:
                visit(child, depth + 1)

    for value in values:
        visit(value)
    return {key: ids for key, ids in collected.items() if ids}
