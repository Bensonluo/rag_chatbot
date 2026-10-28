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
    incomplete = False

    def visit(value: object, depth: int = 0) -> None:
        nonlocal incomplete
        if depth > 8:
            incomplete = True
            return
        if isinstance(value, dict):
            marker = value.get("provenance_incomplete")
            if marker is not None and marker is not False:
                incomplete = True
            for destination, keys in _SOURCE_KEYS.items():
                for key in keys:
                    raw = value.get(key)
                    ids = [raw] if isinstance(raw, str) else raw
                    if isinstance(ids, list):
                        if len(ids) > 50:
                            incomplete = True
                        for item in ids[:50]:
                            if not isinstance(item, str) or not item.strip() or len(item) > 512:
                                incomplete = True
                            elif item in collected[destination]:
                                continue
                            elif len(collected[destination]) < 50:
                                collected[destination].append(item)
                            else:
                                incomplete = True
                    elif raw is not None:
                        incomplete = True
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child, depth + 1)
        elif isinstance(value, list):
            if len(value) > 100:
                incomplete = True
            for child in value[:100]:
                visit(child, depth + 1)

    for value in values:
        visit(value)
    result: dict[str, Any] = {key: ids for key, ids in collected.items() if ids}
    if incomplete:
        result["provenance_incomplete"] = True
    return result
