#!/usr/bin/env python3
"""Import data/*.jsonl through the application's durable ingestion pipeline.

Accepts JSONL records and the existing single-object JSON files. Each record
uses doc_id, title, content (or legacy doc_content), and optional metadata.
Retains the record's content as a TXT source; identical revisions deduplicate.

Run from a matching application checkout/image after database migrations.
Settings must reach Postgres, Qdrant, Redis, the configured embedding provider,
and Neo4j when GRAPH_RAG_ENABLED=true. Use --dry-run for offline validation.

    python scripts/init_kb.py --data-dir data --dry-run
    python scripts/init_kb.py --data-dir data
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.document_import import ImportDocument, ingest_documents

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _records(path: Path) -> list[tuple[int, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []
    try:
        # Preserve support for a single (possibly pretty-printed) JSON object.
        return [(1, json.loads(text))]
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append((line_number, json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number}: Invalid JSON: {exc.msg}") from exc
        return records


def load_documents(data_dir: Path) -> list[ImportDocument]:
    """Validate all input records before connecting to application services."""
    if not data_dir.is_dir():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    documents = []
    seen_ids: set[str] = set()
    for path in sorted(data_dir.glob("*.jsonl")):
        records = _records(path)
        for line_number, record in records:
            location = f"{path.name}:{line_number}"
            if not isinstance(record, dict):
                raise ValueError(f"{location}: Expected a JSON object")
            fallback_id = path.name if len(records) == 1 else f"{path.stem}-{line_number}"
            document_id = record.get("doc_id", fallback_id)
            if (
                not isinstance(document_id, str)
                or not 1 <= len(document_id) <= 255
                or not re.fullmatch(r"[\w.-]+", document_id)
            ):
                raise ValueError(f"{location}: Invalid doc_id")
            if document_id in seen_ids:
                raise ValueError(f"{location}: Duplicate doc_id: {document_id}")
            seen_ids.add(document_id)
            title = record.get("title", document_id)
            content = record.get("content", record.get("doc_content", ""))
            if not isinstance(title, str) or not title.strip() or len(title) > 500:
                raise ValueError(f"{location}: Title must contain 1–500 characters")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{location}: Content must be a non-empty string")
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"{location}: Metadata must be a JSON object")
            documents.append(ImportDocument(document_id, title, content, metadata))
    return documents


async def initialize(data_dir: Path, dry_run: bool) -> None:
    documents = load_documents(data_dir)
    print(f"corpus: {len(documents)} documents")
    if dry_run:
        for doc in documents:
            print(f"  would ingest {doc.document_id} (~{len(doc.content)} chars)")
        return
    if not documents:
        print("No documents found in data directory")
        return
    revisions = await ingest_documents(documents)
    print(f"done: {len(revisions)} retained document revisions indexed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--dry-run", action="store_true", help="validate only, no writes")
    args = parser.parse_args()
    asyncio.run(initialize(args.data_dir, args.dry_run))


if __name__ == "__main__":
    main()
