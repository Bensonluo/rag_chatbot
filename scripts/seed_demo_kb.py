#!/usr/bin/env python3
"""Seed the synthetic demo knowledge base into Qdrant via the app pipeline.

The corpus is ``app/data/demo_kb/demo_corpus.json`` — fictional
e-commerce customer-service policies whose numbers are pinned to
``app/services/facts/policy_facts.json`` (no-drift guard:
``app/tests/unit/data/test_demo_corpus.py``). Synthetic data only; it
mirrors the FAQ/fact-table structure and never represents real business
content.

Idempotent: the durable ingestion queue deduplicates identical revisions.
Replacement publishes new chunks before retiring the previous revision.

Run from a matching application checkout/image after database migrations.
Settings must reach Postgres, Qdrant, Redis, the configured embedding provider,
and Neo4j when GRAPH_RAG_ENABLED=true. Extraction is disabled for this synthetic
corpus; graph storage still connects to clean up replaced document assertions.

    python scripts/seed_demo_kb.py --dry-run
    python scripts/seed_demo_kb.py

In a deployed container built from this revision:

    docker exec <container> python scripts/seed_demo_kb.py
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.document_import import ImportDocument, ingest_documents

DEFAULT_CORPUS = (
    Path(__file__).resolve().parent.parent / "app" / "data" / "demo_kb" / "demo_corpus.json"
)


def load_corpus(path: Path) -> list[dict[str, Any]]:
    """Load and structurally validate the corpus file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    docs: list[dict[str, Any]] = data["docs"]

    seen_ids: set[str] = set()
    topics_by_lang: dict[str, set[str]] = {"zh": set(), "en": set()}
    for doc in docs:
        doc_id = doc["doc_id"]
        if doc_id in seen_ids:
            raise ValueError(f"duplicate doc_id: {doc_id}")
        seen_ids.add(doc_id)
        if not doc.get("content", "").strip() or not doc.get("title", "").strip():
            raise ValueError(f"empty title/content: {doc_id}")
        if doc["lang"] not in topics_by_lang:
            raise ValueError(f"unexpected lang {doc['lang']!r}: {doc_id}")
        topics_by_lang[doc["lang"]].add(doc["topic"])

    unpaired = topics_by_lang["zh"] ^ topics_by_lang["en"]
    if unpaired:
        raise ValueError(f"topics missing a zh/en pair: {sorted(unpaired)}")
    return docs


async def seed(corpus_path: Path, dry_run: bool) -> None:
    docs = load_corpus(corpus_path)
    print(f"corpus: {len(docs)} docs across {len({d['topic'] for d in docs})} topics")

    if dry_run:
        for doc in docs:
            print(f"  would ingest {doc['doc_id']} ({doc['lang']}, ~{len(doc['content'])} chars)")
        return

    await ingest_documents(
        [
            ImportDocument(
                document_id=doc["doc_id"],
                title=doc["title"],
                content=doc["content"],
                metadata={"lang": doc["lang"], "topic": doc["topic"], "synthetic": True},
            )
            for doc in docs
        ],
        graph_enabled=False,
    )
    print(f"done: {len(docs)} retained document revisions indexed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--dry-run", action="store_true", help="validate only, no writes")
    args = parser.parse_args()
    asyncio.run(seed(args.corpus, args.dry_run))


if __name__ == "__main__":
    main()
