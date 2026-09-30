#!/usr/bin/env python3
"""Seed the synthetic demo knowledge base into Qdrant via the app pipeline.

The corpus is ``app/data/demo_kb/demo_corpus.json`` — fictional
e-commerce customer-service policies whose numbers are pinned to
``app/services/facts/policy_facts.json`` (no-drift guard:
``app/tests/unit/data/test_demo_corpus.py``). Synthetic data only; it
mirrors the FAQ/fact-table structure and never represents real business
content.

Idempotent: every doc_id is deleted before re-ingestion, so re-running
never duplicates chunks.

Usage (wherever settings can reach Qdrant + the embedding provider):

    python scripts/seed_demo_kb.py

In the deployed container (no rebuild needed — copy both files in):

    docker cp app/data/demo_kb <container>:/tmp/
    docker cp scripts/seed_demo_kb.py <container>:/tmp/
    docker exec <container> python /tmp/seed_demo_kb.py \
        --corpus /tmp/demo_kb/demo_corpus.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

    # Imported lazily so --dry-run never touches app settings/services.
    # Wired directly (mirroring get_ingestion_service) but WITHOUT the
    # per-chunk LLM entity extractor: ~50 sequential GLM extraction
    # calls against the Coding Plan's low concurrency ceiling would be
    # slow and rate-limit-prone; this seeder fills the vector leg only.
    from app.config.settings import get_settings
    from app.services.documents.ingestion import DocumentIngestionService
    from app.services.embeddings import EmbeddingFactory
    from app.services.retrieval.factory import RetrievalFactory

    settings = get_settings()
    embedding_service = EmbeddingFactory.create_from_settings()
    qdrant_client = RetrievalFactory.create_vector_client(
        client_type="qdrant",
        url=settings.VECTOR_DB_URL,
        collection_name=settings.VECTOR_COLLECTION_NAME,
        api_key=settings.VECTOR_API_KEY,
        embedding_service=embedding_service,
    )
    service = DocumentIngestionService(
        qdrant_client=qdrant_client,
        embedding_provider=settings.EMBEDDING_PROVIDER,
        chunking_strategy="semantic",
        embedding_service=embedding_service,
    )
    total_chunks = 0
    for doc in docs:
        # Idempotency: drop any previous version of this doc first.
        # Best-effort: the first seed has nothing to delete, and a
        # deployed container may predate the FilterSelector fix in
        # delete_by_filter — a failed delete must not abort the seed.
        try:
            deleted = await service.delete_document(doc["doc_id"])
            deleted_chunks = deleted["deleted_chunks"]
        except Exception as exc:
            deleted_chunks = 0
            print(f"  warn: pre-ingest delete failed for {doc['doc_id']}: {exc}")
        result = await service.ingest_text(
            text=doc["content"],
            title=doc["title"],
            document_id=doc["doc_id"],
            metadata={
                "lang": doc["lang"],
                "topic": doc["topic"],
                "synthetic": True,
                "source": "demo_kb",
            },
        )
        total_chunks += result["chunks_count"]
        print(
            f"  ingested {doc['doc_id']}: {result['chunks_count']} chunks"
            f" (replaced {deleted_chunks})"
        )

    print(f"done: {len(docs)} docs, {total_chunks} chunks in Qdrant")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--dry-run", action="store_true", help="validate only, no writes")
    args = parser.parse_args()
    asyncio.run(seed(args.corpus, args.dry_run))


if __name__ == "__main__":
    main()
