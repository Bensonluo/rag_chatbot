"""Periodic BM25 index refresh.

The in-memory keyword leg goes stale after the boot-time warmup:
documents ingested later never reach it, and deleted documents never
leave it — a stale hit returns deleted content, which is a
correctness bug, not just a recall gap. A periodic full rebuild from
the vector store fixes both sides and needs no add/delete hooks on
the ingestion path.

Multi-replica note: every process refreshes its own copy, so replicas
converge within one interval. A shared external index remains the
long-term answer for large corpora.
"""

import asyncio
import logging
import time

from prometheus_client import REGISTRY, Gauge, Histogram

from app.services.retrieval.hybrid_search import HybridSearchService
from app.services.retrieval.qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

# Metrics belong to the process (same pattern as retrieval/metrics.py)
# to avoid duplicate registration across app factories and test clients.
KEYWORD_INDEX_CHUNKS = Gauge(
    "keyword_index_chunks",
    "Chunks currently held in the BM25 keyword index",
    registry=REGISTRY,
)
KEYWORD_INDEX_REFRESH_SECONDS = Histogram(
    "keyword_index_refresh_seconds",
    "Duration of one keyword-index refresh, by outcome",
    labelnames=["outcome"],
    registry=REGISTRY,
)

_refresh_task: asyncio.Task[None] | None = None


async def refresh_keyword_index(
    hybrid_search: HybridSearchService,
    vector_client: QdrantClient,
    max_chunks: int = 50_000,
) -> int:
    """
    Rebuild the BM25 leg from a fresh vector-store snapshot.

    Args:
        hybrid_search: Hybrid service whose keyword leg to rebuild
        vector_client: Vector client to scroll the corpus from
        max_chunks: Upper bound on points pulled

    Returns:
        int: Number of chunks now in the keyword index
    """
    chunks = await vector_client.list_chunks(max_chunks=max_chunks)
    await hybrid_search.keyword_search.rebuild(chunks)
    KEYWORD_INDEX_CHUNKS.set(len(chunks))
    return len(chunks)


async def run_keyword_index_refresher(
    hybrid_search: HybridSearchService,
    vector_client: QdrantClient,
    interval_seconds: float,
    max_chunks: int = 50_000,
) -> None:
    """
    Refresh loop: sleep one interval, rebuild, repeat until cancelled.

    Sleeps first so a boot-time warmup is not immediately duplicated.
    A failed refresh is observed and logged; the loop keeps running
    and retries on the next interval.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        started = time.monotonic()
        try:
            await refresh_keyword_index(hybrid_search, vector_client, max_chunks)
            KEYWORD_INDEX_REFRESH_SECONDS.labels(outcome="ok").observe(time.monotonic() - started)
        except Exception:
            KEYWORD_INDEX_REFRESH_SECONDS.labels(outcome="error").observe(
                time.monotonic() - started
            )
            logger.exception("Keyword index refresh failed; retrying next interval")


def start_keyword_index_refresher(
    hybrid_search: HybridSearchService,
    vector_client: QdrantClient,
    interval_seconds: float,
    max_chunks: int = 50_000,
) -> asyncio.Task[None] | None:
    """
    Spawn the refresher task (no-op if one is already running).

    Must be called from a running event loop (service init).
    """
    global _refresh_task
    if _refresh_task is not None and not _refresh_task.done():
        return _refresh_task
    _refresh_task = asyncio.create_task(
        run_keyword_index_refresher(
            hybrid_search=hybrid_search,
            vector_client=vector_client,
            interval_seconds=interval_seconds,
            max_chunks=max_chunks,
        )
    )
    return _refresh_task


def stop_keyword_index_refresher() -> None:
    """Cancel and clear the refresher task (safe when not running)."""
    global _refresh_task
    if _refresh_task is not None and not _refresh_task.done():
        _refresh_task.cancel()
    _refresh_task = None
