"""Keep each process's BM25 snapshot synchronized with document mutations."""

import asyncio
import logging
import time
from typing import Any, cast

from prometheus_client import REGISTRY, Gauge, Histogram

from app.services.retrieval.hybrid_search import HybridSearchService
from app.services.retrieval.kb_epoch import EPOCH_UNAVAILABLE, get_kb_epoch
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

# The live hybrid service whose keyword leg this process serves. The
# refresher already holds the instance; the registry exposes it so the
# ingestion path can purge deleted documents synchronously instead of
# waiting out the refresh interval (review 2026-09-26, #7). Overwritten
# on re-init; never cleared on stop — a stopped refresher leaves the
# index correct, merely frozen.
_keyword_hybrid: HybridSearchService | None = None


def register_keyword_index(hybrid_search: HybridSearchService) -> None:
    """Register local mutations and check remote mutations before keyword search."""
    global _keyword_hybrid
    _keyword_hybrid = hybrid_search
    hybrid_search.ensure_keyword_fresh = lambda: ensure_keyword_index_current(hybrid_search)


async def publish_keyword_document(document_id: str, payloads: list[dict[str, Any]]) -> int:
    """Replace one document in the local index before publishing the KB epoch."""
    hybrid = _keyword_hybrid
    if hybrid is None:
        return 0
    chunks = [
        {
            "id": payload["chunk_id"],
            "content": payload["content"],
            "metadata": {
                **(payload.get("metadata") or {}),
                "document_id": document_id,
                "chunk_id": payload["chunk_id"],
            },
        }
        for payload in payloads
    ]
    try:
        if hybrid.publication_guard is not None:
            chunks = await hybrid.publication_guard.filter_documents(chunks)
        async with hybrid.keyword_index_lock:
            hybrid.keyword_epoch = None
            await hybrid.keyword_search.remove_document(document_id)
            await hybrid.keyword_search.add_documents(chunks)
            KEYWORD_INDEX_CHUNKS.set(len(hybrid.keyword_search.documents))
        return len(chunks)
    except Exception:
        hybrid.keyword_epoch = None
        logger.warning("Keyword publication failed; next search will refresh", exc_info=True)
        return 0


async def purge_keyword_document(document_id: str) -> int:
    """
    Remove a deleted document's chunks from the live keyword leg.

    Fail-open by design: a missing registration (keyword leg not yet
    warmed) or a purge failure must not break the delete API — the
    periodic rebuild remains the backstop that eventually drops the
    chunks in every process.

    Args:
        document_id: Document whose chunks to purge

    Returns:
        int: Chunks removed (0 when nothing is registered)
    """
    if _keyword_hybrid is None:
        return 0
    try:
        async with _keyword_hybrid.keyword_index_lock:
            _keyword_hybrid.keyword_epoch = None
            removed = await _keyword_hybrid.keyword_search.remove_document(document_id)
            if removed:
                KEYWORD_INDEX_CHUNKS.set(len(_keyword_hybrid.keyword_search.documents))
                logger.info("Purged %d keyword chunks of document %s", removed, document_id)
        return removed
    except Exception:  # noqa: BLE001 - fail-open; rebuild is the backstop
        logger.warning(
            "Keyword purge failed for document %s; periodic rebuild will drop it",
            document_id,
            exc_info=True,
        )
        return 0


async def refresh_keyword_index(
    hybrid_search: HybridSearchService,
    vector_client: QdrantClient,
    max_chunks: int | None = None,
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
    async with hybrid_search.keyword_index_lock:
        return await _refresh_locked(hybrid_search, vector_client, max_chunks)


async def _refresh_locked(
    hybrid_search: HybridSearchService,
    vector_client: QdrantClient,
    max_chunks: int | None = None,
) -> int:
    # A scroll can span a mutation in another process. Never mark that
    # mixed snapshot current: retry against a stable epoch first.
    tracked = hybrid_search.ensure_keyword_fresh is not None
    for _ in range(3):
        epoch = await get_kb_epoch() if tracked else None
        chunks = await vector_client.list_chunks(max_chunks=max_chunks)
        if hybrid_search.publication_guard is not None:
            chunks = await hybrid_search.publication_guard.filter_documents(chunks)
        if tracked and epoch != await get_kb_epoch():
            continue
        await hybrid_search.keyword_search.rebuild(chunks)
        hybrid_search.keyword_epoch = epoch
        KEYWORD_INDEX_CHUNKS.set(len(chunks))
        return len(chunks)
    hybrid_search.keyword_epoch = None
    raise RuntimeError("Knowledge base changed throughout keyword refresh")


async def ensure_keyword_index_current(hybrid_search: HybridSearchService) -> bool:
    """Refresh before serving another process's mutation; skip stale BM25 on failure."""
    try:
        epoch = await get_kb_epoch()
        if epoch == EPOCH_UNAVAILABLE:
            return False
        if hybrid_search.keyword_epoch == epoch:
            return True
        async with hybrid_search.keyword_index_lock:
            # Another request may have refreshed while this one waited.
            epoch = await get_kb_epoch()
            if epoch == EPOCH_UNAVAILABLE:
                return False
            if hybrid_search.keyword_epoch != epoch:
                await _refresh_locked(
                    hybrid_search, cast("QdrantClient", hybrid_search.vector_client)
                )
            return hybrid_search.keyword_epoch not in (None, EPOCH_UNAVAILABLE)
    except Exception:
        logger.warning("Keyword snapshot is stale; using vector search only", exc_info=True)
        return False


async def run_keyword_index_refresher(
    hybrid_search: HybridSearchService,
    vector_client: QdrantClient,
    interval_seconds: float,
    max_chunks: int | None = None,
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
    max_chunks: int | None = None,
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
