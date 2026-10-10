"""Periodic BM25 index refresh: rebuild semantics, loop, lifecycle."""

import asyncio
import contextlib
from typing import Any
from unittest.mock import AsyncMock, Mock

from prometheus_client import REGISTRY

from app.services.retrieval.hybrid_search import HybridSearchService, KeywordSearch
from app.services.retrieval.keyword_refresh import (
    refresh_keyword_index,
    run_keyword_index_refresher,
    start_keyword_index_refresher,
    stop_keyword_index_refresher,
)


def _hybrid() -> HybridSearchService:
    return HybridSearchService(
        vector_client=Mock(), keyword_search=KeywordSearch(), vector_weight=0.5
    )


def _client(chunks: list[dict[str, Any]]) -> Mock:
    client = Mock()
    client.list_chunks = AsyncMock(return_value=chunks)
    return client


def _sample(name: str, labels: dict[str, str] | None = None) -> float:
    value = REGISTRY.get_sample_value(name, labels)
    return value if value is not None else 0.0


class TestRefreshKeywordIndex:
    async def test_rebuilds_index_and_sets_gauge(self):
        hybrid = _hybrid()
        await hybrid.keyword_search.add_documents([{"id": "old", "content": "旧 政策"}])

        refreshed = await refresh_keyword_index(
            hybrid, _client([{"id": "new", "content": "新 政策"}])
        )

        assert refreshed == 1
        assert set(hybrid.keyword_search.documents) == {"new"}
        assert _sample("keyword_index_chunks") == 1.0

    async def test_error_propagates_to_caller(self):
        hybrid = _hybrid()
        client = Mock()
        client.list_chunks = AsyncMock(side_effect=RuntimeError("qdrant down"))

        import pytest

        with pytest.raises(RuntimeError, match="qdrant down"):
            await refresh_keyword_index(hybrid, client)


class TestRefresherLoop:
    async def test_refreshes_periodically_until_cancelled(self):
        hybrid = _hybrid()
        client = _client([{"id": "c1", "content": "政策"}])

        task = asyncio.create_task(
            run_keyword_index_refresher(hybrid, client, interval_seconds=0.01)
        )
        await asyncio.sleep(0.06)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert client.list_chunks.await_count >= 2
        assert task.cancelled()

    async def test_failed_refresh_does_not_kill_the_loop(self):
        hybrid = _hybrid()
        client = Mock()

        calls = 0

        async def flaky_list(**kwargs: Any) -> list[dict[str, Any]]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("transient")
            return [{"id": "c1", "content": "政策"}]

        client.list_chunks = flaky_list

        task = asyncio.create_task(
            run_keyword_index_refresher(hybrid, client, interval_seconds=0.01)
        )
        await asyncio.sleep(0.06)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert calls >= 3  # first failed, loop kept going
        assert _sample("keyword_index_refresh_seconds_count", {"outcome": "error"}) >= 1.0


class TestRefresherLifecycle:
    async def test_start_is_idempotent_and_stop_cancels(self):
        stop_keyword_index_refresher()  # clean slate

        hybrid = _hybrid()
        client = _client([])

        task = start_keyword_index_refresher(hybrid, client, interval_seconds=50)
        assert task is not None
        again = start_keyword_index_refresher(hybrid, client, interval_seconds=50)
        assert task is again

        stop_keyword_index_refresher()
        await asyncio.sleep(0)
        assert task.cancelled()

        stop_keyword_index_refresher()  # no-op when not running


class TestMutationConsistency:
    async def test_publish_replaces_all_old_chunks_immediately(self, monkeypatch):
        import app.services.retrieval.keyword_refresh as module

        hybrid = _hybrid()
        await hybrid.keyword_search.add_documents(
            [
                {"id": "old", "content": "old", "metadata": {"document_id": "doc"}},
                {"id": "other", "content": "other", "metadata": {"document_id": "other"}},
            ]
        )
        monkeypatch.setattr(module, "_keyword_hybrid", hybrid)
        await module.publish_keyword_document(
            "doc",
            [
                {
                    "chunk_id": "new",
                    "content": "new",
                    "metadata": {"title": "Policy"},
                }
            ],
        )
        assert set(hybrid.keyword_search.documents) == {"new", "other"}
        assert hybrid.keyword_search.documents["new"]["metadata"]["document_id"] == "doc"

    async def test_publish_waits_for_inflight_snapshot_so_deleted_chunks_stay_deleted(
        self, monkeypatch
    ):
        import app.services.retrieval.keyword_refresh as module

        hybrid = _hybrid()
        monkeypatch.setattr(module, "_keyword_hybrid", hybrid)
        started, release = asyncio.Event(), asyncio.Event()

        async def old_snapshot(**kwargs):
            started.set()
            await release.wait()
            return [{"id": "old", "content": "old", "metadata": {"document_id": "doc"}}]

        refresh = asyncio.create_task(refresh_keyword_index(hybrid, Mock(list_chunks=old_snapshot)))
        await started.wait()
        purge = asyncio.create_task(module.purge_keyword_document("doc"))
        await asyncio.sleep(0)
        assert not purge.done()
        release.set()
        await asyncio.gather(refresh, purge)
        assert hybrid.keyword_search.documents == {}

    async def test_other_worker_refreshes_before_serving_deleted_document(self, monkeypatch):
        import app.services.retrieval.keyword_refresh as module
        from app.services.retrieval.vector_base import SearchResult, VectorSearchRequest

        vector = _client([])
        vector.search = AsyncMock(return_value=[SearchResult("survivor", "policy", 0.9)])
        hybrid = HybridSearchService(vector, KeywordSearch())
        await hybrid.keyword_search.add_documents(
            [
                {
                    "id": "deleted",
                    "content": "policy",
                    "metadata": {"document_id": "removed"},
                }
            ]
        )
        monkeypatch.setattr(module, "_keyword_hybrid", None)
        monkeypatch.setattr(module, "get_kb_epoch", AsyncMock(return_value="new-epoch"))
        module.register_keyword_index(hybrid)
        hybrid.keyword_epoch = "old-epoch"

        results = await hybrid.search(VectorSearchRequest("policy"))
        assert [result.document_id for result in results] == ["survivor"]
        assert hybrid.keyword_search.documents == {}
        assert hybrid.keyword_epoch == "new-epoch"
        await hybrid.search(VectorSearchRequest("policy"))
        vector.list_chunks.assert_awaited_once()

    async def test_epoch_change_during_scroll_retries_snapshot(self, monkeypatch):
        import app.services.retrieval.keyword_refresh as module

        vector = _client([])
        vector.list_chunks.side_effect = [
            [{"id": "deleted", "content": "old"}],
            [{"id": "new", "content": "new"}],
        ]
        hybrid = HybridSearchService(vector, KeywordSearch())
        monkeypatch.setattr(module, "_keyword_hybrid", None)
        module.register_keyword_index(hybrid)
        monkeypatch.setattr(
            module, "get_kb_epoch", AsyncMock(side_effect=["old", "new", "new", "new"])
        )

        await module.refresh_keyword_index(hybrid, vector)
        assert set(hybrid.keyword_search.documents) == {"new"}
        assert hybrid.keyword_epoch == "new"

    async def test_failed_remote_refresh_never_serves_stale_keyword_hits(self, monkeypatch):
        import app.services.retrieval.keyword_refresh as module
        from app.services.retrieval.vector_base import SearchResult, VectorSearchRequest

        vector = Mock(
            list_chunks=AsyncMock(side_effect=RuntimeError("scroll unavailable")),
            search=AsyncMock(return_value=[SearchResult("live", "policy", 0.9)]),
        )
        hybrid = HybridSearchService(vector, KeywordSearch())
        await hybrid.keyword_search.add_documents([{"id": "stale", "content": "policy"}])
        monkeypatch.setattr(module, "_keyword_hybrid", None)
        monkeypatch.setattr(module, "get_kb_epoch", AsyncMock(return_value="new"))
        module.register_keyword_index(hybrid)

        results = await hybrid.search(VectorSearchRequest("policy"))
        assert [result.document_id for result in results] == ["live"]
