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
