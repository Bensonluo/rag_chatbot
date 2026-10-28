"""Application lifespan shutdown choreography contract.

Rolling deploys and scale-in replace replicas constantly at the north
star's scale, so shutdown must be as wired as startup: every periodic
task started in lifespan teardown order — background refreshers first,
then the connection pools they use. ``stop_containment_refresher`` was
defined for exactly this and never called: on graceful shutdown the
refresher could still be mid-query against a closing pool.

Function-local imports inside ``lifespan`` re-resolve the source module
at call time, so each seam is monkeypatched at its home module.
"""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import app.api.database as database_module
import app.api.v1.chat as chat_api
import app.api.v1.documents as documents_api
import app.middleware.rate_limiter_redis as rate_limiter_module
import app.services.dialogue.checkpointer as checkpointer_module
import app.services.handoff.containment_metrics as containment_module
import app.services.retrieval.keyword_refresh as keyword_refresh_module
from app.config.settings import settings
from app.main import lifespan
from app.models.database.base import Base
from app.services.documents.jobs import DocumentJobService


class _FakeCheckpointerManager:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        # Lifespan reads this to withhold readiness when postgres mode
        # was requested but MemorySaver substituted (review #12).
        self.degraded = False

    async def start(self) -> object:
        self.started = True
        return object()

    async def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def seams(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Patch every startup/shutdown seam at its source module."""
    manager = _FakeCheckpointerManager()
    keyword_stop = Mock()
    redis_close = AsyncMock()
    containment_stop = Mock()
    monkeypatch.setattr(
        checkpointer_module, "create_checkpointer_manager_from_settings", lambda: manager
    )
    monkeypatch.setattr(chat_api, "initialize_chat_service", AsyncMock())
    monkeypatch.setattr(keyword_refresh_module, "stop_keyword_index_refresher", keyword_stop)
    monkeypatch.setattr(rate_limiter_module, "close_rate_limit_redis", redis_close)
    monkeypatch.setattr(containment_module, "stop_containment_refresher", containment_stop)
    return SimpleNamespace(
        manager=manager,
        keyword_stop=keyword_stop,
        redis_close=redis_close,
        containment_stop=containment_stop,
    )


@pytest.fixture
async def document_jobs(monkeypatch):
    """Run the real durable worker against isolated SQL persistence."""
    monkeypatch.setattr("app.services.documents.jobs.bump_kb_epoch", AsyncMock())
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    async with engine.begin() as connection:
        await connection.run_sync(
            lambda sync: Base.metadata.create_all(
                sync,
                tables=[
                    Base.metadata.tables["documents"],
                    Base.metadata.tables["document_versions"],
                ],
            )
        )
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(database_module, "async_session_maker", sessions)
    yield DocumentJobService(sessions)
    await engine.dispose()


async def _queue_document(jobs: DocumentJobService, document_id: str) -> dict[str, Any]:
    return await jobs.submit(
        data=b"Refunds are available within seven days.",
        title="Refund policy",
        filename="refunds.txt",
        file_type="txt",
        metadata={},
        pipeline_config={},
        document_id=document_id,
    )


class TestLifespanShutdown:
    async def test_worker_executes_jobs_and_drains_cancellation_before_pools_close(
        self, seams, document_jobs, monkeypatch
    ):
        """Exercise actual task scheduling, job commits, and awaited cleanup."""
        monkeypatch.setattr(settings, "DOCUMENT_WORKER_ENABLED", True)
        completed = await _queue_document(document_jobs, "complete-first")
        interrupted = await _queue_document(document_jobs, "interrupt-second")
        started = asyncio.Event()
        begin_shutdown = asyncio.Event()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        order = []

        async def ingest_inflight(**kwargs):
            started.set()
            await asyncio.Event().wait()

        async def close_inflight():
            order.append("worker-cleanup-start")
            cleanup_started.set()
            await release_cleanup.wait()
            order.append("worker-cleanup-finished")

        complete_service = SimpleNamespace(
            ingest_bytes=AsyncMock(
                return_value={"chunks_count": 1, "_normalized_text": "Refund policy."}
            ),
            close=AsyncMock(),
        )
        inflight_service = SimpleNamespace(
            ingest_bytes=AsyncMock(side_effect=ingest_inflight),
            close=AsyncMock(side_effect=close_inflight),
        )
        factory = AsyncMock(side_effect=[complete_service, inflight_service])
        monkeypatch.setattr(documents_api, "build_ingestion_service", factory)
        seams.keyword_stop.side_effect = lambda: order.append("keyword-stopped")
        seams.containment_stop.side_effect = lambda: order.append("containment-stopped")
        seams.manager.stop = AsyncMock(side_effect=lambda: order.append("checkpointer-closed"))
        seams.redis_close.side_effect = lambda: order.append("redis-closed")
        app = FastAPI()

        async def serve() -> None:
            async with lifespan(app):
                await begin_shutdown.wait()

        application = asyncio.create_task(serve())
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            assert not app.state.document_worker.done()
            assert factory.await_count == 2
            assert (await document_jobs.get_version(completed["version_id"]))["status"] == "indexed"
            assert (await document_jobs.get_version(interrupted["version_id"]))[
                "status"
            ] == "processing"

            begin_shutdown.set()
            await asyncio.wait_for(cleanup_started.wait(), timeout=2)
            assert not application.done(), "lifespan must await the worker's client cleanup"
            assert not app.state.document_worker.done()
            seams.manager.stop.assert_not_awaited()
            seams.redis_close.assert_not_awaited()
            seams.keyword_stop.assert_not_called()

            release_cleanup.set()
            await asyncio.wait_for(application, timeout=2)
            assert app.state.document_worker.cancelled()
            assert order.index("worker-cleanup-finished") < order.index("checkpointer-closed")
            assert order.index("worker-cleanup-finished") < order.index("redis-closed")
            assert order.index("worker-cleanup-finished") < order.index("keyword-stopped")
            # Cancellation leaves a persisted processing job for a new worker,
            # rather than losing the source or falsely marking it indexed.
            status = await document_jobs.get_version(interrupted["version_id"])
            assert status["status"] == "processing"
            assert status["attempts"] == 1
            assert (await document_jobs.source(interrupted["version_id"]))[0]
            complete_service.close.assert_awaited_once()
            inflight_service.close.assert_awaited_once()
        finally:
            begin_shutdown.set()
            release_cleanup.set()
            if not application.done():
                await asyncio.wait_for(application, timeout=2)

    async def test_disabled_worker_leaves_queued_jobs_unprocessed(
        self, seams, document_jobs, monkeypatch
    ):
        monkeypatch.setattr(settings, "DOCUMENT_WORKER_ENABLED", False)
        uploaded = await _queue_document(document_jobs, "remain-queued")
        factory = AsyncMock()
        monkeypatch.setattr(documents_api, "build_ingestion_service", factory)
        app = FastAPI()
        async with lifespan(app):
            await asyncio.sleep(0)
            assert app.state.document_worker is None
            assert (await document_jobs.get_version(uploaded["version_id"]))["status"] == "queued"
        factory.assert_not_awaited()
        assert seams.manager.stopped
        seams.redis_close.assert_awaited_once()

    async def test_shutdown_stops_every_started_background_task(self, seams):
        async with lifespan(FastAPI()):
            assert seams.manager.started, "startup must have run"

        assert seams.keyword_stop.called, "BM25 keyword refresher must stop"
        assert seams.redis_close.await_count == 1, "rate-limit Redis must close"
        assert seams.containment_stop.called, "containment KPI refresher must stop too"

    async def test_refreshers_stop_before_the_checkpointer_pool_closes(self, seams):
        """Teardown order: background tasks first, then the pools they
        use — a refresher mid-query against a closing pool is exactly
        the shutdown-window noise this contract exists to prevent."""
        order: list[str] = []
        seams.keyword_stop.side_effect = lambda: order.append("keyword")
        seams.containment_stop.side_effect = lambda: order.append("oneshot")
        seams.manager.stop = AsyncMock(side_effect=lambda: order.append("checkpointer"))

        async with lifespan(FastAPI()):
            pass

        assert order.index("oneshot") < order.index("checkpointer")
        assert order.index("keyword") < order.index("checkpointer")
