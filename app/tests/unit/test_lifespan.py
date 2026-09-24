"""Application lifespan shutdown choreography contract.

Rolling deploys and scale-in replace replicas constantly at the north
star's scale, so shutdown must be as wired as startup: every periodic
task started in lifespan teardown order — background refreshers first,
then the connection pools they use. ``stop_one_shot_refresher`` was
defined for exactly this and never called: on graceful shutdown the
refresher could still be mid-query against a closing pool.

Function-local imports inside ``lifespan`` re-resolve the source module
at call time, so each seam is monkeypatched at its home module.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI

import app.api.v1.chat as chat_api
import app.middleware.rate_limiter_redis as rate_limiter_module
import app.services.dialogue.checkpointer as checkpointer_module
import app.services.handoff.one_shot_metrics as one_shot_module
import app.services.retrieval.keyword_refresh as keyword_refresh_module
from app.main import lifespan


class _FakeCheckpointerManager:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

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
    oneshot_stop = Mock()
    monkeypatch.setattr(
        checkpointer_module, "create_checkpointer_manager_from_settings", lambda: manager
    )
    monkeypatch.setattr(chat_api, "initialize_chat_service", AsyncMock())
    monkeypatch.setattr(keyword_refresh_module, "stop_keyword_index_refresher", keyword_stop)
    monkeypatch.setattr(rate_limiter_module, "close_rate_limit_redis", redis_close)
    monkeypatch.setattr(one_shot_module, "stop_one_shot_refresher", oneshot_stop)
    return SimpleNamespace(
        manager=manager,
        keyword_stop=keyword_stop,
        redis_close=redis_close,
        oneshot_stop=oneshot_stop,
    )


class TestLifespanShutdown:
    async def test_shutdown_stops_every_started_background_task(self, seams):
        async with lifespan(FastAPI()):
            assert seams.manager.started, "startup must have run"

        assert seams.keyword_stop.called, "BM25 keyword refresher must stop"
        assert seams.redis_close.await_count == 1, "rate-limit Redis must close"
        assert seams.oneshot_stop.called, "one-shot KPI refresher must stop too"

    async def test_refreshers_stop_before_the_checkpointer_pool_closes(self, seams):
        """Teardown order: background tasks first, then the pools they
        use — a refresher mid-query against a closing pool is exactly
        the shutdown-window noise this contract exists to prevent."""
        order: list[str] = []
        seams.keyword_stop.side_effect = lambda: order.append("keyword")
        seams.oneshot_stop.side_effect = lambda: order.append("oneshot")
        seams.manager.stop = AsyncMock(side_effect=lambda: order.append("checkpointer"))

        async with lifespan(FastAPI()):
            pass

        assert order.index("oneshot") < order.index("checkpointer")
        assert order.index("keyword") < order.index("checkpointer")
