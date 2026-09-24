"""One-shot KPI refresher: the north-star rate becomes scrape-able.

The session-level one-shot rate lives in Postgres (repo query), so
Prometheus cannot see it until something bridges the two. This module
runs the query on a refresh loop and sets process-level Gauges — same
doctrine as the BM25 refresher: every replica reads the same DB rows
so their gauges agree; a DB outage leaves the last value in place
(fail-open, logged) instead of scraping the metric to zero.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from prometheus_client import REGISTRY

from app.services.handoff import one_shot_metrics
from app.services.handoff.one_shot_metrics import (
    refresh_one_shot_stats,
    stop_one_shot_refresher,
)


def _gauge(name: str) -> float | None:
    return REGISTRY.get_sample_value(name)


def _session_maker() -> MagicMock:
    """A session maker whose sessions support ``async with``.

    The repo is monkeypatched in every refresh test, so the sessions
    only need the async context-manager protocol MagicMock provides.
    """
    return MagicMock()


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch) -> Mock:
    repo = Mock()
    repo.get_one_shot_stats = AsyncMock(
        return_value={
            "window_start": "2026-09-18T00:00:00+00:00",
            "sessions_served": 10,
            "sessions_escalated": 2,
            "one_shot_rate": 0.8,
        }
    )
    monkeypatch.setattr(one_shot_metrics, "TicketRepository", lambda db: repo)
    return repo


@pytest.fixture(autouse=True)
def _no_background_refresher():
    stop_one_shot_refresher()
    yield
    stop_one_shot_refresher()


class TestRefreshOneShotStats:
    async def test_sets_gauges_from_repo_result(self, repo: Mock) -> None:
        await refresh_one_shot_stats(session_maker=_session_maker(), window_days=7)

        assert _gauge("chat_one_shot_rate") == 0.8
        assert _gauge("chat_sessions_served") == 10.0
        assert _gauge("chat_sessions_escalated") == 2.0

    async def test_window_days_reaches_the_repo_query(self, repo: Mock) -> None:
        await refresh_one_shot_stats(session_maker=_session_maker(), window_days=30)

        since = repo.get_one_shot_stats.await_args.kwargs["since"]
        assert since <= datetime.now(UTC) - timedelta(days=29)

    async def test_no_traffic_leaves_rate_untouched(self, repo: Mock) -> None:
        """Zero served sessions means the rate is undefined — setting the
        gauge would read as 0% (an outage-shaped lie). Gauges for the
        absolute counts still move so the panel shows why."""
        await refresh_one_shot_stats(session_maker=_session_maker(), window_days=7)
        assert _gauge("chat_one_shot_rate") == 0.8
        repo.get_one_shot_stats.return_value = {
            "window_start": "x",
            "sessions_served": 0,
            "sessions_escalated": 0,
            "one_shot_rate": None,
        }

        await refresh_one_shot_stats(session_maker=_session_maker(), window_days=7)

        assert _gauge("chat_one_shot_rate") == 0.8  # untouched
        assert _gauge("chat_sessions_served") == 0.0

    async def test_db_failure_is_fail_open_and_keeps_last_values(
        self, repo: Mock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await refresh_one_shot_stats(session_maker=_session_maker(), window_days=7)
        assert _gauge("chat_one_shot_rate") == 0.8
        broken = Mock()
        broken.get_one_shot_stats = AsyncMock(side_effect=RuntimeError("db down"))
        monkeypatch.setattr(one_shot_metrics, "TicketRepository", lambda db: broken)

        await refresh_one_shot_stats(session_maker=_session_maker(), window_days=7)  # no raise

        assert _gauge("chat_one_shot_rate") == 0.8
        assert _gauge("chat_sessions_served") == 10.0


class TestSettingsContract:
    def test_refresh_defaults_are_sane(self) -> None:
        from app.config.settings import Settings

        s = Settings()
        assert s.ONE_SHOT_STATS_REFRESH_SECONDS == 300.0
        assert s.ONE_SHOT_STATS_WINDOW_DAYS == 7


class TestRefresherLifecycle:
    async def test_start_spawns_sleep_first_task_and_stop_cancels(self) -> None:
        task = one_shot_metrics.start_one_shot_refresher(
            session_maker=Mock(),
            interval_seconds=3600,
            window_days=7,
        )

        assert task is not None and not task.done()

        stop_one_shot_refresher()
        # Give the loop a beat to observe the cancellation.
        for _ in range(10):
            if task.done():
                break
            await asyncio.sleep(0)
        assert task.cancelled() or task.done()

    async def test_start_twice_returns_same_task(self) -> None:
        first = one_shot_metrics.start_one_shot_refresher(
            session_maker=_session_maker(), interval_seconds=3600, window_days=7
        )
        second = one_shot_metrics.start_one_shot_refresher(
            session_maker=_session_maker(), interval_seconds=3600, window_days=7
        )
        assert first is second
        stop_one_shot_refresher()
        await asyncio.sleep(0)  # let the cancellation land
