"""Containment KPI refresher: the north-star rate becomes scrape-able.

The session-level containment rate lives in Postgres (repo query), so
Prometheus cannot see it until something bridges the two. This module
runs the query on a refresh loop and sets process-level Gauges — same
doctrine as the BM25 refresher: each replica refreshes independently
from the database; a DB outage leaves the last value in place
(fail-open, logged) instead of scraping the metric to zero.
"""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from prometheus_client import REGISTRY

from app.services.handoff import containment_metrics
from app.services.handoff.containment_metrics import (
    refresh_containment_stats,
    stop_containment_refresher,
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
    repo.get_containment_stats = AsyncMock(
        return_value={
            "window_start": "2026-09-18T00:00:00+00:00",
            "sessions_served": 10,
            "sessions_escalated": 2,
            "sessions_with_failed_replies": 3,
            "sessions_with_completed_replies": 9,
            "sessions_latest_reply_failed": 1,
            "containment_rate": 0.8,
        }
    )
    monkeypatch.setattr(containment_metrics, "TicketRepository", lambda db: repo)
    return repo


@pytest.fixture(autouse=True)
def _no_background_refresher():
    stop_containment_refresher()
    yield
    stop_containment_refresher()


class TestRefreshContainmentStats:
    async def test_sets_gauges_from_repo_result(self, repo: Mock, monkeypatch) -> None:
        monkeypatch.setattr(time, "time", lambda: 1234.0)
        await refresh_containment_stats(session_maker=_session_maker(), window_days=7)

        assert _gauge("chat_containment_rate") == 0.8
        assert _gauge("chat_sessions_served") == 10.0
        assert _gauge("chat_sessions_escalated") == 2.0
        assert _gauge("chat_sessions_with_failed_replies") == 3.0
        assert _gauge("chat_sessions_with_completed_replies") == 9.0
        assert _gauge("chat_sessions_latest_reply_failed") == 1.0
        assert _gauge("chat_containment_stats_last_success_timestamp_seconds") == 1234.0

    async def test_window_days_reaches_the_repo_query(self, repo: Mock) -> None:
        await refresh_containment_stats(session_maker=_session_maker(), window_days=30)

        since = repo.get_containment_stats.await_args.kwargs["since"]
        assert since <= datetime.now(UTC) - timedelta(days=29)

    async def test_no_traffic_leaves_rate_untouched(self, repo: Mock, monkeypatch) -> None:
        """Zero served sessions means the rate is undefined — setting the
        gauge would read as 0% (an outage-shaped lie). Gauges for the
        absolute counts still move so the panel shows why."""
        monkeypatch.setattr(time, "time", lambda: 1234.0)
        await refresh_containment_stats(session_maker=_session_maker(), window_days=7)
        assert _gauge("chat_containment_rate") == 0.8
        repo.get_containment_stats.return_value = {
            "window_start": "x",
            "sessions_served": 0,
            "sessions_escalated": 0,
            "sessions_with_failed_replies": 0,
            "sessions_with_completed_replies": 0,
            "sessions_latest_reply_failed": 0,
            "containment_rate": None,
        }

        monkeypatch.setattr(time, "time", lambda: 2345.0)
        await refresh_containment_stats(session_maker=_session_maker(), window_days=7)

        assert _gauge("chat_containment_rate") == 0.8  # untouched
        assert _gauge("chat_sessions_served") == 0.0
        assert _gauge("chat_sessions_escalated") == 0.0
        assert _gauge("chat_sessions_with_failed_replies") == 0.0
        assert _gauge("chat_sessions_with_completed_replies") == 0.0
        assert _gauge("chat_sessions_latest_reply_failed") == 0.0
        assert _gauge("chat_containment_stats_last_success_timestamp_seconds") == 2345.0

    async def test_db_failure_is_fail_open_and_keeps_last_values(
        self, repo: Mock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(time, "time", lambda: 1234.0)
        await refresh_containment_stats(session_maker=_session_maker(), window_days=7)
        assert _gauge("chat_containment_rate") == 0.8
        broken = Mock()
        broken.get_containment_stats = AsyncMock(side_effect=RuntimeError("db down"))
        monkeypatch.setattr(containment_metrics, "TicketRepository", lambda db: broken)
        monkeypatch.setattr(time, "time", lambda: 2345.0)

        await refresh_containment_stats(session_maker=_session_maker(), window_days=7)  # no raise

        assert _gauge("chat_containment_rate") == 0.8
        assert _gauge("chat_sessions_served") == 10.0
        assert _gauge("chat_sessions_escalated") == 2.0
        assert _gauge("chat_sessions_with_failed_replies") == 3.0
        assert _gauge("chat_sessions_with_completed_replies") == 9.0
        assert _gauge("chat_sessions_latest_reply_failed") == 1.0
        assert _gauge("chat_containment_stats_last_success_timestamp_seconds") == 1234.0

    async def test_incomplete_repo_contract_cannot_publish_partial_success(
        self, repo: Mock, monkeypatch
    ) -> None:
        monkeypatch.setattr(time, "time", lambda: 1234.0)
        await refresh_containment_stats(session_maker=_session_maker())
        repo.get_containment_stats.return_value = {
            "sessions_served": 99,
            "sessions_escalated": 98,
            "sessions_with_failed_replies": 97,
            "sessions_with_completed_replies": 96,
            # Missing latest-reply field is a broken contract, not zero.
            "containment_rate": 0.01,
        }
        monkeypatch.setattr(time, "time", lambda: 2345.0)

        result = await refresh_containment_stats(session_maker=_session_maker())

        assert result is None
        assert _gauge("chat_sessions_served") == 10.0
        assert _gauge("chat_sessions_escalated") == 2.0
        assert _gauge("chat_sessions_with_failed_replies") == 3.0
        assert _gauge("chat_sessions_with_completed_replies") == 9.0
        assert _gauge("chat_sessions_latest_reply_failed") == 1.0
        assert _gauge("chat_containment_rate") == 0.8
        assert _gauge("chat_containment_stats_last_success_timestamp_seconds") == 1234.0


class TestSettingsContract:
    def test_refresh_defaults_are_sane(self) -> None:
        from app.config.settings import Settings

        s = Settings()
        assert s.CONTAINMENT_STATS_REFRESH_SECONDS == 300.0
        assert s.CONTAINMENT_STATS_WINDOW_DAYS == 7


class TestRefresherLifecycle:
    async def test_start_spawns_sleep_first_task_and_stop_cancels(self) -> None:
        task = containment_metrics.start_containment_refresher(
            session_maker=Mock(),
            interval_seconds=3600,
            window_days=7,
        )

        assert task is not None and not task.done()

        stop_containment_refresher()
        # Give the loop a beat to observe the cancellation.
        for _ in range(10):
            if task.done():
                break
            await asyncio.sleep(0)
        assert task.cancelled() or task.done()

    async def test_start_twice_returns_same_task(self) -> None:
        first = containment_metrics.start_containment_refresher(
            session_maker=_session_maker(), interval_seconds=3600, window_days=7
        )
        second = containment_metrics.start_containment_refresher(
            session_maker=_session_maker(), interval_seconds=3600, window_days=7
        )
        assert first is second
        stop_containment_refresher()
        await asyncio.sleep(0)  # let the cancellation land
