"""One-shot-stats endpoint: the north-star KPI has a URL.

The session-level one-shot rate (repo layer: served sessions without a
handoff ticket / served sessions) only moves the product if something
can poll it. These tests pin the endpoint contract: default 7-day
window, window_days param control, and the repo result passed through
untouched.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from app.api.v1 import sessions as sessions_api


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
    monkeypatch.setattr(sessions_api, "TicketRepository", lambda db: repo)
    return repo


class TestOneShotStatsEndpoint:
    async def test_returns_repo_result(self, repo: Mock) -> None:
        result = await sessions_api.get_one_shot_stats(db=Mock(), current_user=Mock())

        assert result["sessions_served"] == 10
        assert result["one_shot_rate"] == 0.8

    async def test_default_window_is_seven_days(self, repo: Mock) -> None:
        before = datetime.now(UTC)

        await sessions_api.get_one_shot_stats(db=Mock(), current_user=Mock())

        since = repo.get_one_shot_stats.await_args.kwargs["since"]
        assert before - timedelta(days=7) <= since <= datetime.now(UTC) - timedelta(days=7)

    async def test_window_days_param_controls_since(self, repo: Mock) -> None:
        await sessions_api.get_one_shot_stats(window_days=30, db=Mock(), current_user=Mock())

        since = repo.get_one_shot_stats.await_args.kwargs["since"]
        expected_max = datetime.now(UTC) - timedelta(days=30)
        assert since <= expected_max
