"""Review-queue endpoints: the QA sampling loop has a URL.

Pins the endpoint contracts on top of the repository behavior pinned
in test_review_queue.py: the queue maps sampled sessions with a
bounded last-message preview, a verdict echoes the recorded state,
missing sessions 404, and strict deployments gate both endpoints to
admins (transcripts of other users + KPI shaping is an ops surface,
not a customer one — same doctrine as KB writes).
"""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from app.api.v1 import sessions as sessions_api
from app.config.settings import settings
from app.models.schemas.session import SessionReviewVerdict
from app.repositories.session_repository import ReviewSnapshot
from app.services.chat.coordination import SessionCoordinator


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch) -> Mock:
    repo = Mock()
    repo.sample_review_queue = AsyncMock(return_value=[])
    repo.last_user_message = AsyncMock(return_value=None)
    repo.record_review_verdict = AsyncMock(return_value=None)
    repo.get_review_snapshot = AsyncMock(
        return_value=ReviewSnapshot(_chat_row(), "a" * 64, 12, 0, True)
    )
    monkeypatch.setattr(sessions_api, "SessionRepository", lambda db: repo)
    return repo


def _chat_row(sid: int = 7) -> Mock:
    return Mock(
        id=sid,
        title="Demo Chat",
        updated_at=datetime(2026, 10, 3, tzinfo=UTC),
        reopened_count=1,
        session_key="11111111-1111-4111-8111-111111111111",
        session_metadata=None,
        resolved_at=None,
    )


class _Coordinator:
    @asynccontextmanager
    async def hold(self, session_id):
        yield


def _coordinator() -> SessionCoordinator:
    return cast(SessionCoordinator, _Coordinator())


class TestReviewQueueEndpoint:
    async def test_maps_sampled_sessions_with_preview(self, repo: Mock) -> None:
        repo.sample_review_queue.return_value = [_chat_row()]
        repo.last_user_message.return_value = "x" * 250

        result = await sessions_api.get_review_queue(
            db=AsyncMock(), coordinator=_coordinator(), current_user=Mock()
        )

        item = result["items"][0]
        assert item["session_id"] == 7
        assert item["reopened_count"] == 1
        assert len(item["last_user_message"]) == 200  # bounded preview
        assert "window_start" in result

    async def test_default_window_is_seven_days(self, repo: Mock) -> None:
        before = datetime.now(UTC)

        await sessions_api.get_review_queue(
            db=AsyncMock(), coordinator=_coordinator(), current_user=Mock()
        )

        since = repo.sample_review_queue.await_args.kwargs["since"]
        assert before - timedelta(days=7) <= since <= datetime.now(UTC) - timedelta(days=7)

    async def test_limit_param_forwarded(self, repo: Mock) -> None:
        await sessions_api.get_review_queue(
            limit=5, db=AsyncMock(), coordinator=_coordinator(), current_user=Mock()
        )

        assert repo.sample_review_queue.await_args.kwargs["limit"] == 5

    async def test_strict_mode_requires_admin_for_queue(
        self, repo: Mock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "DEMO_MODE", False)

        with pytest.raises(HTTPException) as exc:
            await sessions_api.get_review_queue(
                db=AsyncMock(), coordinator=_coordinator(), current_user=Mock(is_admin=False)
            )

        assert exc.value.status_code == 403


class TestReviewVerdictEndpoint:
    async def test_verdict_echoes_recorded_state(self, repo: Mock) -> None:
        repo.record_review_verdict.return_value = Mock(
            id=7,
            resolved_at=datetime(2026, 10, 3, tzinfo=UTC),
            session_key="11111111-1111-4111-8111-111111111111",
        )

        result = await sessions_api.review_session(
            7,
            SessionReviewVerdict(
                resolved=True,
                note="确认",
                expected_session_key="11111111-1111-4111-8111-111111111111",
                expected_review_version="a" * 64,
            ),
            db=AsyncMock(),
            coordinator=_coordinator(),
            current_user=Mock(),
        )

        assert result["session_id"] == 7
        assert result["resolved"] is True
        assert result["resolved_at"] is not None
        assert repo.record_review_verdict.await_args.kwargs["resolved"] is True

    async def test_missing_session_404(self, repo: Mock) -> None:
        repo.record_review_verdict.return_value = None

        with pytest.raises(HTTPException) as exc:
            await sessions_api.review_session(
                999,
                SessionReviewVerdict(
                    resolved=False,
                    expected_session_key="11111111-1111-4111-8111-111111111111",
                    expected_review_version="a" * 64,
                ),
                db=AsyncMock(),
                coordinator=_coordinator(),
                current_user=Mock(),
            )

        assert exc.value.status_code == 404

    async def test_unresolved_verdict_keeps_resolved_at_null(self, repo: Mock) -> None:
        repo.record_review_verdict.return_value = Mock(
            id=7, resolved_at=None, session_key="11111111-1111-4111-8111-111111111111"
        )

        result = await sessions_api.review_session(
            7,
            SessionReviewVerdict(
                resolved=False,
                expected_session_key="11111111-1111-4111-8111-111111111111",
                expected_review_version="a" * 64,
            ),
            db=AsyncMock(),
            coordinator=_coordinator(),
            current_user=Mock(),
        )

        assert result["resolved"] is False
        assert result["resolved_at"] is None  # not a resolution — no KPI inflation

    async def test_strict_mode_requires_admin_for_verdict(
        self, repo: Mock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "DEMO_MODE", False)

        with pytest.raises(HTTPException) as exc:
            await sessions_api.review_session(
                7,
                SessionReviewVerdict(resolved=True),
                db=AsyncMock(),
                coordinator=_coordinator(),
                current_user=Mock(is_admin=False),
            )

        assert exc.value.status_code == 403
