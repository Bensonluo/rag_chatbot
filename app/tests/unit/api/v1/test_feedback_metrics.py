"""Feedback counter wiring: every submitted rating becomes measurable.

The feedback endpoint is the user-verified quality signal — the closest
operational proxy to the containment north star (an answer the
user downvoted was not a resolution). These tests pin that a successful
submission increments the labeled counter and a 404 (rating a message
that does not exist) does not pollute it.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from prometheus_client import REGISTRY

from app.api.v1 import feedback as feedback_api
from app.models.schemas.feedback import FeedbackCreate
from app.repositories.feedback_repository import FeedbackReceipt, FeedbackTarget
from app.services.chat.coordination import SessionCoordinator

_KEY = "10000000-0000-4000-8000-000000000001"


def _ratings(rating: str) -> float:
    value = REGISTRY.get_sample_value("feedback_ratings_total", {"rating": rating})
    return value if value is not None else 0.0


def _message(message_id: int, rating: int) -> FeedbackReceipt:
    return FeedbackReceipt(message_id, 17, _KEY, rating, "answer")


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch) -> Mock:
    repo = Mock()
    repo.get_feedback_target = AsyncMock(return_value=FeedbackTarget(5, 17, _KEY, False))
    repo.submit_feedback = AsyncMock()
    monkeypatch.setattr(feedback_api, "FeedbackRepository", lambda db: repo)
    return repo


@pytest.fixture
def coordinator() -> SessionCoordinator:
    @asynccontextmanager
    async def hold(session_id: int) -> AsyncIterator[None]:
        assert session_id == 17
        yield

    return cast(SessionCoordinator, Mock(hold=hold))


class TestFeedbackMetric:
    async def test_successful_rating_increments_counter(
        self, repo: Mock, coordinator: SessionCoordinator
    ) -> None:
        repo.submit_feedback.return_value = _message(5, -1)
        before = _ratings("-1")
        result = await feedback_api.submit_feedback(
            feedback=FeedbackCreate(message_id=5, rating=-1),
            db=AsyncMock(),
            current_user=Mock(id=1),
            coordinator=coordinator,
        )
        assert result.success is True
        assert _ratings("-1") == before + 1.0

    async def test_up_rating_increments_its_own_label(
        self, repo: Mock, coordinator: SessionCoordinator
    ) -> None:
        repo.submit_feedback.return_value = _message(6, 1)
        before = _ratings("1")
        await feedback_api.submit_feedback(
            feedback=FeedbackCreate(message_id=6, rating=1),
            db=AsyncMock(),
            current_user=Mock(id=1),
            coordinator=coordinator,
        )
        assert _ratings("1") == before + 1.0

    async def test_missing_message_is_404_and_uncounted(
        self, repo: Mock, coordinator: SessionCoordinator
    ) -> None:
        repo.submit_feedback.return_value = None
        before = _ratings("-1")
        with pytest.raises(HTTPException) as exc:
            await feedback_api.submit_feedback(
                feedback=FeedbackCreate(message_id=999, rating=-1),
                db=AsyncMock(),
                current_user=Mock(id=1),
                coordinator=coordinator,
            )
        assert exc.value.status_code == 404
        assert _ratings("-1") == before
