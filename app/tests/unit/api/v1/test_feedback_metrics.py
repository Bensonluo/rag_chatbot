"""Feedback counter wiring: every submitted rating becomes measurable.

The feedback endpoint is the user-verified quality signal — the closest
operational proxy to the one-shot-resolution north star (an answer the
user downvoted was not a resolution). These tests pin that a successful
submission increments the labeled counter and a 404 (rating a message
that does not exist) does not pollute it.
"""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from prometheus_client import REGISTRY

from app.api.v1 import feedback as feedback_api
from app.models.schemas.feedback import FeedbackCreate


def _ratings(rating: str) -> float:
    value = REGISTRY.get_sample_value("feedback_ratings_total", {"rating": rating})
    return value if value is not None else 0.0


def _message(message_id: int, rating: int) -> Mock:
    message = Mock()
    message.id = message_id
    message.user_rating = rating
    return message


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch) -> Mock:
    repo = Mock()
    repo.submit_feedback = AsyncMock()
    monkeypatch.setattr(feedback_api, "FeedbackRepository", lambda db: repo)
    return repo


class TestFeedbackMetric:
    async def test_successful_rating_increments_counter(self, repo: Mock) -> None:
        repo.submit_feedback.return_value = _message(5, -1)
        before = _ratings("-1")
        result = await feedback_api.submit_feedback(
            feedback=FeedbackCreate(message_id=5, rating=-1),
            db=Mock(),
            current_user=Mock(),
        )
        assert result.success is True
        assert _ratings("-1") == before + 1.0

    async def test_up_rating_increments_its_own_label(self, repo: Mock) -> None:
        repo.submit_feedback.return_value = _message(6, 1)
        before = _ratings("1")
        await feedback_api.submit_feedback(
            feedback=FeedbackCreate(message_id=6, rating=1),
            db=Mock(),
            current_user=Mock(),
        )
        assert _ratings("1") == before + 1.0

    async def test_missing_message_is_404_and_uncounted(self, repo: Mock) -> None:
        repo.submit_feedback.return_value = None
        before = _ratings("-1")
        with pytest.raises(HTTPException) as exc:
            await feedback_api.submit_feedback(
                feedback=FeedbackCreate(message_id=999, rating=-1),
                db=Mock(),
                current_user=Mock(),
            )
        assert exc.value.status_code == 404
        assert _ratings("-1") == before
