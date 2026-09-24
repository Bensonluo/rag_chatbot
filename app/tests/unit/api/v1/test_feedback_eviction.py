"""Feedback → pyramid eviction wiring at the API boundary.

A downvote is the user saying "this answer was not a resolution". The
API must translate it into eviction of that response from the cache
pyramid (L0 exact + L1 semantic) — otherwise the pyramid keeps
replaying the rejected answer to every near-duplicate ask until
TTL/epoch rotation. An upvote must NOT evict: it is a confirmation,
and evicting on it would drain the pyramid's hottest entries.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from app.api.v1 import feedback as feedback_api
from app.models.schemas.feedback import FeedbackCreate


def _message(message_id: int, rating: int, content: str) -> Mock:
    message = Mock()
    message.id = message_id
    message.user_rating = rating
    message.content = content
    return message


@pytest.fixture
def repo(monkeypatch: pytest.MonkeyPatch) -> Mock:
    repo = Mock()
    repo.submit_feedback = AsyncMock()
    monkeypatch.setattr(feedback_api, "FeedbackRepository", lambda db: repo)
    return repo


@pytest.fixture
def evictor(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock(return_value=2)
    monkeypatch.setattr("app.services.chat.cache_eviction.evict_downvoted_response", mock)
    return mock


class TestDownvoteEvictsCachedAnswer:
    async def test_downvote_calls_eviction_with_the_answer_text(
        self, repo: Mock, evictor: AsyncMock
    ) -> None:
        repo.submit_feedback.return_value = _message(7, -1, "这条答案答非所问")
        result = await feedback_api.submit_feedback(
            feedback=FeedbackCreate(message_id=7, rating=-1),
            db=Mock(),
            current_user=Mock(),
        )
        assert result.success is True
        evictor.assert_awaited_once_with("这条答案答非所问")

    async def test_upvote_does_not_evict(self, repo: Mock, evictor: AsyncMock) -> None:
        repo.submit_feedback.return_value = _message(8, 1, "很好的答案")
        await feedback_api.submit_feedback(
            feedback=FeedbackCreate(message_id=8, rating=1),
            db=Mock(),
            current_user=Mock(),
        )
        evictor.assert_not_awaited()

    async def test_missing_message_404_does_not_evict(self, repo: Mock, evictor: AsyncMock) -> None:
        from fastapi import HTTPException

        repo.submit_feedback.return_value = None
        with pytest.raises(HTTPException) as exc:
            await feedback_api.submit_feedback(
                feedback=FeedbackCreate(message_id=999, rating=-1),
                db=Mock(),
                current_user=Mock(),
            )
        assert exc.value.status_code == 404
        evictor.assert_not_awaited()
