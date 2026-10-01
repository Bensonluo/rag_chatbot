"""Feedback message ownership (review 2026-09-26, #1 remaining slice).

The feedback endpoint checked only that the rated message existed —
any authenticated caller could rate anyone's message, and a downvote
fires a cache eviction keyed on that message's content, so a foreign
rating could even evict another tenant's cached answers (OWASP BOLA).

One switch, DEMO_MODE, selects the posture (same as the history
endpoints in test_authorization_boundary.py):
- strict: the message must belong to the caller (via its session);
  a foreign and a missing id both answer 404 with the same wording
  so existence is not leaked, and the rating is never applied.
- demo: unchanged — existence check only, the demo DB's messages
  belong to seeded users.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_current_user
from app.api.v1 import feedback as feedback_api
from app.config.settings import settings
from app.models.database.user import User


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch: Any) -> None:
    """These tests exercise the ownership boundary, not the rate
    budget — skip the limiter so the shared in-memory bucket isn't
    drained for rate-limit-sensitive tests that run later."""
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)


def _user() -> User:
    return User(id=7, email="u7@test.local", is_active=True)


def _repo(*, owned: bool) -> Mock:
    """Repository stand-in: get_owned_message answers the ownership
    probe; submit_feedback applies the rating."""
    repo = Mock()
    repo.get_owned_message = AsyncMock(return_value=Mock() if owned else None)
    repo.submit_feedback = AsyncMock(return_value=Mock(id=5, user_rating=-1, content="answer text"))
    return repo


def _client(current_user: User | None, repo: Mock, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from app.main import create_app

    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: current_user
    # FeedbackRepository is a plain class the endpoint instantiates, not
    # a FastAPI dependency — patch the module attribute the call site
    # resolves (same seam as test_feedback_metrics.py).
    monkeypatch.setattr(feedback_api, "FeedbackRepository", lambda db: repo)
    return TestClient(app)


def _post_rating(client: TestClient, message_id: int = 5) -> Any:
    return client.post("/api/v1/feedback", json={"message_id": message_id, "rating": -1})


class TestStrictModeOwnership:
    @pytest.fixture(autouse=True)
    def _strict_mode(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(settings, "DEMO_MODE", False)

    def test_foreign_message_is_404_and_never_rated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = _repo(owned=False)
        client = _client(_user(), repo, monkeypatch)

        response = _post_rating(client)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
        # The rating is never applied and the downvote's cache
        # eviction never sees the foreign content.
        repo.submit_feedback.assert_not_awaited()

    def test_own_message_is_rated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = _repo(owned=True)
        client = _client(_user(), repo, monkeypatch)

        response = _post_rating(client)

        assert response.status_code == 200
        assert response.json()["success"] is True
        repo.get_owned_message.assert_awaited_once_with(5, 7)
        repo.submit_feedback.assert_awaited_once()

    def test_missing_message_404_wording_matches_foreign(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Foreign and missing must be indistinguishable (BOLA)."""
        repo = _repo(owned=True)
        repo.submit_feedback = AsyncMock(return_value=None)
        client = _client(_user(), repo, monkeypatch)

        response = _post_rating(client, message_id=999)

        assert response.status_code == 404
        assert response.json()["detail"] == "Message 999 not found"


class TestDemoModeUnchanged:
    @pytest.fixture(autouse=True)
    def _demo_mode(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(settings, "DEMO_MODE", True)

    def test_ownership_probe_never_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Demo posture: seeded users own the demo messages — strict
        ownership would break the public demo, so the probe is absent
        and the existence check alone decides."""
        repo = _repo(owned=False)
        client = _client(_user(), repo, monkeypatch)

        response = _post_rating(client)

        assert response.status_code == 200
        repo.get_owned_message.assert_not_awaited()
        repo.submit_feedback.assert_awaited_once()
