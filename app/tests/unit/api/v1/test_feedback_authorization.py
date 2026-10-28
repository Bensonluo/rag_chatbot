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
- demo: shared seeded rows remain rateable without a key; private rows
  still require their owner and matching server incarnation.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_current_user
from app.api.v1 import feedback as feedback_api
from app.config.settings import settings
from app.models.database.user import User
from app.repositories.feedback_repository import FeedbackReceipt, FeedbackTarget
from app.services.chat.coordination import SessionCoordinator, get_session_coordinator

_KEY = "10000000-0000-4000-8000-000000000001"


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch: Any) -> None:
    """These tests exercise the ownership boundary, not the rate
    budget — skip the limiter so the shared in-memory bucket isn't
    drained for rate-limit-sensitive tests that run later."""
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)


def _user() -> User:
    return User(id=7, email="u7@test.local", is_active=True)


def _repo(*, owned: bool, private: bool = True) -> Mock:
    """The read and conditional-write seams return concrete scalar snapshots."""
    repo = Mock()
    repo.get_feedback_target = AsyncMock(
        return_value=FeedbackTarget(5, 17, _KEY, private) if owned else None
    )
    repo.submit_feedback = AsyncMock(return_value=FeedbackReceipt(5, 17, _KEY, -1, "answer text"))
    return repo


def _client(current_user: User | None, repo: Mock, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from app.main import create_app

    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: current_user
    app.dependency_overrides[feedback_api._get_db] = lambda: AsyncMock()

    @asynccontextmanager
    async def hold(session_id: int) -> AsyncIterator[None]:
        assert session_id == 17
        yield

    coordinator = cast(SessionCoordinator, Mock(hold=hold))
    app.dependency_overrides[get_session_coordinator] = lambda: coordinator
    # FeedbackRepository is a plain class the endpoint instantiates, not
    # a FastAPI dependency — patch the module attribute the call site
    # resolves (same seam as test_feedback_metrics.py).
    monkeypatch.setattr(feedback_api, "FeedbackRepository", lambda db: repo)
    return TestClient(app)


def _post_rating(client: TestClient, message_id: int = 5, key: str | None = _KEY) -> Any:
    payload: dict[str, object] = {"message_id": message_id, "rating": -1}
    if key is not None:
        payload["expected_session_key"] = key
    return client.post("/api/v1/feedback", json=payload)


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
        assert response.json()["session_id"] == 17
        assert response.json()["session_key"] == _KEY
        assert repo.get_feedback_target.await_count == 2
        repo.get_feedback_target.assert_awaited_with(5, 7, allow_shared=False, session_id=17)
        repo.submit_feedback.assert_awaited_once_with(
            message_id=5,
            rating=-1,
            text=None,
            user_id=7,
            session_id=17,
            session_key=_KEY,
            allow_shared=False,
        )

    @pytest.mark.parametrize("key", [None, "20000000-0000-4000-8000-000000000002"])
    def test_private_missing_or_stale_key_is_not_rated(
        self, monkeypatch: pytest.MonkeyPatch, key: str | None
    ) -> None:
        repo = _repo(owned=True)
        response = _post_rating(_client(_user(), repo, monkeypatch), key=key)
        assert response.status_code == 409
        repo.submit_feedback.assert_not_awaited()

    def test_owner_removed_between_discovery_and_gate_is_not_rated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _repo(owned=True)
        repo.get_feedback_target.side_effect = [FeedbackTarget(5, 17, _KEY, True), None]
        response = _post_rating(_client(_user(), repo, monkeypatch))
        assert response.status_code == 404
        repo.submit_feedback.assert_not_awaited()

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


class TestDemoModeSharedAccess:
    @pytest.fixture(autouse=True)
    def _demo_mode(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(settings, "DEMO_MODE", True)

    def test_shared_access_probe_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Demo permits public seeded rows but still probes private access."""
        repo = _repo(owned=True, private=False)
        client = _client(_user(), repo, monkeypatch)

        response = _post_rating(client, key=None)

        assert response.status_code == 200
        assert repo.get_feedback_target.await_count == 2
        repo.get_feedback_target.assert_awaited_with(5, 7, allow_shared=True, session_id=17)
        repo.submit_feedback.assert_awaited_once()

    def test_explicit_shared_key_still_has_to_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = _repo(owned=True, private=False)
        response = _post_rating(
            _client(_user(), repo, monkeypatch), key="20000000-0000-4000-8000-000000000002"
        )
        assert response.status_code == 409
        repo.submit_feedback.assert_not_awaited()

    def test_keyless_shared_request_does_not_adopt_replacement_after_discovery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = _repo(owned=True, private=False)
        repo.get_feedback_target.side_effect = [
            FeedbackTarget(5, 17, _KEY, False),
            FeedbackTarget(5, 17, "20000000-0000-4000-8000-000000000002", False),
        ]
        response = _post_rating(_client(_user(), repo, monkeypatch), key=None)
        assert response.status_code == 409
        repo.submit_feedback.assert_not_awaited()
