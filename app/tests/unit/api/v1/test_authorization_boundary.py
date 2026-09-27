"""Identity/session-ownership boundary (review 2026-09-26, #1).

Before this boundary the chat entries trusted the body-supplied
``user_id`` for anonymous callers, and the history read/clear
endpoints operated on any externally supplied ``session_id`` —
anyone could read or wipe another visitor's dialogue (OWASP BOLA).

One switch, DEMO_MODE, selects the posture:
- demo (default): today's public-demo behavior — anonymous visitors
  chat under a body user_id and read/clear history by bare session id.
- strict (DEMO_MODE=false): identity comes only from the token.
  Anonymous chat/history is 401; a body user_id contradicting the
  token is 403; history endpoints only touch sessions owned by the
  caller (foreign or missing both read as 404, so existence is not
  leaked); knowledge-base writes additionally require admin.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_current_user
from app.api.v1.chat import get_chat_service
from app.api.v1.sessions import get_session_service
from app.config.settings import settings
from app.core.exceptions import NotFoundError
from app.models.database.user import User


@pytest.fixture(autouse=True)
def _no_chat_rate_limit(monkeypatch: Any) -> None:
    """These tests exercise the authorization boundary, not the
    chat-path rate budget — skip the limiter so the shared in-memory
    bucket isn't drained for rate-limit-sensitive tests that run
    later in the suite."""
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)


def _user(uid: int = 7, *, admin: bool = False) -> User:
    return User(id=uid, email=f"u{uid}@test.local", is_active=True, is_admin=admin)


def _chat_service() -> Mock:
    service = Mock()
    service.process_message = AsyncMock(
        return_value=Mock(
            content="好的。",
            session_id=1,
            intent="greeting",
            sources=None,
            metadata=None,
        )
    )
    service.get_chat_history = AsyncMock(return_value=[])
    service.clear_chat_history = AsyncMock()
    return service


def _session_service(owned: bool) -> Mock:
    """Session service answering the ownership probe.

    owned=True → the session exists and belongs to the caller;
    owned=False → NotFoundError, the same answer a foreign or a
    missing session must give (existence is not leaked).
    """
    service = Mock()
    if owned:
        service.get_session_by_id = AsyncMock(return_value=Mock(user_id=7))
    else:
        service.get_session_by_id = AsyncMock(side_effect=NotFoundError("Session", "1"))
    return service


def _client(current_user: User | None, chat_service: Mock, session_service: Mock) -> TestClient:
    from app.main import create_app

    app = create_app()
    app.dependency_overrides[get_current_user] = lambda: current_user
    app.dependency_overrides[get_chat_service] = lambda: chat_service
    app.dependency_overrides[get_session_service] = lambda: session_service
    return TestClient(app)


class TestDemoModePosture:
    """DEMO_MODE=true keeps the public demo exactly as it was."""

    @pytest.fixture(autouse=True)
    def _demo_mode(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(settings, "DEMO_MODE", True)

    def test_anonymous_chat_uses_body_user_id(self) -> None:
        chat = _chat_service()
        client = _client(None, chat, _session_service(owned=True))
        response = client.post(
            "/api/v1/chat", json={"message": "你好", "session_id": 1, "user_id": 3}
        )
        assert response.status_code == 200
        chat.process_message.assert_awaited_once()
        assert chat.process_message.await_args.kwargs["user_id"] == 3

    def test_anonymous_history_read_skips_ownership_probe(self) -> None:
        sessions = _session_service(owned=False)
        client = _client(None, _chat_service(), sessions)
        response = client.get("/api/v1/chat/history?session_id=1")
        assert response.status_code == 200
        sessions.get_session_by_id.assert_not_awaited()

    def test_token_identity_wins_over_body_user_id(self) -> None:
        chat = _chat_service()
        client = _client(_user(7), chat, _session_service(owned=True))
        response = client.post(
            "/api/v1/chat", json={"message": "你好", "session_id": 1, "user_id": 3}
        )
        assert response.status_code == 200
        assert chat.process_message.await_args.kwargs["user_id"] == 7


class TestStrictModePosture:
    """DEMO_MODE=false: identity and ownership are enforced."""

    @pytest.fixture(autouse=True)
    def _strict_mode(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(settings, "DEMO_MODE", False)

    def test_anonymous_chat_rejected(self) -> None:
        client = _client(None, _chat_service(), _session_service(owned=True))
        response = client.post("/api/v1/chat", json={"message": "你好", "session_id": 1})
        assert response.status_code == 401

    def test_anonymous_stream_rejected(self) -> None:
        client = _client(None, _chat_service(), _session_service(owned=True))
        response = client.post("/api/v1/chat/stream", json={"message": "你好", "session_id": 1})
        assert response.status_code == 401

    def test_anonymous_history_read_rejected(self) -> None:
        client = _client(None, _chat_service(), _session_service(owned=True))
        response = client.get("/api/v1/chat/history?session_id=1")
        assert response.status_code == 401

    def test_anonymous_history_clear_rejected(self) -> None:
        client = _client(None, _chat_service(), _session_service(owned=True))
        response = client.delete("/api/v1/chat/history?session_id=1")
        assert response.status_code == 401

    def test_foreign_session_reads_as_not_found(self) -> None:
        chat = _chat_service()
        client = _client(_user(7), chat, _session_service(owned=False))
        response = client.get("/api/v1/chat/history?session_id=1")
        assert response.status_code == 404
        chat.get_chat_history.assert_not_awaited()

    def test_owner_reads_history(self) -> None:
        chat = _chat_service()
        client = _client(_user(7), chat, _session_service(owned=True))
        response = client.get("/api/v1/chat/history?session_id=1")
        assert response.status_code == 200
        chat.get_chat_history.assert_awaited_once()

    def test_owner_clears_history_after_ownership_probe(self) -> None:
        chat = _chat_service()
        sessions = _session_service(owned=True)
        client = _client(_user(7), chat, sessions)
        response = client.delete("/api/v1/chat/history?session_id=1")
        assert response.status_code == 204
        sessions.get_session_by_id.assert_awaited_once()
        chat.clear_chat_history.assert_awaited_once()

    def test_body_user_id_contradicting_token_rejected(self) -> None:
        client = _client(_user(7), _chat_service(), _session_service(owned=True))
        response = client.post(
            "/api/v1/chat", json={"message": "你好", "session_id": 1, "user_id": 3}
        )
        assert response.status_code == 403

    def test_body_user_id_matching_token_allowed(self) -> None:
        chat = _chat_service()
        client = _client(_user(7), chat, _session_service(owned=True))
        response = client.post(
            "/api/v1/chat", json={"message": "你好", "session_id": 1, "user_id": 7}
        )
        assert response.status_code == 200
        assert chat.process_message.await_args.kwargs["user_id"] == 7


class TestKbWriterGate:
    """Knowledge-base writes: admin in strict mode, any active user in demo."""

    def test_strict_non_admin_rejected(self, monkeypatch: Any) -> None:
        from fastapi import HTTPException

        from app.api.deps.authorization import require_kb_writer

        monkeypatch.setattr(settings, "DEMO_MODE", False)
        with pytest.raises(HTTPException) as exc:
            require_kb_writer(_user(7, admin=False))
        assert exc.value.status_code == 403

    def test_strict_admin_allowed(self, monkeypatch: Any) -> None:
        from app.api.deps.authorization import require_kb_writer

        monkeypatch.setattr(settings, "DEMO_MODE", False)
        require_kb_writer(_user(1, admin=True))  # no raise

    def test_demo_non_admin_allowed(self, monkeypatch: Any) -> None:
        from app.api.deps.authorization import require_kb_writer

        monkeypatch.setattr(settings, "DEMO_MODE", True)
        require_kb_writer(_user(7, admin=False))  # no raise
