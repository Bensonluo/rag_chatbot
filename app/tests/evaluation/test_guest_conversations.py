"""Real cookie-authenticated visitor requests stay in their owned conversation.

Reuses the local JWT/SQL/graph fixture, but exercises a new FastAPI guest
boundary with independent HTTPS cookie jars. No authentication result or
business result is substituted; external sockets and DNS are prohibited.
"""

import json
import socket
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event, select, update
from sqlalchemy.exc import OperationalError

from app.api.deps import get_db
from app.api.v1 import chat as chat_api
from app.api.v1 import feedback as feedback_api
from app.api.v1 import guest as guest_api
from app.api.v1 import sessions as sessions_api
from app.config.settings import settings
from app.core.security import create_access_token
from app.models.database.commerce import PurchaseSession
from app.models.database.guest import GuestVisitor
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.models.enums.message import MessageRole
from app.services.chat.coordination import get_session_coordinator
from app.services.chat.metrics import FEEDBACK_RATINGS
from app.services.commerce.service import PurchaseService
from app.services.guest_identity import GuestIdentityService, GuestIdentityUnavailableError
from app.tests.evaluation.test_human_conversation import _bot
from app.tests.evaluation.test_human_conversation import _Case as Case
from app.tests.evaluation.test_human_conversation import case as case

_BROWSER = {"Origin": "http://localhost:3000", "X-Rag-Guest": "1"}


@pytest.fixture(autouse=True)
def no_external_dns(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    attempts: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        attempts.append("DNS")
        raise AssertionError("Guest conversation evaluation forbids DNS")

    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        monkeypatch.setattr(socket, name, forbidden)
    yield
    assert attempts == []


@dataclass
class GuestHttp:
    case: Case
    first: AsyncClient
    second: AsyncClient
    identity: GuestIdentityService
    writes: list[str]


@pytest.fixture
async def visitors(case: Case, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[GuestHttp, None]:
    monkeypatch.setattr(settings, "GUEST_COOKIE_SECURE", True)
    monkeypatch.setattr(settings, "CORS_ORIGINS", ["http://localhost:3000"])
    app = FastAPI()
    for router in (guest_api.router, chat_api.router, sessions_api.router, feedback_api.router):
        app.include_router(router, prefix="/api/v1")

    async def local_db() -> AsyncGenerator[Any, None]:
        async with case.maker() as db:
            yield db

    identity = GuestIdentityService(case.maker)
    app.dependency_overrides[get_db] = local_db
    app.dependency_overrides[feedback_api._get_db] = local_db
    app.dependency_overrides[chat_api.get_chat_service] = lambda: case.chat
    app.dependency_overrides[guest_api.get_guest_chat_service] = lambda: case.chat
    app.dependency_overrides[get_session_coordinator] = lambda: case.coordinator
    app.dependency_overrides[sessions_api.get_session_chat_service] = lambda: case.chat
    app.dependency_overrides[guest_api.get_guest_identity_service] = lambda: identity
    writes: list[str] = []
    engine = case.maker.kw["bind"].sync_engine

    def observe(_connection, _cursor, statement, _parameters, _context, _many):
        verb = statement.lstrip().split(None, 1)[0].upper()
        if verb in {"INSERT", "UPDATE", "DELETE", "REPLACE"}:
            writes.append(verb)

    event.listen(engine, "before_cursor_execute", observe)
    try:
        async with (
            AsyncClient(
                transport=ASGITransport(app=app), base_url="https://guest-evaluation.invalid"
            ) as first,
            AsyncClient(
                transport=ASGITransport(app=app), base_url="https://guest-evaluation.invalid"
            ) as second,
        ):
            yield GuestHttp(case, first, second, identity, writes)
    finally:
        event.remove(engine, "before_cursor_execute", observe)


async def _bootstrap(client: AsyncClient) -> dict[str, Any]:
    response = await client.post("/api/v1/guest/bootstrap", headers=_BROWSER, json={})
    assert response.status_code == 200
    value = response.json()
    assert isinstance(value, dict)
    return value


@pytest.fixture
def shopping(visitors: GuestHttp) -> GuestHttp:
    app = cast(ASGITransport, visitors.first._transport).app
    assert isinstance(app, FastAPI)
    purchases = PurchaseService(visitors.case.maker)
    app.dependency_overrides[guest_api.get_guest_purchase_service] = lambda: purchases
    return visitors


def _body(bootstrap: dict[str, Any], message: str = "Hello") -> dict[str, Any]:
    return {
        "session_id": bootstrap["session"]["id"],
        "expected_session_key": bootstrap["session"]["session_key"],
        "message": message,
    }


def _scope(bootstrap: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": bootstrap["session"]["id"],
        "expected_session_key": bootstrap["session"]["session_key"],
    }


async def _snapshot(visitors: GuestHttp) -> dict[str, Any]:
    async with visitors.case.maker() as db:
        guests = (
            await db.execute(
                select(
                    GuestVisitor.id,
                    GuestVisitor.user_id,
                    GuestVisitor.current_session_id,
                    GuestVisitor.expires_at,
                    GuestVisitor.token_hash,
                ).order_by(GuestVisitor.id)
            )
        ).all()
        users = (
            await db.execute(
                select(
                    User.id,
                    User.is_guest,
                    User.email,
                    User.hashed_password,
                    User.is_active,
                    User.is_admin,
                ).order_by(User.id)
            )
        ).all()
        sessions = (
            await db.execute(
                select(
                    ChatSession.id,
                    ChatSession.user_id,
                    ChatSession.session_key,
                    ChatSession.automation_epoch,
                    ChatSession.resolved_at,
                    ChatSession.reopened_count,
                    ChatSession.session_metadata,
                ).order_by(ChatSession.id)
            )
        ).all()
        messages = (
            await db.execute(
                select(
                    Message.id,
                    Message.session_id,
                    Message.role,
                    Message.status,
                    Message.content,
                    Message.user_rating,
                    Message.feedback_text,
                ).order_by(Message.id)
            )
        ).all()
    return {
        "guests": tuple(tuple(row) for row in guests),
        "users": tuple(tuple(row) for row in users),
        "sessions": tuple(tuple(row) for row in sessions),
        "messages": tuple(tuple(row) for row in messages),
        "writes": len(visitors.writes),
        "graph": visitors.case.calls.graph,
        "cache_epoch": visitors.case.calls.cache_epoch,
        "tools": len(visitors.case.calls.tool_users),
        "ratings": tuple(
            FEEDBACK_RATINGS.labels(rating=str(value))._value.get() for value in (-1, 0, 1)
        ),
        "evictions": tuple(visitors.case.calls.evictions),
    }


@pytest.mark.parametrize("operation", ["chat", "stream", "history", "clear"])
async def test_guest_cookie_cannot_target_legacy_shared_demo_conversation(
    visitors: GuestHttp, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    assert (await _bot(visitors.case, "Hello", session_id=10)).status_code == 200
    await _bootstrap(visitors.first)
    async with visitors.case.maker() as db:
        shared = await db.get(ChatSession, 10)
        assert shared is not None and not shared.is_private
        key = shared.session_key
    before = await _snapshot(visitors)
    scope: dict[str, Any] = {"session_id": 10, "expected_session_key": key}
    if operation in {"chat", "stream"}:
        response = await visitors.first.post(
            "/api/v1/guest/chat" + ("/stream" if operation == "stream" else ""),
            headers=_BROWSER,
            json={**scope, "message": "Guest-private content must never enter the public demo"},
        )
    elif operation == "history":
        response = await visitors.first.get("/api/v1/guest/history", headers=_BROWSER, params=scope)
    else:
        response = await visitors.first.delete(
            "/api/v1/guest/history", headers=_BROWSER, params=scope
        )
    assert response.status_code == 404
    assert await _snapshot(visitors) == before


async def test_cookie_bootstrap_is_private_secret_free_and_reuses_without_rotation(
    visitors: GuestHttp,
) -> None:
    response = await visitors.first.post("/api/v1/guest/bootstrap", headers=_BROWSER, json={})
    assert response.status_code == 200
    attributes = {part.strip().lower() for part in response.headers["set-cookie"].split(";")[1:]}
    assert {"httponly", "secure", "samesite=lax", "path=/"} <= attributes
    assert not any(value.startswith(("max-age=", "expires=")) for value in attributes)
    assert "no-store" in response.headers["cache-control"]
    value = response.json()
    assert set(value) == {"visitor_id", "expires_at", "session"}
    assert set(value["session"]) == {"id", "session_key"}
    assert str(UUID(value["visitor_id"])) == value["visitor_id"]
    assert str(UUID(value["session"]["session_key"])) == value["session"]["session_key"]
    raw = visitors.first.cookies.get(guest_api.COOKIE_NAME)
    assert isinstance(raw, str) and len(raw) == 43 and raw not in response.text
    assert not any(
        field in response.text for field in ('"email"', '"password"', '"token"', '"user_id"')
    )
    async with visitors.case.maker() as db:
        visitor = await db.get(GuestVisitor, value["visitor_id"])
        assert visitor is not None
        user = await db.get(User, visitor.user_id)
        chat = await db.get(ChatSession, value["session"]["id"])
        assert user is not None and chat is not None
        assert user.is_guest and user.email is None and user.hashed_password is None
        assert not user.is_admin and chat.is_private and chat.user_id == user.id
    before = await _snapshot(visitors)
    reused = await visitors.first.post("/api/v1/guest/bootstrap", headers=_BROWSER, json={})
    assert reused.status_code == 200 and reused.json() == value
    assert "set-cookie" not in reused.headers
    assert await _snapshot(visitors) == before


async def test_independent_cookie_jars_get_distinct_visitors_and_cookie_possession_is_identity(
    visitors: GuestHttp,
) -> None:
    first, second = await _bootstrap(visitors.first), await _bootstrap(visitors.second)
    assert first["visitor_id"] != second["visitor_id"]
    assert first["session"]["id"] != second["session"]["id"]
    assert first["session"]["session_key"] != second["session"]["session_key"]
    first_secret = visitors.first.cookies.get(guest_api.COOKIE_NAME)
    second_secret = visitors.second.cookies.get(guest_api.COOKIE_NAME)
    assert first_secret and second_secret and first_secret != second_secret
    before = await _snapshot(visitors)
    visitors.second.cookies.clear()
    visitors.second.cookies.set(guest_api.COOKIE_NAME, first_secret)
    restored = await _bootstrap(visitors.second)
    assert restored == first
    assert await _snapshot(visitors) == before


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_real_guest_chat_and_history_are_own_only_with_original_completion_scope(
    visitors: GuestHttp, transport: str
) -> None:
    own = await _bootstrap(visitors.first)
    await _bootstrap(visitors.second)
    response = await visitors.first.post(
        "/api/v1/guest/chat" + ("/stream" if transport == "stream" else ""),
        headers=_BROWSER,
        json={**_body(own), "include_completion": True},
    )
    assert response.status_code == 200
    if transport == "normal":
        receipt = response.json()
        assert receipt["content"].strip()
    else:
        assert response.text.endswith("data: [DONE]\n\n")
        frames = [
            json.loads(frame.split("data: ", 1)[1])
            for frame in response.text.split("\n\n")
            if frame.startswith("event: completion\n")
        ]
        assert len(frames) == 1, response.text
        receipt = frames[0]
    assert receipt["session_id"] == own["session"]["id"]
    assert receipt["session_key"] == own["session"]["session_key"]
    assert receipt["status"] == "completed" and type(receipt["message_id"]) is int
    async with visitors.case.maker() as db:
        row = await db.get(Message, receipt["message_id"])
        assert (
            row is not None
            and row.session_id == own["session"]["id"]
            and row.role == MessageRole.ASSISTANT
        )
    history = await visitors.first.get(
        "/api/v1/guest/history", headers=_BROWSER, params=_scope(own)
    )
    assert history.status_code == 200
    assert history.json()["session_key"] == own["session"]["session_key"]
    assert len(history.json()["messages"]) == 2
    assert history.json()["messages"][-1]["message_id"] == receipt["message_id"]
    before = await _snapshot(visitors)
    foreign_read = await visitors.second.get(
        "/api/v1/guest/history", headers=_BROWSER, params=_scope(own)
    )
    foreign_write = await visitors.second.post(
        "/api/v1/guest/chat", headers=_BROWSER, json=_body(own)
    )
    assert foreign_read.status_code == 404 and foreign_write.status_code == 404
    assert await _snapshot(visitors) == before


@pytest.mark.parametrize("credential", ["missing", "unknown", "expired"])
async def test_invalid_guest_credentials_reject_without_identity_or_message_writes(
    visitors: GuestHttp, credential: str
) -> None:
    own = await _bootstrap(visitors.first)
    answer = await visitors.first.post("/api/v1/guest/chat", headers=_BROWSER, json=_body(own))
    assert answer.status_code == 200
    if credential == "missing":
        visitors.first.cookies.clear()
    elif credential == "unknown":
        visitors.first.cookies.clear()
        visitors.first.cookies.set(guest_api.COOKIE_NAME, "a" * 43)
    else:
        async with visitors.case.maker() as db:
            await db.execute(
                update(GuestVisitor)
                .where(GuestVisitor.id == own["visitor_id"])
                .values(expires_at=datetime.now(UTC) - timedelta(seconds=1))
            )
            await db.commit()
    before = await _snapshot(visitors)
    responses = [
        await visitors.first.post("/api/v1/guest/chat", headers=_BROWSER, json=_body(own)),
        await visitors.first.get("/api/v1/guest/history", headers=_BROWSER, params=_scope(own)),
        await visitors.first.post(
            "/api/v1/guest/feedback",
            headers=_BROWSER,
            json={
                "message_id": answer.json()["message_id"],
                "rating": -1,
                "expected_session_key": own["session"]["session_key"],
            },
        ),
    ]
    assert all(response.status_code == 401 for response in responses)
    assert all("set-cookie" not in response.headers for response in responses)
    assert all("no-store" in response.headers["cache-control"] for response in responses)
    assert await _snapshot(visitors) == before


@pytest.mark.parametrize("violation", ["missing_marker", "missing_origin", "evil_origin"])
async def test_guest_bootstrap_and_authenticated_write_require_browser_csrf_boundary(
    visitors: GuestHttp, violation: str
) -> None:
    own = await _bootstrap(visitors.first)
    headers = dict(_BROWSER)
    if violation == "missing_marker":
        headers.pop("X-Rag-Guest")
    elif violation == "missing_origin":
        headers.pop("Origin")
    else:
        headers["Origin"] = "https://localhost.evil.invalid"
    before = await _snapshot(visitors)
    creation = await visitors.second.post("/api/v1/guest/bootstrap", headers=headers, json={})
    mutation = await visitors.first.post("/api/v1/guest/chat", headers=headers, json=_body(own))
    assert creation.status_code == 403 and mutation.status_code == 403
    assert "set-cookie" not in creation.headers
    assert await _snapshot(visitors) == before


@pytest.mark.parametrize("demo", [False, True])
async def test_guest_cookie_or_body_actor_cannot_bypass_standard_private_endpoints(
    visitors: GuestHttp, monkeypatch: pytest.MonkeyPatch, demo: bool
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", demo)
    own = await _bootstrap(visitors.first)
    async with visitors.case.maker() as db:
        visitor = await db.get(GuestVisitor, own["visitor_id"])
        assert visitor is not None
        actor_id = visitor.user_id
    before = await _snapshot(visitors)
    bare = await visitors.first.post("/api/v1/chat", json={**_body(own), "user_id": actor_id})
    history = await visitors.first.get("/api/v1/chat/history", params=_scope(own))
    detail = await visitors.first.get(f"/api/v1/sessions/{own['session']['id']}")
    assert bare.status_code == 401 and history.status_code == 401 and detail.status_code == 401
    forged_body = await visitors.first.post(
        "/api/v1/guest/chat", headers=_BROWSER, json={**_body(own), "user_id": 1}
    )
    assert forged_body.status_code == 422
    registered = await visitors.first.post(
        "/api/v1/chat", headers=visitors.case.headers[1], json={**_body(own), "user_id": actor_id}
    )
    assert registered.status_code == (404 if demo else 403)
    guest_as_account = create_access_token({"sub": "guest-has-no-account", "user_id": actor_id})
    forged_account = await visitors.first.get(
        f"/api/v1/sessions/{own['session']['id']}",
        headers={"Authorization": "Bearer " + guest_as_account},
    )
    assert forged_account.status_code == 401
    assert await _snapshot(visitors) == before


async def test_clear_rotates_key_and_original_feedback_cannot_rate_reused_guest_answer(
    visitors: GuestHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    own = await _bootstrap(visitors.first)
    old = await visitors.first.post("/api/v1/guest/chat", headers=_BROWSER, json=_body(own))
    assert old.status_code == 200
    message_id = old.json()["message_id"]
    clear = await visitors.first.delete(
        "/api/v1/guest/history", headers=_BROWSER, params=_scope(own)
    )
    assert clear.status_code == 204
    async with visitors.case.maker() as db:
        row = await db.get(ChatSession, own["session"]["id"])
        assert row is not None
        assert row.session_key != own["session"]["session_key"] and row.automation_epoch == 1
    before = await _snapshot(visitors)
    stale = await visitors.first.post("/api/v1/guest/chat", headers=_BROWSER, json=_body(own))
    stale_history = await visitors.first.get(
        "/api/v1/guest/history", headers=_BROWSER, params=_scope(own)
    )
    assert stale.status_code == 409 and stale_history.status_code == 409
    assert await _snapshot(visitors) == before
    current = await _bootstrap(visitors.first)
    assert current["visitor_id"] == own["visitor_id"]
    assert current["session"]["id"] == own["session"]["id"]
    assert current["session"]["session_key"] != own["session"]["session_key"]
    new = await visitors.first.post("/api/v1/guest/chat", headers=_BROWSER, json=_body(current))
    assert new.status_code == 200 and new.json()["message_id"] == message_id
    before = await _snapshot(visitors)
    feedback = await visitors.first.post(
        "/api/v1/guest/feedback",
        headers=_BROWSER,
        json={
            "message_id": message_id,
            "rating": -1,
            "expected_session_key": own["session"]["session_key"],
        },
    )
    assert feedback.status_code == 409
    assert await _snapshot(visitors) == before


@pytest.mark.parametrize("demo", [False, True])
async def test_guest_feedback_checks_own_instance_and_never_inherits_shared_demo_scope(
    visitors: GuestHttp, monkeypatch: pytest.MonkeyPatch, demo: bool
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", demo)
    own = await _bootstrap(visitors.first)
    await _bootstrap(visitors.second)
    answer = await visitors.first.post("/api/v1/guest/chat", headers=_BROWSER, json=_body(own))
    assert answer.status_code == 200
    body = {
        "message_id": answer.json()["message_id"],
        "rating": -1,
        "expected_session_key": own["session"]["session_key"],
        "text": "Guest assessment",
    }
    before = await _snapshot(visitors)
    foreign = await visitors.second.post("/api/v1/guest/feedback", headers=_BROWSER, json=body)
    assert foreign.status_code == 404 and await _snapshot(visitors) == before
    accepted = await visitors.first.post("/api/v1/guest/feedback", headers=_BROWSER, json=body)
    assert accepted.status_code == 200
    assert accepted.json() == {
        "success": True,
        "message_id": body["message_id"],
        "rating": -1,
        "session_id": own["session"]["id"],
        "session_key": own["session"]["session_key"],
    }
    async with visitors.case.maker() as db:
        message = await db.get(Message, body["message_id"])
        assert (
            message is not None
            and message.user_rating == -1
            and message.feedback_text == "Guest assessment"
        )
    assert len(visitors.case.calls.evictions) == len(before["evictions"]) + 1
    if demo:
        legacy = await _bot(visitors.case, "Hello", session_id=10)
        assert legacy.status_code == 200
        async with visitors.case.maker() as db:
            shared = await db.get(ChatSession, 10)
            assert shared is not None and not shared.is_private
            shared_key = shared.session_key
        before_shared = await _snapshot(visitors)
        shared_feedback = await visitors.first.post(
            "/api/v1/guest/feedback",
            headers=_BROWSER,
            json={
                "message_id": legacy.json()["message_id"],
                "rating": -1,
                "expected_session_key": shared_key,
            },
        )
        assert shared_feedback.status_code == 404 and await _snapshot(visitors) == before_shared
        shared_control = await visitors.case.client.post(
            "/api/v1/feedback",
            headers=visitors.case.headers[1],
            json={"message_id": legacy.json()["message_id"], "rating": 1},
        )
        assert shared_control.status_code == 200


async def test_guest_feedback_cannot_fill_in_a_missing_displayed_key(visitors: GuestHttp) -> None:
    own = await _bootstrap(visitors.first)
    answer = await visitors.first.post("/api/v1/guest/chat", headers=_BROWSER, json=_body(own))
    assert answer.status_code == 200
    before = await _snapshot(visitors)
    response = await visitors.first.post(
        "/api/v1/guest/feedback",
        headers=_BROWSER,
        json={"message_id": answer.json()["message_id"], "rating": 1},
    )
    assert response.status_code == 409 and await _snapshot(visitors) == before


async def test_guest_bootstrap_cannot_accept_a_client_selected_owner_or_credential(
    visitors: GuestHttp,
) -> None:
    before = await _snapshot(visitors)
    response = await visitors.first.post(
        "/api/v1/guest/bootstrap",
        headers=_BROWSER,
        json={
            "user_id": 1,
            "session_id": visitors.case.private_id,
            "token": "client-selected-credential",
            "is_admin": True,
        },
    )
    assert response.status_code == 422
    assert "set-cookie" not in response.headers
    assert await _snapshot(visitors) == before


@pytest.mark.parametrize("operation", ["bootstrap", "chat"])
@pytest.mark.parametrize("fault", ["sql", "coordination"])
async def test_guest_unavailable_is_503_without_new_identity_or_cookie(
    visitors: GuestHttp, monkeypatch: pytest.MonkeyPatch, operation: str, fault: str
) -> None:
    own = await _bootstrap(visitors.first)
    before = await _snapshot(visitors)
    engine = visitors.case.maker.kw["bind"].sync_engine

    def refuse_read(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().startswith("SELECT") and "FROM guest_visitors" in statement:
            raise OperationalError(
                "guest credential lookup", {}, RuntimeError("synthetic unavailable")
            )

    async def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise GuestIdentityUnavailableError("synthetic coordination unavailable")

    if fault == "sql":
        event.listen(engine, "before_cursor_execute", refuse_read)
    else:
        monkeypatch.setattr(
            visitors.identity,
            "bootstrap" if operation == "bootstrap" else "authenticate",
            unavailable,
        )
    try:
        response = await visitors.first.post(
            "/api/v1/guest/bootstrap" if operation == "bootstrap" else "/api/v1/guest/chat",
            headers=_BROWSER,
            json={} if operation == "bootstrap" else _body(own),
        )
    finally:
        if fault == "sql":
            event.remove(engine, "before_cursor_execute", refuse_read)
    assert response.status_code == 503
    assert "set-cookie" not in response.headers
    assert await _snapshot(visitors) == before


async def test_guest_second_turn_preserves_the_displayed_keyboard_candidate(
    shopping: GuestHttp,
) -> None:
    own = await _bootstrap(shopping.first)
    shown = await shopping.first.post(
        "/api/v1/guest/chat", headers=_BROWSER, json=_body(own, "想买键盘")
    )
    assert shown.status_code == 200
    catalog = next(block for block in shown.json()["ui"]["blocks"] if block["type"] == "catalog")
    assert [item["sku_id"] for item in catalog["items"]] == ["KEYBOARD-87"]
    chosen = await shopping.first.post(
        "/api/v1/guest/chat", headers=_BROWSER, json=_body(own, "第一个")
    )
    assert chosen.status_code == 200
    cart = next(block for block in chosen.json()["ui"]["blocks"] if block["type"] == "cart")
    assert cart["cart"]["lines"][0]["sku_id"] == "KEYBOARD-87"
    assert chosen.json()["ui"]["purchase_id"] == shown.json()["ui"]["purchase_id"]


@pytest.mark.parametrize("message", ["黑色还是白色，哪个更适合？", "黑色 -2 个", "黑色 102 个"])
async def test_guest_ambiguous_choice_or_invalid_full_quantity_cannot_change_the_cart(
    shopping: GuestHttp, message: str
) -> None:
    own = await _bootstrap(shopping.first)
    shown = await shopping.first.post(
        "/api/v1/guest/chat", headers=_BROWSER, json=_body(own, "想买耳机")
    )
    assert shown.status_code == 200
    purchase_id = shown.json()["ui"]["purchase_id"]
    async with shopping.case.maker() as db:
        row = await db.get(PurchaseSession, purchase_id)
        assert row is not None
        before = (row.stage, row.version, row.sku_id, row.quantity, row.cart_snapshot)
    response = await shopping.first.post(
        "/api/v1/guest/chat", headers=_BROWSER, json=_body(own, message)
    )
    assert response.status_code == 200
    async with shopping.case.maker() as db:
        row = await db.get(PurchaseSession, purchase_id)
        assert row is not None
        assert (row.stage, row.version, row.sku_id, row.quantity, row.cart_snapshot) == before


async def test_guest_negated_black_selection_cannot_add_the_rejected_variant(
    shopping: GuestHttp,
) -> None:
    own = await _bootstrap(shopping.first)
    assert (
        await shopping.first.post(
            "/api/v1/guest/chat", headers=_BROWSER, json=_body(own, "想买耳机")
        )
    ).status_code == 200
    response = await shopping.first.post(
        "/api/v1/guest/chat", headers=_BROWSER, json=_body(own, "不要黑色，要白色2个")
    )
    assert response.status_code == 200
    view = response.json()["ui"]
    if view["stage"] == "cart":
        cart = next(block for block in view["blocks"] if block["type"] == "cart")
        assert (cart["cart"]["lines"][0]["sku_id"], cart["cart"]["lines"][0]["quantity"]) == (
            "ANC-WHITE",
            2,
        )
    else:
        assert view["stage"] == "shopping" and view["version"] == 0
        assert response.json()["content"].strip()


@pytest.mark.parametrize("message", ["black two pairs", "black quantity 2"])
async def test_guest_explicit_english_quantity_must_not_default_to_one(
    shopping: GuestHttp, message: str
) -> None:
    own = await _bootstrap(shopping.first)
    shown = await shopping.first.post(
        "/api/v1/guest/chat", headers=_BROWSER, json=_body(own, "想买耳机")
    )
    assert shown.status_code == 200
    view = shown.json()["ui"]
    catalog = next(block for block in view["blocks"] if block["type"] == "catalog")
    displayed = {item["sku_id"] for item in catalog["items"]}
    assert "ANC-BLACK" in displayed
    response = await shopping.first.post(
        "/api/v1/guest/chat", headers=_BROWSER, json=_body(own, message)
    )
    assert response.status_code == 200
    assert response.json()["ui"]["purchase_id"] == view["purchase_id"]
    async with shopping.case.maker() as db:
        row = await db.get(PurchaseSession, view["purchase_id"])
        assert row is not None
        if row.cart_snapshot is None:
            assert (row.stage, row.version, row.quantity) == ("shopping", 0, 0)
        else:
            assert row.sku_id in displayed
            assert (row.stage, row.sku_id, row.quantity) == ("cart", "ANC-BLACK", 2)


async def test_replayed_shopping_action_fences_a_refund_staged_after_the_first_click(
    shopping: GuestHttp,
) -> None:
    own = await _bootstrap(shopping.first)
    shown = await shopping.first.post(
        "/api/v1/guest/chat", headers=_BROWSER, json=_body(own, "想买耳机")
    )
    assert shown.status_code == 200
    view = shown.json()["ui"]
    action = next(item for item in view["actions"] if item["kind"] == "select_sku")
    body = {
        **_scope(own),
        "action_ref": action["action_ref"],
        "expected_version": action["expected_version"],
        "client_request_id": str(uuid4()),
        "input": {"quantity": 1},
    }
    url = f"/api/v1/guest/commerce/purchases/{view['purchase_id']}/actions"
    first = await shopping.first.post(url, headers=_BROWSER, json=body)
    assert first.status_code == 200
    async with shopping.case.maker() as db:
        row = await db.get(ChatSession, own["session"]["id"])
        assert row is not None
        epoch = row.automation_epoch
    for text in ("I want a refund", "ORD1001", "damaged item"):
        staged = await shopping.first.post(
            "/api/v1/guest/chat", headers=_BROWSER, json=_body(own, text)
        )
        assert staged.status_code == 200
    thread = str(own["session"]["id"]) + (f":automation:{epoch}" if epoch else "")
    state = await shopping.case.chat.graph.aget_state({"configurable": {"thread_id": thread}})
    assert state.values.get("pending_confirmation")
    replay = await shopping.first.post(
        url, headers=_BROWSER, json={**body, "client_request_id": str(uuid4())}
    )
    assert replay.status_code == 200
    assert (
        replay.json()["message_receipt"]["message_id"]
        == first.json()["message_receipt"]["message_id"]
    )
    async with shopping.case.maker() as db:
        row = await db.get(ChatSession, own["session"]["id"])
        assert row is not None and row.automation_epoch > epoch
    refunds = len(shopping.case.calls.refunds)
    confirmation = await shopping.first.post(
        "/api/v1/guest/chat", headers=_BROWSER, json=_body(own, "确认退款")
    )
    assert confirmation.status_code == 200
    assert len(shopping.case.calls.refunds) == refunds


async def test_guest_commerce_native_planner_failure_ends_with_an_explicit_stream_error(
    shopping: GuestHttp, monkeypatch: pytest.MonkeyPatch
) -> None:
    class UnavailablePlanner:
        async def generate_with_tools(self, messages: Any, tools: Any) -> Any:
            raise RuntimeError("isolated-native-planner-failure-detail")

    monkeypatch.setattr(shopping.case.chat, "llm_service", UnavailablePlanner())
    own = await _bootstrap(shopping.first)
    before = await _snapshot(shopping)
    response = await shopping.first.post(
        "/api/v1/guest/chat/stream",
        headers=_BROWSER,
        json={**_body(own, "想买耳机"), "include_completion": True},
    )
    assert response.status_code == 200
    assert response.text == "data: [STREAM_ERROR]\n\ndata: [DONE]\n\n"
    assert "isolated-native-planner-failure-detail" not in response.text
    assert "event: completion" not in response.text
    expected = dict(before)
    sessions = []
    for original in before["sessions"]:
        row = list(original)
        if row[0] == own["session"]["id"]:
            row[3] += 1
            metadata = dict(row[6] or {})
            metadata.pop("commerce_purchase_id", None)
            row[6] = metadata
        sessions.append(tuple(row))
    expected["sessions"] = tuple(sessions)
    expected["writes"] += 1
    assert await _snapshot(shopping) == expected
    async with shopping.case.maker() as db:
        assert (await db.scalars(select(PurchaseSession.id))).all() == []
