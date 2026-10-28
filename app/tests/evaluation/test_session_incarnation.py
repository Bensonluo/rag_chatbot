"""Explicit conversation keys bind normal private workspace operations.

Uses the existing local JWT/HTTP/SQLite-FK fixture and its compiled graph.
Only deliberate cleanup failure is injected. No external model, service,
or prior cross-customer reproduction is exercised here.
"""

import socket
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from httpx import Response
from sqlalchemy import func, select

from app.models.database.action_execution import ActionExecution
from app.models.database.handoff_message import HandoffMessage
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.tests.evaluation.test_human_conversation import _bot
from app.tests.evaluation.test_human_conversation import _Case as Case
from app.tests.evaluation.test_human_conversation import case as case

_MISSING = object()


@pytest.fixture(autouse=True)
def no_external_dns(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    attempts: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        attempts.append("DNS")
        raise AssertionError("session-incarnation evaluation forbids DNS")

    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        monkeypatch.setattr(socket, name, forbidden)
    yield
    assert attempts == []


@dataclass(frozen=True)
class _Snapshot:
    key: str
    epoch: int
    resolved_at: datetime | None
    reopened: int
    bot_rows: int
    human_rows: int
    actions: int
    tickets: tuple[tuple[int, str, int | None], ...]


async def _key(case: Case, session_id: int | None = None) -> str:
    response = await case.client.get(
        f"/api/v1/sessions/{session_id or case.private_id}", headers=case.headers[1]
    )
    assert response.status_code == 200
    value = response.json()["session_key"]
    assert isinstance(value, str) and str(UUID(value)) == value
    return value


async def _snapshot(case: Case, session_id: int | None = None) -> _Snapshot:
    sid = session_id or case.private_id
    async with case.maker() as db:
        row = await db.get(ChatSession, sid)
        assert row is not None
        bot = await db.scalar(
            select(func.count()).select_from(Message).where(Message.session_id == sid)
        )
        human = await db.scalar(
            select(func.count()).select_from(HandoffMessage).where(HandoffMessage.session_id == sid)
        )
        actions = await db.scalar(
            select(func.count())
            .select_from(ActionExecution)
            .where(ActionExecution.scope_session_id == sid)
        )
        tickets = (
            await db.execute(
                select(HandoffTicket.id, HandoffTicket.status, HandoffTicket.assigned_to)
                .where(HandoffTicket.session_id == sid)
                .order_by(HandoffTicket.id)
            )
        ).all()
        assert isinstance(bot, int) and isinstance(human, int) and isinstance(actions, int)
        return _Snapshot(
            row.session_key,
            row.automation_epoch,
            row.resolved_at,
            row.reopened_count,
            bot,
            human,
            actions,
            tuple((identifier, status, assigned) for identifier, status, assigned in tickets),
        )


def _work_calls(case: Case) -> tuple[int, int, int, int]:
    return (
        case.calls.graph,
        case.calls.cache_epoch,
        len(case.calls.tool_users),
        len(case.calls.refunds),
    )


async def _ticket(case: Case, *, session_id: int | None = None, claim: bool = True) -> int:
    sid = session_id or case.private_id
    created = await case.handoff.create_ticket_for_session(
        sid, 1, "explicit", {"user_message": "Ordinary fixture support request"}
    )
    identifier = created["ticket_id"]
    assert type(identifier) is int
    if claim:
        response = await case.client.post(
            f"/api/v1/handoff/tickets/{identifier}/claim",
            params={"expected_session_key": await _key(case, sid)},
            headers=case.headers[3],
        )
        assert response.status_code == 200
        assert response.json()["session_key"] == await _key(case, sid)
    return identifier


def _human_url(case: Case, side: str, ticket_id: int, session_id: int | None = None) -> str:
    return (
        f"/api/v1/handoff/tickets/{ticket_id}/messages"
        if side == "staff"
        else f"/api/v1/handoff/sessions/{session_id or case.private_id}/messages"
    )


def _human_body(side: str, ticket_id: int, key: object = _MISSING) -> dict[str, Any]:
    body: dict[str, Any] = {
        "client_message_id": str(uuid4()),
        "content": "问题解决了",
    }
    if side == "staff":
        body["resolve"] = False
    else:
        body["expected_ticket_id"] = ticket_id
    if key is not _MISSING:
        body["expected_session_key"] = key
    return body


async def _ai(
    case: Case,
    transport: str,
    key: object = _MISSING,
    *,
    actor: int = 1,
    session_id: int | None = None,
) -> Response:
    body: dict[str, Any] = {"session_id": session_id or case.private_id, "message": "Hello"}
    if key is not _MISSING:
        body["expected_session_key"] = key
    return await case.client.post(
        "/api/v1/chat" + ("/stream" if transport == "stream" else ""),
        headers=case.headers.get(actor, {}),
        json=body,
    )


async def test_server_generated_key_is_unique_and_stable_for_normal_row_reads(case: Case) -> None:
    key = await _key(case)
    assert await _key(case) == key
    listing = await case.client.get("/api/v1/sessions", headers=case.headers[1])
    assert listing.status_code == 200
    assert (
        next(row for row in listing.json()["items"] if row["id"] == case.private_id)["session_key"]
        == key
    )
    renamed = await case.client.put(
        f"/api/v1/sessions/{case.private_id}",
        headers=case.headers[1],
        json={"title": "Renamed ordinary consultation", "expected_session_key": key},
    )
    assert renamed.status_code == 200 and renamed.json()["session_key"] == key
    created = await case.client.post(
        "/api/v1/sessions", headers=case.headers[1], json={"title": "Another ordinary consultation"}
    )
    assert created.status_code == 200
    new_key = created.json()["session_key"]
    assert isinstance(new_key, str) and str(UUID(new_key)) == new_key and new_key != key


async def test_normal_private_pages_ack_and_both_histories_identify_the_session(case: Case) -> None:
    key = await _key(case)
    normal = await _bot(case, "Hello")
    assert normal.status_code == 200
    ticket_id = await _ticket(case)
    for side, actor in (("customer", 1), ("staff", 3)):
        response = await case.client.post(
            _human_url(case, side, ticket_id),
            headers=case.headers[actor],
            json=_human_body(side, ticket_id, key),
        )
        assert response.status_code == 200 and response.json()["session_key"] == key
        page = await case.client.get(
            _human_url(case, side, ticket_id),
            params={"expected_session_key": key},
            headers=case.headers[actor],
        )
        assert page.status_code == 200 and page.json()["session_key"] == key
    for url, actor, params in (
        (f"/api/v1/handoff/tickets/{ticket_id}/history", 3, {}),
        ("/api/v1/chat/history", 1, {"session_id": case.private_id}),
    ):
        history = await case.client.get(
            url, params={**params, "expected_session_key": key}, headers=case.headers[actor]
        )
        assert history.status_code == 200 and history.json()["session_key"] == key


@pytest.mark.parametrize("side", ["customer", "staff"])
@pytest.mark.parametrize("kind", ["missing", "mismatch", "null", "malformed", "boolean", "array"])
async def test_human_key_rejection_has_no_persisted_or_resolution_side_effects(
    case: Case, side: str, kind: str
) -> None:
    ticket_id = await _ticket(case)
    value: object = {
        "missing": _MISSING,
        "mismatch": str(uuid4()),
        "null": None,
        "malformed": "invalid-key",
        "boolean": True,
        "array": [],
    }[kind]
    before = await _snapshot(case)
    calls = _work_calls(case)
    response = await case.client.post(
        _human_url(case, side, ticket_id),
        headers=case.headers[3 if side == "staff" else 1],
        json=_human_body(side, ticket_id, value),
    )
    assert response.status_code == (409 if kind == "mismatch" else 422)
    assert await _snapshot(case) == before
    assert _work_calls(case) == calls


@pytest.mark.parametrize("endpoint", ["customer-page", "staff-page", "staff-history", "ai-history"])
@pytest.mark.parametrize("kind", ["missing", "mismatch"])
async def test_private_reads_require_the_expected_key_before_returning_a_body(
    case: Case, endpoint: str, kind: str
) -> None:
    ticket_id = await _ticket(case)
    params: dict[str, Any] = {}
    if kind == "mismatch":
        params["expected_session_key"] = str(uuid4())
    if endpoint == "ai-history":
        url, actor = "/api/v1/chat/history", 1
        params["session_id"] = case.private_id
    elif endpoint == "staff-history":
        url, actor = f"/api/v1/handoff/tickets/{ticket_id}/history", 3
    else:
        side = "staff" if endpoint == "staff-page" else "customer"
        url, actor = _human_url(case, side, ticket_id), 3 if side == "staff" else 1
    before = await _snapshot(case)
    calls = _work_calls(case)
    response = await case.client.get(url, params=params, headers=case.headers[actor])
    assert response.status_code == 409
    assert "messages" not in response.json()
    assert await _snapshot(case) == before and _work_calls(case) == calls


@pytest.mark.parametrize("transport", ["normal", "stream"])
@pytest.mark.parametrize("kind", ["missing", "mismatch"])
async def test_private_ai_refuses_missing_or_old_key_before_graph_or_stream_start(
    case: Case, transport: str, kind: str
) -> None:
    before = await _snapshot(case)
    response = await _ai(case, transport, _MISSING if kind == "missing" else str(uuid4()))
    assert response.status_code == 409
    assert response.headers["content-type"].startswith("application/json")
    assert _work_calls(case) == (0, 0, 0, 0)
    assert await _snapshot(case) == before


@pytest.mark.parametrize(
    ("endpoint", "actor", "expected"),
    [
        ("customer", 0, 401),
        ("customer", 2, 404),
        ("customer", 3, 404),
        ("staff", 0, 401),
        ("staff", 1, 403),
        ("staff", 4, 404),
        ("ai", 0, 401),
        ("ai", 2, 404),
        ("ai", 3, 404),
    ],
)
async def test_existing_auth_and_assignment_boundary_precedes_key_mismatch(
    case: Case, endpoint: str, actor: int, expected: int
) -> None:
    ticket_id = await _ticket(case)
    before = await _snapshot(case)
    if endpoint == "ai":
        response = await _ai(case, "normal", str(uuid4()), actor=actor)
    else:
        response = await case.client.post(
            _human_url(case, endpoint, ticket_id),
            headers=case.headers.get(actor, {}),
            json=_human_body(endpoint, ticket_id, str(uuid4())),
        )
    assert response.status_code == expected
    assert await _snapshot(case) == before and _work_calls(case) == (0, 0, 0, 0)


@pytest.mark.parametrize("command", ["customer", "staff", "ai-normal", "ai-stream"])
async def test_normal_same_owner_numeric_id_reuse_does_not_reuse_the_old_session_key(
    case: Case, command: str
) -> None:
    created = await case.client.post(
        "/api/v1/sessions", headers=case.headers[1], json={"title": "Temporary ordinary session"}
    )
    assert created.status_code == 200
    sid, old_key = created.json()["id"], created.json()["session_key"]
    deleted = await case.client.delete(
        f"/api/v1/sessions/{sid}",
        params={"expected_session_key": old_key},
        headers=case.headers[1],
    )
    assert deleted.status_code == 204
    replacement = await case.client.post(
        "/api/v1/sessions", headers=case.headers[1], json={"title": "New ordinary session"}
    )
    assert replacement.status_code == 200
    assert replacement.json()["id"] == sid  # SQLite normal allocation, same owner/namespace.
    assert replacement.json()["session_key"] != old_key
    if command in {"customer", "staff"}:
        ticket_id = await _ticket(case, session_id=sid)
        before = await _snapshot(case, sid)
        response = await case.client.post(
            _human_url(case, command, ticket_id, sid),
            headers=case.headers[3 if command == "staff" else 1],
            json=_human_body(command, ticket_id, old_key),
        )
    else:
        before = await _snapshot(case, sid)
        response = await _ai(case, command.removeprefix("ai-"), old_key, session_id=sid)
    assert response.status_code == 409
    assert await _snapshot(case, sid) == before and _work_calls(case) == (0, 0, 0, 0)


async def test_history_clear_advances_key_and_epoch_and_refuses_an_old_human_command(
    case: Case,
) -> None:
    key = await _key(case)
    ticket_id = await _ticket(case, claim=False)
    original = _human_body("customer", ticket_id, key)
    original["content"] = "An ordinary note before history reset"
    first = await case.client.post(
        _human_url(case, "customer", ticket_id), headers=case.headers[1], json=original
    )
    assert first.status_code == 200
    before = await _snapshot(case)
    clear = await case.client.delete(
        "/api/v1/chat/history",
        params={"session_id": case.private_id, "expected_session_key": key},
        headers=case.headers[1],
    )
    assert clear.status_code == 204
    after = await _snapshot(case)
    assert after.key != key and after.epoch == before.epoch + 1
    assert after.human_rows == 0 and after.bot_rows == 0 and after.actions == before.actions
    retry = await case.client.post(
        _human_url(case, "customer", ticket_id), headers=case.headers[1], json=original
    )
    assert retry.status_code == 409 and await _snapshot(case) == after
    fresh = _human_body("customer", ticket_id, after.key)
    fresh["content"] = "A new ordinary note after reset"
    accepted = await case.client.post(
        _human_url(case, "customer", ticket_id), headers=case.headers[1], json=fresh
    )
    assert accepted.status_code == 200 and accepted.json()["session_key"] == after.key


async def test_clear_cleanup_failure_does_not_restore_the_invalidated_old_key(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = await _snapshot(case)

    async def unavailable_cleanup(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("isolated post-commit checkpoint cleanup failure")

    monkeypatch.setattr(case.chat, "discard_session_state", unavailable_cleanup)
    response = await case.client.delete(
        "/api/v1/chat/history",
        params={"session_id": case.private_id, "expected_session_key": old.key},
        headers=case.headers[1],
    )
    assert response.status_code in {500, 503}
    current = await _snapshot(case)
    assert current.key != old.key and current.epoch == old.epoch + 1
    assert await _key(case) == current.key
    refused = await _ai(case, "normal", old.key)
    assert refused.status_code == 409 and _work_calls(case) == (0, 0, 0, 0)


async def test_normal_staff_close_preserves_key_and_exact_original_receipt(case: Case) -> None:
    key = await _key(case)
    ticket_id = await _ticket(case)
    before = await _snapshot(case)
    body = _human_body("staff", ticket_id, key)
    body.update({"content": "An ordinary final handling note", "resolve": True})
    first = await case.client.post(
        _human_url(case, "staff", ticket_id), headers=case.headers[3], json=body
    )
    assert first.status_code == 200 and first.json()["session_key"] == key
    assert first.json()["ticket"]["status"] == "resolved"
    after = await _snapshot(case)
    assert after.key == key and after.epoch == before.epoch + 1 and after.human_rows == 1
    replay = await case.client.post(
        _human_url(case, "staff", ticket_id), headers=case.headers[3], json=body
    )
    assert replay.status_code == 200 and replay.json()["reused"] is True
    assert replay.json()["session_key"] == key
    assert replay.json()["message"]["id"] == first.json()["message"]["id"]
    assert await _snapshot(case) == after
