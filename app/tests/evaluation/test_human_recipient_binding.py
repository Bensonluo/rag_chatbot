"""Actual JWT/SQL customer sends stay bound to the ticket they were composed for."""

from uuid import uuid4

import pytest

from app.models.database.session import ChatSession
from app.tests.evaluation.test_human_conversation import (
    _Case,
    _customer_url,
    _human_rows,
    _send,
    _session_key,
    _staff_url,
    _ticket,
)
from app.tests.evaluation.test_human_conversation import case as case


async def _bound_send(case: _Case, ticket_id: object, *, key: str | None = None, actor: int = 1):
    return await case.client.post(
        _customer_url(case),
        headers=case.headers.get(actor, {}),
        json={
            "client_message_id": key or str(uuid4()),
            "content": "Original ticket draft",
            "expected_ticket_id": ticket_id,
            "expected_session_key": await _session_key(case),
        },
    )


async def test_same_ticket_bound_send_and_exact_replay_use_one_row(case: _Case) -> None:
    ticket_id = await _ticket(case)
    key = str(uuid4())
    first = await _bound_send(case, ticket_id, key=key)
    second = await _bound_send(case, ticket_id, key=key)
    assert first.status_code == second.status_code == 200
    assert first.json()["message"] == second.json()["message"]
    assert second.json()["reused"] is True
    assert len(await _human_rows(case)) == 1


@pytest.mark.parametrize("committed", [False, True])
async def test_new_ticket_cannot_receive_an_old_immutable_attempt(
    case: _Case, committed: bool
) -> None:
    original = await _ticket(case)
    key = str(uuid4())
    if committed:
        assert (await _bound_send(case, original, key=key)).status_code == 200
    assert (
        await _send(case, _staff_url(original), 3, "Handling ended", resolve=True)
    ).status_code == 200
    newer = await _ticket(case)
    assert newer != original
    before = [(row.id, row.ticket_id, row.content) for row in await _human_rows(case)]
    response = await _bound_send(case, original, key=key)
    assert response.status_code == 409
    assert "no-store" in response.headers["cache-control"]
    assert [(row.id, row.ticket_id, row.content) for row in await _human_rows(case)] == before
    assert (await _bound_send(case, newer)).status_code == 200
    assert (await _human_rows(case))[-1].ticket_id == newer


async def test_binding_mismatch_does_not_change_resolution(case: _Case) -> None:
    ticket_id = await _ticket(case)
    async with case.maker() as db:
        before = await db.get(ChatSession, case.private_id)
        assert before is not None
        snapshot = (before.resolved_at, before.reopened_count)
    response = await case.client.post(
        _customer_url(case),
        headers=case.headers[1],
        json={
            "client_message_id": str(uuid4()),
            "content": "Problem solved.",
            "expected_ticket_id": ticket_id + 1,
            "expected_session_key": await _session_key(case),
        },
    )
    assert response.status_code == 409
    assert await _human_rows(case) == []
    async with case.maker() as db:
        after = await db.get(ChatSession, case.private_id)
        assert after is not None
        assert (after.resolved_at, after.reopened_count) == snapshot


@pytest.mark.parametrize("ticket_id", [0, -1, True, "8", 2_147_483_648])
async def test_expected_ticket_validation_is_strict(case: _Case, ticket_id: object) -> None:
    await _ticket(case)
    assert (await _bound_send(case, ticket_id)).status_code == 422
    assert await _human_rows(case) == []


@pytest.mark.parametrize(("actor", "status"), [(0, 401), (2, 404), (3, 404)])
async def test_ticket_hint_does_not_grant_session_ownership(
    case: _Case, actor: int, status: int
) -> None:
    ticket_id = await _ticket(case)
    assert (await _bound_send(case, ticket_id, actor=actor)).status_code == status
    assert await _human_rows(case) == []


async def test_legacy_omitted_hint_remains_compatible(case: _Case) -> None:
    ticket_id = await _ticket(case)
    response = await _send(case, _customer_url(case), 1, "Legacy client message")
    assert response.status_code == 200
    assert response.json()["message"]["ticket_id"] == ticket_id
