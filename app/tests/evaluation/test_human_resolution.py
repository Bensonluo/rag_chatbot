"""Customer resolution evidence follows the real human-message transaction.

The reused fixture has real JWT/HTTP, SQLite foreign keys, a compiled graph,
coordination and persistence, with a socket guard and local Redis boundary.
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.tests.evaluation.test_human_conversation import (
    _ack,
    _bot,
    _bot_rows,
    _Case,
    _customer_url,
    _human_rows,
    _send,
    _session_key,
    _staff_url,
    _stats,
    _ticket,
)
from app.tests.evaluation.test_human_conversation import case as case


@dataclass(frozen=True)
class _State:
    resolved_at: datetime | None
    reopened_count: int
    ticket_status: str
    automation_epoch: int


async def _state(case: _Case, ticket_id: int) -> _State:
    async with case.maker() as db:
        session = await db.get(ChatSession, case.private_id)
        ticket = await db.get(HandoffTicket, ticket_id)
        assert session is not None and ticket is not None
        return _State(
            session.resolved_at, session.reopened_count, ticket.status, session.automation_epoch
        )


async def _resolved_by_bot_then_claimed(case: _Case) -> int:
    ticket_id = await _ticket(case, claim=False)
    assert (await _bot(case, "Problem solved.")).status_code == 200
    assert (await _state(case, ticket_id)).resolved_at is not None
    assert (
        await case.client.post(
            f"/api/v1/handoff/tickets/{ticket_id}/claim?expected_session_key={await _session_key(case)}",
            headers=case.headers[3],
        )
    ).status_code == 200
    return ticket_id


@pytest.mark.parametrize(
    ("content", "confirms"),
    [
        ("问题解决了，谢谢！", True),
        ("Problem solved.", True),
        ("Thank you, the issue is resolved.", True),
        ("谢谢", False),
        ("Thanks", False),
        ("问题还没解决", False),
        ("It is not solved.", False),
        ("Problem solved?", False),
        ("问题解决了吗？", False),
        ("He said 'Problem solved'.", False),
        ("如果问题解决了我会告诉你", False),
        ("Problem solved, but I need more help.", False),
    ],
)
async def test_customer_human_confirmation_is_durable_resolution_evidence(
    case: _Case, content: str, confirms: bool
) -> None:
    ticket_id = await _ticket(case)
    ack = _ack(await _send(case, _customer_url(case), 1, content))
    state = await _state(case, ticket_id)
    assert (state.resolved_at is not None) is confirms
    assert state.reopened_count == 0
    assert state.ticket_status == "claimed"
    assert state.automation_epoch == 0
    assert ack["ticket"]["status"] == "claimed"
    assert (await _human_rows(case))[0].content == content
    assert await _bot_rows(case) == []
    assert case.calls.graph == case.calls.cache_epoch == 0


async def test_old_positive_replay_cannot_resolve_after_a_new_negative_message(case: _Case) -> None:
    ticket_id = await _ticket(case)
    positive, negative = str(uuid4()), str(uuid4())
    first = _ack(await _send(case, _customer_url(case), 1, "Problem solved.", key=positive))
    assert (await _state(case, ticket_id)).resolved_at is not None
    _ack(await _send(case, _customer_url(case), 1, "I still need help.", key=negative))
    reopened = await _state(case, ticket_id)
    assert reopened.resolved_at is None
    assert reopened.reopened_count == 1
    replay = _ack(await _send(case, _customer_url(case), 1, "Problem solved.", key=positive))
    assert replay["reused"] is True
    assert replay["message"] == first["message"]
    assert await _state(case, ticket_id) == reopened
    assert len(await _human_rows(case)) == 2
    stats = await _stats(case)
    # Positive/negative customer evidence exists, but no bot served this
    # session: retain the documented bot-served cross-tab denominator.
    for key in [
        "sessions_served",
        "sessions_resolved",
        "sessions_reopened",
        "sessions_with_completed_replies",
        "sessions_with_failed_replies",
    ]:
        assert stats[key] == 0


async def test_old_negative_replay_cannot_reopen_after_a_new_positive_message(case: _Case) -> None:
    ticket_id = await _resolved_by_bot_then_claimed(case)
    negative, positive = str(uuid4()), str(uuid4())
    first = _ack(await _send(case, _customer_url(case), 1, "I still need help.", key=negative))
    assert (await _state(case, ticket_id)).reopened_count == 1
    _ack(await _send(case, _customer_url(case), 1, "问题解决了", key=positive))
    confirmed = await _state(case, ticket_id)
    assert confirmed.resolved_at is not None
    assert confirmed.reopened_count == 1
    replay = _ack(await _send(case, _customer_url(case), 1, "I still need help.", key=negative))
    assert replay["reused"] is True
    assert replay["message"] == first["message"]
    assert await _state(case, ticket_id) == confirmed
    assert len(await _human_rows(case)) == 2


@pytest.mark.parametrize("initially_resolved", [False, True])
@pytest.mark.parametrize("close", [False, True])
async def test_staff_content_and_staff_close_are_not_customer_resolution_evidence(
    case: _Case, initially_resolved: bool, close: bool
) -> None:
    ticket_id = await _ticket(case)
    if initially_resolved:
        _ack(await _send(case, _customer_url(case), 1, "Problem solved."))
    before = await _state(case, ticket_id)
    assert (before.resolved_at is not None) is initially_resolved
    content = "I still need help." if initially_resolved else "Problem solved."
    _ack(await _send(case, _staff_url(ticket_id), 3, content, resolve=close))
    after = await _state(case, ticket_id)
    assert after.resolved_at == before.resolved_at
    assert after.reopened_count == before.reopened_count
    assert after.ticket_status == ("resolved" if close else "claimed")
    assert after.automation_epoch == (1 if close else 0)


async def test_administrative_close_preserves_the_customers_resolution_state(case: _Case) -> None:
    ticket_id = await _ticket(case)
    _ack(await _send(case, _customer_url(case), 1, "Problem solved."))
    before = await _state(case, ticket_id)
    assert before.resolved_at is not None
    response = await case.client.post(
        f"/api/v1/handoff/tickets/{ticket_id}/resolve?expected_session_key={await _session_key(case)}",
        headers=case.headers[3],
    )
    assert response.status_code == 200
    after = await _state(case, ticket_id)
    assert after.resolved_at == before.resolved_at
    assert after.reopened_count == before.reopened_count
    assert after.ticket_status == "resolved"
    assert after.automation_epoch == 1


@pytest.mark.parametrize("initially_resolved", [False, True])
async def test_real_customer_insert_failure_rolls_back_resolution_and_reopen(
    case: _Case, initially_resolved: bool
) -> None:
    ticket_id = (
        await _resolved_by_bot_then_claimed(case) if initially_resolved else await _ticket(case)
    )
    before = await _state(case, ticket_id)
    key = str(uuid4())
    content = "I still need help." if initially_resolved else "Problem solved."
    async with case.maker() as db:
        await db.execute(
            text(
                "CREATE TRIGGER reject_customer BEFORE INSERT ON handoff_messages "
                "WHEN NEW.author_kind = 'customer' BEGIN "
                "SELECT RAISE(ABORT, 'isolated customer write rejected'); END"
            )
        )
        await db.commit()
    response = await _send(case, _customer_url(case), 1, content, key=key)
    assert response.status_code == 503
    assert "isolated customer write rejected" not in response.text
    assert await _state(case, ticket_id) == before
    assert await _human_rows(case) == []
    async with case.maker() as db:
        await db.execute(text("DROP TRIGGER reject_customer"))
        await db.commit()
    ack = _ack(await _send(case, _customer_url(case), 1, content, key=key))
    assert ack["reused"] is False
    after = await _state(case, ticket_id)
    assert (after.resolved_at is None) is initially_resolved
    assert after.reopened_count == (1 if initially_resolved else 0)
    assert len(await _human_rows(case)) == 1


async def test_real_resolution_update_failure_cannot_leave_a_delivered_customer_message(
    case: _Case,
) -> None:
    ticket_id = await _ticket(case)
    key = str(uuid4())
    async with case.maker() as db:
        await db.execute(
            text(
                "CREATE TRIGGER reject_resolution BEFORE UPDATE OF resolved_at ON chat_sessions "
                "WHEN NEW.resolved_at IS NOT NULL BEGIN "
                "SELECT RAISE(ABORT, 'isolated resolution update rejected'); END"
            )
        )
        await db.commit()
    response = await _send(case, _customer_url(case), 1, "Problem solved.", key=key)
    assert response.status_code == 503
    assert await _human_rows(case) == []
    assert (await _state(case, ticket_id)).resolved_at is None
    async with case.maker() as db:
        await db.execute(text("DROP TRIGGER reject_resolution"))
        await db.commit()
    assert (
        _ack(await _send(case, _customer_url(case), 1, "Problem solved.", key=key))["reused"]
        is False
    )
    assert (await _state(case, ticket_id)).resolved_at is not None


async def test_multiple_new_followups_reopen_one_confirmed_episode_once(case: _Case) -> None:
    ticket_id = await _ticket(case)
    _ack(await _send(case, _customer_url(case), 1, "问题解决了"))
    key = str(uuid4())
    _ack(await _send(case, _customer_url(case), 1, "问题还没解决", key=key))
    reopened = await _state(case, ticket_id)
    assert reopened.resolved_at is None and reopened.reopened_count == 1
    assert (
        _ack(await _send(case, _customer_url(case), 1, "问题还没解决", key=key))["reused"] is True
    )
    _ack(await _send(case, _customer_url(case), 1, "Please help with the remaining issue."))
    assert await _state(case, ticket_id) == reopened
    assert len(await _human_rows(case)) == 3


async def test_human_resolution_changes_quality_cross_tabs_without_changing_bot_populations(
    case: _Case,
) -> None:
    await _resolved_by_bot_then_claimed(case)
    original_bot = [(row.id, row.content, row.status) for row in await _bot_rows(case)]
    before = await _stats(case)
    assert before["sessions_served"] == before["sessions_resolved"] == 1
    assert before["sessions_reopened"] == 0
    _ack(await _send(case, _customer_url(case), 1, "The issue is not solved; I still need help."))
    reopened = await _stats(case)
    assert reopened["sessions_resolved"] == 0
    assert reopened["sessions_reopened"] == 1
    _ack(await _send(case, _customer_url(case), 1, "Problem solved."))
    confirmed = await _stats(case)
    assert confirmed["sessions_resolved"] == 1
    assert confirmed["sessions_reopened"] == 1
    for key in [
        "sessions_served",
        "sessions_with_completed_replies",
        "sessions_with_failed_replies",
        "sessions_latest_reply_failed",
        "sessions_escalated",
        "containment_rate",
    ]:
        assert before[key] == reopened[key] == confirmed[key]
    assert [(row.id, row.content, row.status) for row in await _bot_rows(case)] == original_bot
    assert case.calls.graph == 1
