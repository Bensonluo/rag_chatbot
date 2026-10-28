"""Quality writes must refer to the displayed conversation and transcript.

Real JWT/HTTP, SQLite-FK, repositories, compiled graph and counters are used.
Only local Redis/model boundaries and explicit timing barriers are controlled.
No negative command discovers or substitutes a fresh server identity.
"""

import asyncio
import socket
from collections.abc import Generator
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from app.config.settings import settings
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.models.enums.message import MessageRole
from app.repositories.feedback_repository import FeedbackRepository
from app.repositories.session_repository import SessionRepository
from app.services.chat.metrics import FEEDBACK_RATINGS
from app.tests.evaluation.test_human_conversation import _bot
from app.tests.evaluation.test_human_conversation import _Case as Case
from app.tests.evaluation.test_human_conversation import case as case

_MISSING = object()


@pytest.fixture(autouse=True)
def no_external_dns(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    attempts: list[str] = []

    def blocked(*args: Any, **kwargs: Any) -> None:
        attempts.append("DNS")
        raise AssertionError("Quality mutation evaluation forbids external DNS")

    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        monkeypatch.setattr(socket, name, blocked)
    yield
    assert attempts == []


async def _key(case: Case, sid: int | None = None, *, actor: int = 1) -> str:
    response = await case.client.get(
        f"/api/v1/sessions/{sid or case.private_id}", headers=case.headers[actor]
    )
    assert response.status_code == 200
    return str(response.json()["session_key"])


async def _last(case: Case, sid: int | None = None) -> int:
    async with case.maker() as db:
        value = await db.scalar(
            select(func.max(Message.id)).where(Message.session_id == (sid or case.private_id))
        )
    assert type(value) is int
    return value


async def _assistant(case: Case, sid: int | None = None) -> dict[str, Any]:
    async with case.maker() as db:
        row = await db.scalar(
            select(Message)
            .where(
                Message.session_id == (sid or case.private_id),
                Message.role == MessageRole.ASSISTANT,
            )
            .order_by(Message.id.desc())
        )
        assert row is not None
        return {
            "id": row.id,
            "content": row.content,
            "rating": row.user_rating,
            "feedback_text": row.feedback_text,
        }


def _rating_count() -> float:
    return float(
        sum(FEEDBACK_RATINGS.labels(rating=str(rating))._value.get() for rating in (-1, 0, 1))
    )


async def _snapshot(case: Case, sid: int | None = None) -> dict[str, Any]:
    sid = sid or case.private_id
    async with case.maker() as db:
        row = await db.get(ChatSession, sid)
        assert row is not None
        messages = (
            await db.execute(
                select(
                    Message.id,
                    Message.role,
                    Message.status,
                    Message.content,
                    Message.user_rating,
                    Message.feedback_text,
                )
                .where(Message.session_id == sid)
                .order_by(Message.id)
            )
        ).all()
        return {
            "owner": row.user_id,
            "key": row.session_key,
            "epoch": row.automation_epoch,
            "resolved": row.resolved_at,
            "reopened": row.reopened_count,
            "metadata": row.session_metadata,
            "messages": tuple(tuple(message) for message in messages),
            "ratings": _rating_count(),
            "evictions": tuple(case.calls.evictions),
        }


async def _feedback(
    case: Case, message_id: int, key: object = _MISSING, *, actor: int | None = 1
) -> Any:
    body: dict[str, Any] = {"message_id": message_id, "rating": -1, "text": "Original assessment"}
    if key is not _MISSING:
        body["expected_session_key"] = key
    return await case.client.post(
        "/api/v1/feedback", headers=case.headers[actor] if actor is not None else {}, json=body
    )


async def _review(
    case: Case,
    key: object = _MISSING,
    version: object = _MISSING,
    *,
    actor: int = 3,
    sid: int | None = None,
    resolved: bool = True,
) -> Any:
    body: dict[str, Any] = {"resolved": resolved, "note": "Review of the displayed transcript"}
    if key is not _MISSING:
        body["expected_session_key"] = key
    if version is not _MISSING:
        body["expected_review_version"] = version
    return await case.client.post(
        f"/api/v1/sessions/{sid or case.private_id}/review", headers=case.headers[actor], json=body
    )


async def _proposal(case: Case, sid: int | None = None) -> dict[str, Any]:
    response = await case.client.get(
        f"/api/v1/sessions/{sid or case.private_id}/review", headers=case.headers[3]
    )
    assert response.status_code == 200
    value = response.json()
    assert isinstance(value, dict)
    assert value["session_id"] == (sid or case.private_id)
    version = value["review_version"]
    assert isinstance(version, str) and len(version) == 64
    assert all(character in "0123456789abcdef" for character in version)
    return value


async def _replace(case: Case, key: str, *, new_owner: int) -> tuple[str, dict[str, Any]]:
    sid = case.private_id
    deleted = await case.client.delete(
        f"/api/v1/sessions/{sid}",
        params={"expected_session_key": key},
        headers=case.headers[1],
    )
    assert deleted.status_code == 204
    created = await case.client.post(
        "/api/v1/sessions",
        headers=case.headers[new_owner],
        json={"title": "Replacement consultation"},
    )
    assert created.status_code == 200
    assert created.json()["id"] == sid
    new_key = created.json()["session_key"]
    assert new_key != key
    assert (await _bot(case, "我要投诉", actor=new_owner)).status_code == 200
    return new_key, await _assistant(case)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("missing", 409),
        ("null", 409),
        ("mismatch", 409),
        ("malformed", 422),
        ("boolean", 422),
        ("array", 422),
    ],
)
async def test_feedback_key_rejections_have_no_quality_side_effects(
    case: Case, kind: str, expected: int
) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    row = await _assistant(case)
    value = {
        "missing": _MISSING,
        "null": None,
        "mismatch": str(uuid4()),
        "malformed": "bad-key",
        "boolean": True,
        "array": [],
    }[kind]
    before = await _snapshot(case)
    response = await _feedback(case, row["id"], value)
    assert response.status_code == expected
    assert await _snapshot(case) == before


async def test_private_feedback_requires_identity_in_demo_mode(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    assert (await _bot(case, "Hello")).status_code == 200
    before = await _snapshot(case)
    response = await _feedback(case, (await _assistant(case))["id"])
    assert response.status_code == 409 and await _snapshot(case) == before


@pytest.mark.parametrize(("actor", "expected"), [(2, 404), (None, 401)])
async def test_feedback_actor_is_checked_before_missing_instance(
    case: Case, actor: int | None, expected: int
) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    before = await _snapshot(case)
    response = await _feedback(case, (await _assistant(case))["id"], actor=actor)
    assert response.status_code == expected and await _snapshot(case) == before


@pytest.mark.parametrize("demo", [False, True])
async def test_feedback_valid_receipt_identifies_the_rated_instance(
    case: Case, monkeypatch: pytest.MonkeyPatch, demo: bool
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", demo)
    assert (await _bot(case, "Hello")).status_code == 200
    key = await _key(case)
    before = _rating_count()
    row = await _assistant(case)
    response = await _feedback(case, row["id"], key)
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "message_id": row["id"],
        "rating": -1,
        "session_id": case.private_id,
        "session_key": key,
    }
    assert (await _assistant(case))["rating"] == -1
    assert _rating_count() == before + 1 and case.calls.evictions == [row["content"]]


async def test_old_feedback_cannot_rate_naturally_reused_id_after_clear(case: Case) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    old_key, old = await _key(case), await _assistant(case)
    response = await case.client.delete(
        "/api/v1/chat/history",
        params={"session_id": case.private_id, "expected_session_key": old_key},
        headers=case.headers[1],
    )
    assert response.status_code == 204 and await _key(case) != old_key
    assert (await _bot(case, "我要投诉")).status_code == 200
    assert (await _assistant(case))["id"] == old["id"]
    before = await _snapshot(case)
    response = await _feedback(case, old["id"], old_key)
    assert response.status_code == 409 and await _snapshot(case) == before


@pytest.mark.parametrize(("new_owner", "expected"), [(1, 409), (2, 404)])
async def test_old_feedback_cannot_cross_deleted_session_incarnation(
    case: Case, new_owner: int, expected: int
) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    old_key, old = await _key(case), await _assistant(case)
    _, fresh = await _replace(case, old_key, new_owner=new_owner)
    assert fresh["id"] == old["id"]
    before = await _snapshot(case)
    response = await _feedback(case, old["id"], old_key)
    assert response.status_code == expected and await _snapshot(case) == before


async def test_inflight_feedback_never_rates_another_owner_replacement(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    key, row = await _key(case), await _assistant(case)
    entered, release = asyncio.Event(), asyncio.Event()
    lookup = (
        "get_feedback_target"
        if hasattr(FeedbackRepository, "get_feedback_target")
        else "get_owned_message"
    )
    original = getattr(FeedbackRepository, lookup)
    pause = True

    async def observed(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal pause
        owned = await original(self, *args, **kwargs)
        if owned is not None and pause:
            pause = False
            entered.set()
            await asyncio.wait_for(release.wait(), 5)
        return owned

    monkeypatch.setattr(FeedbackRepository, lookup, observed)
    pending = asyncio.create_task(_feedback(case, row["id"], key))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        deleted = await case.client.delete(
            f"/api/v1/sessions/{case.private_id}",
            params={"expected_session_key": key},
            headers=case.headers[1],
        )
        if deleted.status_code == 204:
            new = await case.client.post(
                "/api/v1/sessions",
                headers=case.headers[2],
                json={"title": "Other customer replacement"},
            )
            assert new.status_code == 200 and new.json()["id"] == case.private_id
            assert (await _bot(case, "我要投诉", actor=2)).status_code == 200
            assert (await _assistant(case))["id"] == row["id"]
            before = await _snapshot(case)
        else:
            assert deleted.status_code == 409
            before = None
    finally:
        release.set()
    response = await asyncio.wait_for(pending, 5)
    if before is not None:
        assert response.status_code in {404, 409}
        assert await _snapshot(case) == before
    else:
        assert response.status_code == 200
        assert (await _snapshot(case))["owner"] == 1


async def test_user_message_remains_ineligible_for_feedback_with_valid_key(case: Case) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    async with case.maker() as db:
        identifier = await db.scalar(
            select(Message.id).where(
                Message.session_id == case.private_id, Message.role == MessageRole.USER
            )
        )
    assert isinstance(identifier, int)
    before = await _snapshot(case)
    response = await _feedback(case, identifier, await _key(case))
    assert response.status_code == 404 and await _snapshot(case) == before


@pytest.mark.parametrize("missing", ["both", "key", "version"])
async def test_private_review_requires_instance_and_transcript_snapshot(
    case: Case, missing: str
) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    proposal = await _proposal(case)
    before = await _snapshot(case)
    response = await _review(
        case,
        _MISSING if missing in {"both", "key"} else proposal["session_key"],
        _MISSING if missing in {"both", "version"} else proposal["review_version"],
    )
    assert response.status_code == 409 and await _snapshot(case) == before


async def test_review_queue_and_receipt_bind_the_exact_reviewed_transcript(case: Case) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    proposal = await _proposal(case)
    key, last, version = (
        proposal["session_key"],
        proposal["last_message_id"],
        proposal["review_version"],
    )
    queue = await case.client.get("/api/v1/sessions/review-queue", headers=case.headers[3])
    assert queue.status_code == 200
    item = next(row for row in queue.json()["items"] if row["session_id"] == case.private_id)
    assert item["session_key"] == key and item["last_message_id"] == last
    assert item["review_version"] == version and item["last_human_message_id"] == 0
    response = await _review(case, key, version)
    assert response.status_code == 200
    assert response.json()["session_key"] == key and response.json()["last_message_id"] == last
    assert response.json()["review_version"] != version
    after = await _snapshot(case)
    assert after["resolved"] is not None
    assert after["metadata"]["qa_verdict"]["resolved"] is True
    queue = await case.client.get("/api/v1/sessions/review-queue", headers=case.headers[3])
    assert case.private_id not in {row["session_id"] for row in queue.json()["items"]}


async def test_old_review_cannot_resolve_a_new_question_with_the_same_session_key(
    case: Case,
) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    proposal = await _proposal(case)
    key, last, version = (
        proposal["session_key"],
        proposal["last_message_id"],
        proposal["review_version"],
    )
    assert (await _bot(case, "我还有新的问题，我要投诉")).status_code == 200
    assert await _key(case) == key and await _last(case) > last
    before = await _snapshot(case)
    response = await _review(case, key, version)
    assert response.status_code == 409 and await _snapshot(case) == before


async def test_old_review_cannot_use_reused_message_cursor_after_clear(case: Case) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    proposal = await _proposal(case)
    key, last, version = (
        proposal["session_key"],
        proposal["last_message_id"],
        proposal["review_version"],
    )
    clear = await case.client.delete(
        "/api/v1/chat/history",
        params={"session_id": case.private_id, "expected_session_key": key},
        headers=case.headers[1],
    )
    assert clear.status_code == 204
    assert (await _bot(case, "我要投诉")).status_code == 200
    assert await _key(case) != key and await _last(case) == last
    before = await _snapshot(case)
    response = await _review(case, key, version)
    assert response.status_code == 409 and await _snapshot(case) == before


async def test_old_review_cannot_resolve_another_owner_replacement(case: Case) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    proposal = await _proposal(case)
    key, last, version = (
        proposal["session_key"],
        proposal["last_message_id"],
        proposal["review_version"],
    )
    await _replace(case, key, new_owner=2)
    assert await _last(case) == last
    before = await _snapshot(case)
    response = await _review(case, key, version)
    assert response.status_code == 409 and await _snapshot(case) == before


async def test_customer_review_role_rejection_precedes_missing_identity(case: Case) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    before = await _snapshot(case)
    response = await _review(case, actor=1)
    assert response.status_code == 403 and await _snapshot(case) == before


async def test_inactive_reviewer_cannot_change_quality(case: Case) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    async with case.maker() as db:
        user = await db.get(User, 3)
        assert user is not None
        user.is_active = False
        await db.commit()
    before = await _snapshot(case)
    response = await _review(case, await _key(case), "0" * 64)
    assert response.status_code == 403 and await _snapshot(case) == before


@pytest.mark.parametrize("version", [True, "bad-version", [], "A" * 64])
async def test_review_malformed_snapshot_is_rejected_without_side_effects(
    case: Case, version: Any
) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    before = await _snapshot(case)
    response = await _review(case, await _key(case), version)
    assert response.status_code == 422 and await _snapshot(case) == before


async def test_inflight_review_cannot_overwrite_a_newly_accepted_question(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    proposal = await _proposal(case)
    key, last, version = (
        proposal["session_key"],
        proposal["last_message_id"],
        proposal["review_version"],
    )
    entered, release = asyncio.Event(), asyncio.Event()
    original = SessionRepository.record_review_verdict

    async def observed(self: Any, *args: Any, **kwargs: Any) -> Any:
        entered.set()
        await asyncio.wait_for(release.wait(), 5)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(SessionRepository, "record_review_verdict", observed)
    pending = asyncio.create_task(_review(case, key, version))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        newer = await _bot(case, "我还有新的问题，我要投诉")
        assert newer.status_code in {200, 409}
        before = await _snapshot(case)
    finally:
        release.set()
    response = await asyncio.wait_for(pending, 5)
    if newer.status_code == 200:
        assert await _last(case) > last
        assert response.status_code == 409 and await _snapshot(case) == before
    else:
        assert response.status_code == 200 and await _last(case) == last


async def test_feedback_change_invalidates_old_review_without_advancing_message_id(
    case: Case,
) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    proposal = await _proposal(case)
    message = await _assistant(case)
    assert (await _feedback(case, message["id"], proposal["session_key"])).status_code == 200
    current = await _proposal(case)
    assert current["session_key"] == proposal["session_key"]
    assert current["last_message_id"] == proposal["last_message_id"]
    assert current["review_version"] != proposal["review_version"]
    before = await _snapshot(case)
    response = await _review(case, proposal["session_key"], proposal["review_version"])
    assert response.status_code == 409 and await _snapshot(case) == before


async def test_new_human_question_invalidates_old_review_without_advancing_bot_cursor(
    case: Case,
) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    key = await _key(case)
    created = await case.handoff.create_ticket_for_session(
        case.private_id, 1, "explicit", {"user_message": "A local support question"}
    )
    ticket_id = created["ticket_id"]
    claimed = await case.client.post(
        f"/api/v1/handoff/tickets/{ticket_id}/claim",
        params={"expected_session_key": key},
        headers=case.headers[3],
    )
    assert claimed.status_code == 200
    proposal = await _proposal(case)
    sent = await case.client.post(
        f"/api/v1/handoff/sessions/{case.private_id}/messages",
        headers=case.headers[1],
        json={
            "client_message_id": str(uuid4()),
            "content": "I have another unresolved question",
            "expected_session_key": key,
            "expected_ticket_id": ticket_id,
        },
    )
    assert sent.status_code == 200
    current = await _proposal(case)
    assert current["session_key"] == key
    assert current["last_message_id"] == proposal["last_message_id"]
    assert current["last_human_message_id"] != proposal["last_human_message_id"]
    assert current["review_version"] != proposal["review_version"]
    before = await _snapshot(case)
    response = await _review(case, key, proposal["review_version"])
    assert response.status_code == 409 and await _snapshot(case) == before


async def test_review_correction_requires_the_new_authoritative_version(case: Case) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    proposal = await _proposal(case)
    first = await _review(case, proposal["session_key"], proposal["review_version"])
    assert first.status_code == 200
    before = await _snapshot(case)
    stale = await _review(case, proposal["session_key"], proposal["review_version"], resolved=False)
    assert stale.status_code == 409 and await _snapshot(case) == before
    current = await _proposal(case)
    assert current["review_version"] == first.json()["review_version"]
    correction = await _review(
        case, current["session_key"], current["review_version"], resolved=False
    )
    assert correction.status_code == 200
    after = await _snapshot(case)
    assert after["resolved"] is None and after["metadata"]["qa_verdict"]["resolved"] is False
    assert after["reopened"] == before["reopened"]


async def test_public_demo_quality_review_also_requires_explicit_proposal(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    assert (await _bot(case, "Hello", session_id=10)).status_code == 200
    before = await _snapshot(case, 10)
    response = await _review(case, sid=10)
    assert response.status_code == 409 and await _snapshot(case, 10) == before


async def test_well_formed_but_wrong_review_version_is_refused(case: Case) -> None:
    assert (await _bot(case, "Hello")).status_code == 200
    key = await _key(case)
    before = await _snapshot(case)
    response = await _review(case, key, "0" * 64)
    assert response.status_code == 409 and await _snapshot(case) == before


async def test_committed_feedback_releases_gate_before_slow_eviction_and_keeps_old_receipt(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services.chat import cache_eviction

    assert (await _bot(case, "Hello")).status_code == 200
    old_key, old_message = await _key(case), await _assistant(case)
    before_ratings = _rating_count()
    entered, release = asyncio.Event(), asyncio.Event()
    original = cache_eviction.evict_downvoted_response

    async def delayed_eviction(content: str) -> int:
        assert content == old_message["content"]
        entered.set()
        await asyncio.wait_for(release.wait(), 5)
        return await original(content)

    monkeypatch.setattr(cache_eviction, "evict_downvoted_response", delayed_eviction)
    pending = asyncio.create_task(_feedback(case, old_message["id"], old_key))
    new_key: str | None = None
    new_message: dict[str, Any] | None = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        # An independent SQL read proves that remote cleanup is after commit.
        assert (await _assistant(case))["rating"] == -1
        cleared = await case.client.delete(
            "/api/v1/chat/history",
            params={"session_id": case.private_id, "expected_session_key": old_key},
            headers=case.headers[1],
        )
        if cleared.status_code == 204:
            new_key = await _key(case)
            assert new_key != old_key
            assert (await _bot(case, "我要投诉")).status_code == 200
            new_message = await _assistant(case)
            assert new_message["id"] == old_message["id"]
    finally:
        release.set()
        response = await asyncio.wait_for(pending, 5)
    assert cleared.status_code == 204
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "message_id": old_message["id"],
        "rating": -1,
        "session_id": case.private_id,
        "session_key": old_key,
    }
    assert new_key is not None and new_message is not None
    assert new_message["rating"] is None
    assert new_message["content"] != old_message["content"]
    assert case.calls.evictions == [old_message["content"]]
    assert _rating_count() == before_ratings + 1
