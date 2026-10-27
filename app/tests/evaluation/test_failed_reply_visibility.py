"""Persisted reply failures must remain visible beside the containment proxy."""

import socket
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY, generate_latest
from prometheus_client.parser import text_string_to_metric_families
from sqlalchemy import delete, event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_db
from app.api.v1 import sessions as sessions_api
from app.config.settings import settings
from app.core.security import create_access_token
from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.models.database.user import User
from app.models.enums.message import MessageRole, MessageStatus
from app.services.handoff import containment_metrics

_NOW = datetime(2026, 10, 28, 12, tzinfo=UTC)
_SINCE = _NOW - timedelta(days=7)
_RECENT = _NOW.replace(tzinfo=None) - timedelta(hours=1)
_FIELDS = (
    "sessions_with_failed_replies",
    "sessions_with_completed_replies",
    "sessions_latest_reply_failed",
)


class _Clock:
    @staticmethod
    def now(tz: Any = None) -> datetime:
        return _NOW.astimezone(tz) if tz is not None else _NOW.replace(tzinfo=None)


@dataclass
class _Case:
    maker: async_sessionmaker[AsyncSession]
    client: AsyncClient
    headers: dict[str, str]


@pytest.fixture
async def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[_Case, None]:
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "SECRET_KEY", "isolated-reply-visibility-signing-key-32-bytes")
    monkeypatch.setattr(sessions_api, "datetime", _Clock)
    monkeypatch.setattr(containment_metrics, "datetime", _Clock)
    attempts: list[str] = []

    def forbid_network(_socket: Any, address: Any, *args: Any, **kwargs: Any) -> None:
        attempts.append(str(address))
        raise AssertionError("reply-visibility evaluation must not connect externally")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid_network)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/reply-visibility.sqlite3")

    @event.listens_for(engine.sync_engine, "connect")
    def foreign_keys(connection: Any, _record: Any) -> None:
        connection.execute("PRAGMA foreign_keys=ON")

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(
            lambda conn: Base.metadata.create_all(
                conn,
                tables=[
                    Base.metadata.tables[name]
                    for name in ("users", "chat_sessions", "messages", "handoff_tickets")
                ],
            )
        )
    async with maker() as db:
        db.add(User(id=1, email="reply-ops@evaluation.invalid", hashed_password="!"))
        await db.flush()
        db.add_all([ChatSession(id=sid, user_id=1) for sid in (1, 2, 3)])
        await db.commit()
    app = FastAPI()
    app.include_router(sessions_api.router, prefix="/api/v1")

    async def isolated_db() -> AsyncGenerator[AsyncSession, None]:
        async with maker() as db:
            yield db

    app.dependency_overrides[get_db] = isolated_db
    token = create_access_token({"sub": "reply-ops@evaluation.invalid", "user_id": 1})
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://isolated-reply-ops.invalid"
        ) as client:
            yield _Case(maker, client, {"Authorization": f"Bearer {token}"})
    finally:
        await engine.dispose()
        assert attempts == []


async def _snapshot(
    case: _Case, expected: tuple[int, int, int], *, days: int = 7
) -> dict[str, Any]:
    response = await case.client.get(
        "/api/v1/sessions/containment-stats",
        params={"window_days": days},
        headers=case.headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, dict)
    assert tuple(body[field] for field in _FIELDS) == expected
    refreshed = await containment_metrics.refresh_containment_stats(case.maker, window_days=days)
    assert refreshed == body
    exported = {
        sample.name: sample
        for family in text_string_to_metric_families(generate_latest(REGISTRY).decode())
        for sample in family.samples
    }
    for field, value in zip(_FIELDS, expected, strict=True):
        assert type(body[field]) is int
        assert REGISTRY.get_sample_value(f"chat_{field}") == float(value)
        assert exported[f"chat_{field}"].value == float(value)
        assert exported[f"chat_{field}"].labels == {}
    return body


async def _reply(
    case: _Case,
    status: MessageStatus | str,
    *,
    session_id: int = 1,
    at: datetime = _RECENT,
    role: MessageRole = MessageRole.ASSISTANT,
    message_id: int | None = None,
    rating: int | None = None,
) -> None:
    async with case.maker() as db:
        message = Message(
            session_id=session_id,
            role=role,
            content="Synthetic reply-state evidence.",
            status=status if isinstance(status, MessageStatus) else MessageStatus.PENDING,
            created_at=at,
            user_rating=rating,
        )
        if message_id is not None:
            message.id = message_id
        db.add(message)
        await db.flush()
        if not isinstance(status, MessageStatus):
            # A future/unrecognized stored status must not become success.
            # Aggregate queries can inspect it without ORM enum decoding.
            await db.execute(
                text("UPDATE messages SET status = :status WHERE id = :id"),
                {"status": status, "id": message.id},
            )
        await db.commit()


async def test_failed_only_reply_is_visible_without_redefining_containment(case: _Case) -> None:
    async with case.maker() as db:
        db.add(
            Message(
                session_id=1,
                role=MessageRole.ASSISTANT,
                content="An interrupted partial reply.",
                status=MessageStatus.FAILED,
                created_at=_RECENT,
            )
        )
        await db.commit()
    stats = await _snapshot(case, (1, 0, 1))
    assert stats["sessions_served"] == 1
    assert stats["sessions_resolved"] == 0
    assert stats["sessions_reopened"] == 0
    assert stats["containment_rate"] == 1.0


@pytest.mark.parametrize(
    "first,last,expected",
    [
        (MessageStatus.FAILED, MessageStatus.COMPLETED, (1, 1, 0)),
        (MessageStatus.COMPLETED, MessageStatus.FAILED, (1, 1, 1)),
        (MessageStatus.FAILED, MessageStatus.PENDING, (1, 0, 0)),
        (MessageStatus.FAILED, MessageStatus.PROCESSING, (1, 0, 0)),
        (MessageStatus.FAILED, "UNKNOWN_FUTURE_STATUS", (1, 0, 0)),
    ],
)
async def test_latest_status_is_independent_of_having_a_failed_or_completed_reply(
    case: _Case,
    first: MessageStatus,
    last: MessageStatus | str,
    expected: tuple[int, int, int],
) -> None:
    await _reply(case, first, at=_RECENT)
    await _reply(case, last, at=_RECENT + timedelta(seconds=1))
    stats = await _snapshot(case, expected)
    assert stats["sessions_served"] == 1
    assert stats["sessions_resolved"] == 0
    assert stats["containment_rate"] == 1.0


@pytest.mark.parametrize("newest_status", [MessageStatus.COMPLETED, MessageStatus.FAILED])
async def test_equal_created_times_use_the_higher_id_not_insertion_order(
    case: _Case, newest_status: MessageStatus
) -> None:
    older_status = (
        MessageStatus.FAILED
        if newest_status == MessageStatus.COMPLETED
        else MessageStatus.COMPLETED
    )
    # Insert the greater ID first, so insertion order is the wrong oracle.
    await _reply(case, newest_status, message_id=20)
    await _reply(case, older_status, message_id=10)
    await _snapshot(case, (1, 1, int(newest_status == MessageStatus.FAILED)))


async def test_user_and_system_statuses_do_not_change_assistant_latest_or_cohorts(
    case: _Case,
) -> None:
    await _reply(case, MessageStatus.COMPLETED)
    await _reply(
        case,
        MessageStatus.FAILED,
        role=MessageRole.USER,
        at=_RECENT + timedelta(seconds=1),
        rating=-1,
    )
    await _reply(
        case, MessageStatus.FAILED, role=MessageRole.SYSTEM, at=_RECENT + timedelta(seconds=2)
    )
    stats = await _snapshot(case, (0, 1, 0))
    assert stats["sessions_served"] == 1
    assert stats["sessions_downvoted"] == 0
    assert stats["containment_rate"] == 1.0


async def test_cutoff_is_inclusive_and_old_completion_does_not_hide_window_failure(
    case: _Case,
) -> None:
    boundary = _SINCE.replace(tzinfo=None)
    await _reply(case, MessageStatus.COMPLETED, at=boundary - timedelta(microseconds=1))
    await _reply(case, MessageStatus.FAILED, at=boundary)
    await _reply(
        case, MessageStatus.COMPLETED, session_id=2, at=boundary + timedelta(microseconds=1)
    )
    await _reply(case, MessageStatus.FAILED, session_id=3, at=boundary - timedelta(microseconds=1))
    stats = await _snapshot(case, (1, 1, 1))
    assert stats["sessions_served"] == 2
    assert stats["containment_rate"] == 1.0


async def test_repeated_failed_messages_count_the_session_once(case: _Case) -> None:
    for offset in range(3):
        await _reply(case, MessageStatus.FAILED, at=_RECENT + timedelta(seconds=offset))
    stats = await _snapshot(case, (1, 0, 1))
    assert stats["sessions_served"] == 1


async def test_empty_window_zeros_new_gauges_and_keeps_the_existing_undefined_proxy(
    case: _Case,
) -> None:
    await _reply(case, MessageStatus.FAILED)
    await _snapshot(case, (1, 0, 1))
    async with case.maker() as db:
        await db.execute(delete(Message))
        await db.commit()
    await _reply(case, MessageStatus.FAILED, role=MessageRole.USER)
    stats = await _snapshot(case, (0, 0, 0))
    assert stats["sessions_served"] == 0
    assert stats["containment_rate"] is None
    assert REGISTRY.get_sample_value("chat_containment_rate") == 1.0


async def test_ticket_downvote_union_and_resolution_evidence_remain_independent(
    case: _Case,
) -> None:
    await _reply(case, MessageStatus.FAILED, session_id=1)
    await _reply(case, MessageStatus.COMPLETED, session_id=2, rating=-1)
    await _reply(case, MessageStatus.FAILED, session_id=3)
    await _reply(
        case, MessageStatus.COMPLETED, session_id=3, at=_RECENT + timedelta(seconds=1), rating=-1
    )
    async with case.maker() as db:
        session = await db.get(ChatSession, 3)
        assert session is not None
        session.resolved_at = _NOW - timedelta(days=20)
        session.reopened_count = 2
        db.add_all(
            [
                HandoffTicket(
                    session_id=3,
                    user_id=1,
                    reason="explicit",
                    summary="{}",
                    status=status,
                    created_at=(_NOW - timedelta(days=20)).replace(tzinfo=None),
                )
                for status in ("resolved", "open")
            ]
        )
        await db.commit()
    stats = await _snapshot(case, (2, 2, 1))
    assert stats["sessions_served"] == 3
    assert stats["sessions_escalated"] == 1
    assert stats["sessions_downvoted"] == 2
    assert stats["containment_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert stats["sessions_resolved"] == 1
    assert stats["sessions_reopened"] == 1
    async with case.maker() as db:
        session = await db.get(ChatSession, 3)
        assert session is not None
        assert session.resolved_at is not None and session.reopened_count == 2


async def test_requested_window_applies_to_all_new_cohorts_and_exported_counts(case: _Case) -> None:
    await _reply(case, MessageStatus.FAILED, at=(_NOW - timedelta(days=2)).replace(tzinfo=None))
    await _reply(case, MessageStatus.COMPLETED)
    seven_days = await _snapshot(case, (1, 1, 0), days=7)
    one_day = await _snapshot(case, (0, 1, 0), days=1)
    assert seven_days["sessions_served"] == one_day["sessions_served"] == 1
    assert one_day["window_start"] == (_NOW - timedelta(days=1)).isoformat()


async def test_database_failure_preserves_existing_and_new_gauge_samples(case: _Case) -> None:
    await _reply(case, MessageStatus.FAILED)
    await _snapshot(case, (1, 0, 1))
    names = [f"chat_{field}" for field in _FIELDS] + [
        "chat_containment_rate",
        "chat_sessions_served",
        "chat_sessions_escalated",
        "chat_containment_stats_last_success_timestamp_seconds",
    ]
    previous = {name: REGISTRY.get_sample_value(name) for name in names}
    assert all(value is not None for value in previous.values())
    async with case.maker() as db:
        await db.execute(text("DROP TABLE messages"))
        await db.commit()
    refreshed = await containment_metrics.refresh_containment_stats(case.maker, window_days=7)
    assert refreshed is None
    assert {name: REGISTRY.get_sample_value(name) for name in names} == previous
