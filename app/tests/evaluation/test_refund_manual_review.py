"""A confirmed high-value refund must leave an actual staff work item.

The graph, registry, handoff service and message persistence are real. SQLite
has foreign keys enabled; no model, Redis, vector service or business API runs.
"""

import json
import re
import socket
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import event, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_db
from app.api.v1 import chat as chat_api
from app.config.settings import settings
from app.core.security import create_access_token
from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.ticket import PRIORITY_HIGH, PRIORITY_NORMAL, HandoffTicket
from app.models.database.user import User
from app.models.enums.message import MessageRole
from app.services.chat.chat_service import HEARTBEAT, STREAM_ERROR, ChatService
from app.services.chat.persistence import ChatMessagePersister
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.tools import MOCK_ORDERS, ToolRegistry, create_default_tool_registry
from app.services.handoff.service import HandoffService
from app.services.intent.rule_based import RuleBasedIntentDetector

_TURNS = {
    "zh": ["我要退款", "ORD3001", "商品存在质量问题", "确认"],
    "en": ["I want a refund", "ORD3001", "damaged item", "confirm"],
}


@dataclass
class _Scenario:
    maker: async_sessionmaker[AsyncSession]
    handoff: HandoffService
    registry: ToolRegistry
    chat: ChatService
    refund_calls: list[dict[str, Any]]


def _chat(
    maker: async_sessionmaker[AsyncSession],
    registry: ToolRegistry,
    handoff: HandoffService | None,
) -> ChatService:
    return ChatService(
        graph=build_dialogue_graph(
            intent_detector=RuleBasedIntentDetector(),
            slot_filler=None,
            tool_registry=registry,
            retrieval_pipeline=None,
            llm_service=None,
            checkpointer=MemorySaver(),
            handoff_service=handoff,
        ),
        persister=ChatMessagePersister(maker),
    )


@pytest.fixture
async def scenario(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[_Scenario, None]:
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "CHAT_TRACE_STREAM_ENABLED", False)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr(settings, "SECRET_KEY", "isolated-manual-review-signing-key-32-bytes")
    monkeypatch.setitem(
        MOCK_ORDERS,
        "ORD3002",
        {**MOCK_ORDERS["ORD3001"], "total_amount": 299.0},
    )

    def forbid_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("manual-review evaluation must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid_network)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/manual-review.sqlite3")

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
        db.add_all(
            [
                User(id=1, email="small-refund@evaluation.invalid", hashed_password="!"),
                User(id=3, email="large-refund@evaluation.invalid", hashed_password="!"),
                User(id=4, email="staff@evaluation.invalid", hashed_password="!", is_admin=True),
            ]
        )
        await db.flush()
        db.add_all([ChatSession(id=1, user_id=3), ChatSession(id=2, user_id=1)])
        await db.commit()
    registry = create_default_tool_registry()
    refund = registry.get_tool_for_intent("refund")
    assert refund is not None
    original = refund.handler
    refund_calls: list[dict[str, Any]] = []

    def recorded_refund(args: dict[str, Any]) -> Any:
        refund_calls.append(dict(args))
        return original(args)

    monkeypatch.setattr(refund, "handler", recorded_refund)
    handoff = HandoffService(maker)
    chat = _chat(maker, registry, handoff)
    try:
        yield _Scenario(maker, handoff, registry, chat, refund_calls)
    finally:
        assert chat.persister is not None
        await chat.persister.drain()
        await engine.dispose()


async def _send(
    case: _Scenario,
    message: str,
    transport: str,
    *,
    session_id: int = 1,
    user_id: int = 3,
) -> tuple[str, dict[str, Any]]:
    if transport == "normal":
        response = await case.chat.process_message(session_id, message, user_id)
        content = response.content
    else:
        chunks = [
            chunk
            async for chunk in case.chat.process_message_stream(
                session_id, message, user_id, include_citations=True
            )
        ]
        assert STREAM_ERROR not in chunks
        content = "".join(
            chunk for chunk in chunks if isinstance(chunk, str) and chunk != HEARTBEAT
        )
    async with case.maker() as db:
        saved = await db.scalar(
            select(Message)
            .where(Message.session_id == session_id, Message.role == MessageRole.ASSISTANT)
            .order_by(Message.id.desc())
            .limit(1)
        )
    assert saved is not None
    assert saved.content == content
    metadata = saved.message_metadata or {}
    if transport == "normal":
        assert metadata.get("handoff_outcome") == (response.metadata or {}).get("handoff_outcome")
        assert metadata.get("handoff_ticket_id") == (response.metadata or {}).get(
            "handoff_ticket_id"
        )
    return content, metadata


async def _tickets(case: _Scenario) -> list[HandoffTicket]:
    async with case.maker() as db:
        return list((await db.scalars(select(HandoffTicket).order_by(HandoffTicket.id))).all())


def _honest_review_copy(content: str, language: str) -> None:
    for promise in (
        "退款成功",
        "已为您退款",
        "退款已完成",
        "客服已经接手",
        "refund has been processed",
        "has already been refunded",
        "an agent has picked up",
    ):
        assert promise not in content.casefold()
    assert not re.search(r"\bRF\d+", content)
    if language == "en":
        assert not re.search(r"[一-鿿]", content)


@pytest.mark.parametrize("language", ["zh", "en"])
@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_confirmed_large_refund_creates_a_real_priority_review_ticket(
    scenario: _Scenario, language: str, transport: str
) -> None:
    for message in _TURNS[language][:-1]:
        await _send(scenario, message, transport)
    assert await _tickets(scenario) == []
    assert scenario.refund_calls == []

    content, metadata = await _send(scenario, _TURNS[language][-1], transport)
    tickets = await _tickets(scenario)
    assert len(tickets) == 1
    ticket = tickets[0]
    assert (ticket.session_id, ticket.user_id) == (1, 3)
    assert (ticket.reason, ticket.priority, ticket.status) == (
        "refund_threshold",
        PRIORITY_HIGH,
        "open",
    )
    assert ticket.assigned_to is None and ticket.claimed_at is None
    assert metadata["handoff_outcome"] == "queued"
    assert metadata["handoff_ticket_id"] == ticket.id
    assert f"#{ticket.id}" in content
    _honest_review_copy(content, language)
    summary = json.dumps(json.loads(ticket.summary), ensure_ascii=False)
    for context in ("ORD3001", "1299", "process_refund", scenario.refund_calls[0]["reason"]):
        assert str(context) in summary
    assert len(scenario.refund_calls) == 1
    state = (await scenario.chat.graph.aget_state({"configurable": {"thread_id": "1"}})).values
    assert state["task_executed"] is True
    assert state["pending_confirmation"] is None
    assert metadata["executed_tools"][0]["tool"] == "process_refund"


@pytest.mark.parametrize("language", ["zh", "en"])
@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_refused_ticket_write_is_honest_and_explicit_retry_does_not_repeat_refund(
    scenario: _Scenario, language: str, transport: str
) -> None:
    async with scenario.maker() as db:
        await db.execute(
            text(
                "CREATE TRIGGER refuse_review BEFORE INSERT ON handoff_tickets "
                "BEGIN SELECT RAISE(ABORT, 'isolated review-ticket write refusal'); END"
            )
        )
        await db.commit()
    for message in _TURNS[language][:-1]:
        await _send(scenario, message, transport)
    content, metadata = await _send(scenario, _TURNS[language][-1], transport)
    assert await _tickets(scenario) == []
    assert metadata["handoff_outcome"] == "failed"
    assert not metadata.get("handoff_ticket_id")
    assert "#" not in content
    assert not re.search(r"第\s*\d+\s*位|position.*:\s*\d+", content, re.IGNORECASE)
    assert ("未能" in content) if language == "zh" else ("couldn't" in content.casefold())
    _honest_review_copy(content, language)
    assert len(scenario.refund_calls) == 1
    async with scenario.maker() as db:
        await db.execute(text("DROP TRIGGER refuse_review"))
        await db.commit()
    # Another tool may replace the live result/audit before the customer
    # retries staff review. The failed refund request must remain intact.
    await _send(
        scenario,
        "查询订单ORD3002" if language == "zh" else "check order ORD3002",
        transport,
    )
    intervening = (
        await scenario.chat.graph.aget_state({"configurable": {"thread_id": "1"}})
    ).values
    assert intervening["executed_tools"][0]["tool"] == "query_order_status"
    assert intervening["executed_tools"][0]["args"]["order_id"] == "ORD3002"
    retry, retry_metadata = await _send(
        scenario, "转人工" if language == "zh" else "human agent", transport
    )
    tickets = await _tickets(scenario)
    assert len(tickets) == 1
    assert retry_metadata["handoff_outcome"] == "queued"
    assert retry_metadata["handoff_ticket_id"] == tickets[0].id
    assert (tickets[0].reason, tickets[0].priority) == ("refund_threshold", PRIORITY_HIGH)
    summary = json.dumps(json.loads(tickets[0].summary), ensure_ascii=False)
    for context in ("ORD3001", "1299", "process_refund", scenario.refund_calls[0]["reason"]):
        assert str(context) in summary
    assert f"#{tickets[0].id}" in retry
    assert len(scenario.refund_calls) == 1

    # Once accepted, the pending refund review must not reclassify an
    # unrelated future staff request as another urgent refund escalation.
    await scenario.handoff.resolve_ticket(tickets[0].id, 4)
    _content, future_metadata = await _send(
        scenario, "转人工" if language == "zh" else "human agent", transport
    )
    future = await _tickets(scenario)
    assert len(future) == 2
    assert future_metadata["handoff_ticket_id"] == future[-1].id
    assert (future[-1].reason, future[-1].priority) == ("explicit", PRIORITY_NORMAL)
    assert len(scenario.refund_calls) == 1


@pytest.mark.parametrize("claimed", [False, True])
@pytest.mark.parametrize("existing_reason", ["explicit", "emotion"])
async def test_existing_active_request_is_reused_and_promoted_without_losing_assignment(
    scenario: _Scenario, claimed: bool, existing_reason: str
) -> None:
    existing = await scenario.handoff.create_ticket_for_session(
        1, 3, existing_reason, {"original_context": "keep this original request"}
    )
    ticket_id = existing["ticket_id"]
    if claimed:
        await scenario.handoff.claim_ticket(ticket_id, 4)
    for message in _TURNS["en"]:
        content, metadata = await _send(scenario, message, "normal")
    tickets = await _tickets(scenario)
    assert len(tickets) == 1
    ticket = tickets[0]
    assert ticket.id == ticket_id == metadata["handoff_ticket_id"]
    assert ticket.priority == PRIORITY_HIGH
    assert ticket.reason == ("emotion" if existing_reason == "emotion" else "refund_threshold")
    assert metadata["handoff_outcome"] == "queued"
    summary = json.dumps(json.loads(ticket.summary), ensure_ascii=False)
    assert "keep this original request" in summary and "ORD3001" in summary
    assert "process_refund" in summary and "1299" in summary
    assert "refund_threshold" in json.loads(ticket.summary)["escalation_contexts"]
    assert ticket.status == ("claimed" if claimed else "open")
    assert ticket.assigned_to == (4 if claimed else None)
    if claimed:
        assert ticket.claimed_at is not None
        assert not re.search(r"position.*:\s*\d+", content, re.IGNORECASE)
    _honest_review_copy(content, "en")
    assert len(scenario.refund_calls) == 1


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_small_refund_still_requires_confirmation_and_does_not_escalate(
    scenario: _Scenario, transport: str
) -> None:
    for message in ("I want a refund", "ORD1001", "damaged item"):
        await _send(scenario, message, transport, session_id=2, user_id=1)
    assert scenario.refund_calls == []
    _content, metadata = await _send(scenario, "yes", transport, session_id=2, user_id=1)
    assert len(scenario.refund_calls) == 1
    assert await _tickets(scenario) == []
    assert not metadata.get("handoff_outcome")
    assert not metadata.get("handoff_ticket_id")
    state = (await scenario.chat.graph.aget_state({"configurable": {"thread_id": "2"}})).values
    assert state["tool_result"]["status"] == "success"
    assert re.fullmatch(r"RF\d+", state["tool_result"]["refund_id"])


async def test_agent_style_staged_action_reaches_the_same_confirmed_review_path(
    scenario: _Scenario,
) -> None:
    # The Agent loop's staged-action contract is persisted in the graph;
    # no model is invoked to produce that contract in this closure test.
    await scenario.chat.graph.aupdate_state(
        {"configurable": {"thread_id": "1"}},
        {
            "intent": "refund",
            "pending_confirmation": {
                "intent": "refund",
                "args": {"order_id": "ORD3001", "reason": "damaged item"},
                "staged_at": time.time(),
            },
        },
    )
    content, metadata = await _send(scenario, "confirm", "normal")
    tickets = await _tickets(scenario)
    assert len(tickets) == 1
    assert tickets[0].reason == "refund_threshold"
    assert metadata["handoff_ticket_id"] == tickets[0].id
    assert metadata["handoff_outcome"] == "queued"
    assert len(scenario.refund_calls) == 1
    _honest_review_copy(content, "en")


async def test_missing_handoff_service_cannot_confirm_large_refund_review(
    scenario: _Scenario,
) -> None:
    scenario.chat = _chat(scenario.maker, scenario.registry, None)
    for message in _TURNS["en"]:
        content, metadata = await _send(scenario, message, "normal")
    assert await _tickets(scenario) == []
    assert metadata["handoff_outcome"] == "failed"
    assert not metadata.get("handoff_ticket_id")
    assert "couldn't" in content.casefold()
    assert "#" not in content
    assert len(scenario.refund_calls) == 1
    _honest_review_copy(content, "en")


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_review_failure_does_not_substitute_old_order_for_a_foreign_query(
    scenario: _Scenario, transport: str
) -> None:
    async with scenario.maker() as db:
        await db.execute(
            text(
                "CREATE TRIGGER refuse_review BEFORE INSERT ON handoff_tickets "
                "BEGIN SELECT RAISE(ABORT, 'isolated review-ticket write refusal'); END"
            )
        )
        await db.commit()
    for message in _TURNS["en"]:
        await _send(scenario, message, transport)
    content, _metadata = await _send(scenario, "check order ORD1001", transport)
    state = (await scenario.chat.graph.aget_state({"configurable": {"thread_id": "1"}})).values
    query = state["executed_tools"][0]
    assert query["tool"] == "query_order_status"
    assert query["args"]["order_id"] == "ORD1001"
    assert query["ok"] is False
    assert "error" in state["tool_result"]
    assert "not accessible" in content.casefold()
    assert len(scenario.refund_calls) == 1
    assert await _tickets(scenario) == []
    async with scenario.maker() as db:
        await db.execute(text("DROP TRIGGER refuse_review"))
        await db.commit()
    _retry, metadata = await _send(scenario, "human agent", transport)
    tickets = await _tickets(scenario)
    assert len(tickets) == 1
    assert metadata["handoff_ticket_id"] == tickets[0].id
    assert tickets[0].reason == "refund_threshold"
    assert tickets[0].priority == PRIORITY_HIGH
    summary = json.dumps(json.loads(tickets[0].summary), ensure_ascii=False)
    assert "ORD3001" in summary and "1299" in summary and "process_refund" in summary
    assert len(scenario.refund_calls) == 1


@pytest.mark.parametrize("transport", ["normal", "stream"])
@pytest.mark.parametrize("abandon", ["cancel", "no"])
async def test_abandoned_failed_review_does_not_reclassify_future_staff_request(
    scenario: _Scenario, transport: str, abandon: str
) -> None:
    async with scenario.maker() as db:
        await db.execute(
            text(
                "CREATE TRIGGER refuse_review BEFORE INSERT ON handoff_tickets "
                "BEGIN SELECT RAISE(ABORT, 'isolated review-ticket write refusal'); END"
            )
        )
        await db.commit()
    for message in _TURNS["en"]:
        await _send(scenario, message, transport)
    assert await _tickets(scenario) == []
    assert len(scenario.refund_calls) == 1
    async with scenario.maker() as db:
        await db.execute(text("DROP TRIGGER refuse_review"))
        await db.commit()
    await _send(scenario, abandon, transport)
    _content, metadata = await _send(scenario, "human agent", transport)
    tickets = await _tickets(scenario)
    assert len(tickets) == 1
    assert (tickets[0].reason, tickets[0].priority) == ("explicit", PRIORITY_NORMAL)
    assert metadata["handoff_outcome"] == "queued"
    assert metadata["handoff_ticket_id"] == tickets[0].id
    assert len(scenario.refund_calls) == 1


@pytest.mark.parametrize("transport", ["normal", "stream"])
async def test_authenticated_http_confirmation_publishes_the_durable_review_once(
    scenario: _Scenario, transport: str
) -> None:
    # Establish the staged request through the real service. Only its
    # final confirmation crosses HTTP here; ownership/auth are not faked.
    for message in _TURNS["en"][:-1]:
        await _send(scenario, message, "normal")
    app = FastAPI()
    app.include_router(chat_api.router, prefix="/api/v1")

    async def isolated_db() -> AsyncGenerator[AsyncSession, None]:
        async with scenario.maker() as db:
            yield db

    app.dependency_overrides[get_db] = isolated_db
    app.dependency_overrides[chat_api.get_chat_service] = lambda: scenario.chat
    token = create_access_token({"sub": "large-refund@evaluation.invalid", "user_id": 3})
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://isolated-review.invalid"
    ) as client:
        response = await client.post(
            "/api/v1/chat" if transport == "normal" else "/api/v1/chat/stream",
            headers={"Authorization": f"Bearer {token}"},
            json={"session_id": 1, "message": "confirm", "include_citations": True},
        )
    assert response.status_code == 200
    tickets = await _tickets(scenario)
    assert len(tickets) == 1
    ticket = tickets[0]
    assert (ticket.session_id, ticket.user_id, ticket.reason, ticket.priority) == (
        1,
        3,
        "refund_threshold",
        PRIORITY_HIGH,
    )
    if transport == "normal":
        body = response.json()
        content = body["content"]
        assert body["metadata"]["handoff_outcome"] == "queued"
        assert body["metadata"]["handoff_ticket_id"] == ticket.id
    else:
        # Independent SSE consumer: content can contain newlines, and
        # named citation events are not text emitted to the customer.
        text_parts = []
        terminals = []
        for frame in response.text.replace("\r\n", "\n").split("\n\n"):
            lines = frame.split("\n")
            event_type = next(
                (line[6:].strip() for line in lines if line.startswith("event:")), "message"
            )
            data = "\n".join(
                line[5:].removeprefix(" ") for line in lines if line.startswith("data:")
            )
            if event_type != "message" or not data:
                continue
            if data in {"[DONE]", "[STREAM_ERROR]"}:
                terminals.append(data)
            else:
                text_parts.append(data)
        assert terminals == ["[DONE]"]
        content = "".join(text_parts)
    assert content.count(f"ticket #{ticket.id}") == 1
    assert "no refund has been issued" in content
    _honest_review_copy(content, "en")
    async with scenario.maker() as db:
        saved = await db.scalar(
            select(Message)
            .where(Message.session_id == 1, Message.role == MessageRole.ASSISTANT)
            .order_by(Message.id.desc())
            .limit(1)
        )
    assert saved is not None and saved.content == content
    assert (saved.message_metadata or {})["handoff_outcome"] == "queued"
    assert (saved.message_metadata or {})["handoff_ticket_id"] == ticket.id
    assert len(scenario.refund_calls) == 1
