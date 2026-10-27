"""The probe runner observes real local chat routes and durable outcomes.

The default runner only sees server-reported public responses. Direct SQLite
and a separate local staff client are independent test observations; neither
establishes live model semantics or an online human agent.
"""

import json
import re
import socket
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import event, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_db
from app.api.v1 import auth as auth_api
from app.api.v1 import chat as chat_api
from app.api.v1 import handoff as handoff_api
from app.api.v1 import sessions as sessions_api
from app.config.settings import settings
from app.core.security import create_access_token
from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.models.database.user import User
from app.services.chat.chat_service import ChatService
from app.services.chat.persistence import ChatMessagePersister
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.tools import create_default_tool_registry
from app.services.embeddings.base import EmbeddingResult
from app.services.faq.store import DEFAULT_FAQ_FILE, FAQEntry, FAQService, load_faq_entries
from app.services.guardrails.factory import GuardrailFactory
from app.services.handoff.service import HandoffService
from app.services.intent.rule_based import RuleBasedIntentDetector
from app.services.llm.base import LLMMessage, LLMResponse, LLMServiceBase
from scripts import probe_live as probe

_KNOWLEDGE_TURNS = {
    "退货政策是什么",
    "what is the return policy",
    "and how long do refunds take",
}


class _LocalFAQEmbeddings:
    def __init__(self, entries: list[FAQEntry]) -> None:
        self.dimensions = len(entries)
        self.lookup = {
            value.casefold(): index
            for index, entry in enumerate(entries)
            for value in (entry.question, *entry.variants)
        }
        by_id = {entry.faq_id: index for index, entry in enumerate(entries)}
        self.lookup["what is the return policy"] = by_id["returns_policy_en"]
        self.lookup["and how long do refunds take"] = by_id["refund_timeline_en"]

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        vectors = []
        for value in texts:
            vector = [0.0] * self.dimensions
            index = self.lookup.get(value.casefold())
            if index is not None:
                vector[index] = 1.0
            vectors.append(vector)
        return EmbeddingResult(
            embeddings=vectors,
            model="local-probe-faq-fixture",
            dimensions=self.dimensions,
            tokens_used=0,
        )


class _UncitedPolicyLLM(LLMServiceBase):
    """Intentionally ungrounded provider: a numeric anchor is no evidence."""

    def __init__(self) -> None:
        super().__init__(api_key="offline", model="local-uncited-policy-fixture")
        self.calls = 0

    async def generate(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls += 1
        question = messages[-1].content
        if "refunds take" in question.casefold():
            answer = "Please review your refund status."
        elif re.search(r"[一-鿿]", question):
            answer = "您可以在签收后7天内申请无理由退货。"
        else:
            answer = "Our return policy allows returns within 7 days of delivery."
        return LLMResponse(content=answer, model=self.model)

    async def generate_stream(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        yield (await self.generate(messages, max_tokens, temperature, **kwargs)).content

    def estimate_tokens(self, value: str) -> int:
        return len(value)

    async def count_tokens(self, messages: list[LLMMessage]) -> int:
        return sum(len(message.content) for message in messages)


@dataclass
class _RouteCase:
    mode: str
    maker: async_sessionmaker[AsyncSession]
    client: AsyncClient
    staff: AsyncClient
    entries: dict[str, FAQEntry]
    provider: _UncitedPolicyLLM
    customer_headers: dict[str, str]
    replies: list[dict[str, Any]] = field(default_factory=list)
    requests: list[tuple[str, str, str]] = field(default_factory=list)


@pytest.fixture
async def route_case(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[_RouteCase, None]:
    mode = getattr(request, "param", "healthy")
    monkeypatch.setattr(settings, "DEMO_MODE", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr(settings, "CHAT_TRACE_STREAM_ENABLED", False)
    monkeypatch.setattr(settings, "SECRET_KEY", "isolated-live-probe-signing-key-32-bytes")
    attempts: list[str] = []

    def forbid_network(_socket: Any, address: Any, *args: Any, **kwargs: Any) -> None:
        attempts.append(str(address))
        raise AssertionError("live-probe evaluation must not open an external connection")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid_network)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/live-probe.sqlite3")

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
        if mode == "ticket_rejected":
            await connection.execute(
                text(
                    "CREATE TRIGGER refuse_probe_ticket BEFORE INSERT ON handoff_tickets "
                    "BEGIN SELECT RAISE(ABORT, 'isolated probe-ticket refusal'); END"
                )
            )
    async with maker() as db:
        db.add_all(
            [
                User(id=1, email="probe-customer@example.com", hashed_password="!"),
                User(id=2, email="probe-staff@example.com", hashed_password="!", is_admin=True),
            ]
        )
        await db.commit()
    entries = [
        entry
        for entry in load_faq_entries(DEFAULT_FAQ_FILE)
        if entry.faq_id in {"returns_policy", "returns_policy_en", "refund_timeline_en"}
    ]
    provider = _UncitedPolicyLLM()
    handoff = HandoffService(maker)
    persister = ChatMessagePersister(maker)
    chat = ChatService(
        graph=build_dialogue_graph(
            intent_detector=RuleBasedIntentDetector(),
            slot_filler=None,
            tool_registry=create_default_tool_registry(),
            retrieval_pipeline=None,
            llm_service=provider if mode == "uncited_policy" else None,
            faq_service=(
                None
                if mode == "uncited_policy"
                else FAQService(_LocalFAQEmbeddings(entries), entries)
            ),
            guardrail_service=GuardrailFactory.create(),
            checkpointer=MemorySaver(),
            handoff_service=handoff,
        ),
        persister=persister,
    )
    app = FastAPI()
    for router in (auth_api.router, chat_api.router, handoff_api.router, sessions_api.router):
        app.include_router(router, prefix="/api/v1")

    async def isolated_db() -> AsyncGenerator[AsyncSession, None]:
        async with maker() as db:
            yield db

    app.dependency_overrides[get_db] = isolated_db
    app.dependency_overrides[chat_api.get_chat_service] = lambda: chat
    app.dependency_overrides[handoff_api.get_handoff_service] = lambda: handoff
    replies: list[dict[str, Any]] = []
    requests: list[tuple[str, str, str]] = []

    async def observe(response: Response) -> None:
        requests.append(("customer", response.request.method, response.request.url.path))
        if response.request.method == "POST" and response.request.url.path == "/api/v1/chat":
            await response.aread()
            payload = json.loads(response.request.content)
            body = response.json()
            assert isinstance(payload, dict) and isinstance(body, dict)
            replies.append({"request": payload, "response": body})

    async def observe_staff(response: Response) -> None:
        requests.append(("staff", response.request.method, response.request.url.path))

    customer = create_access_token({"sub": "probe-customer@example.com", "user_id": 1})
    staff = create_access_token({"sub": "probe-staff@example.com", "user_id": 2})
    try:
        async with (
            AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://isolated-probe.invalid",
                event_hooks={"response": [observe]},
            ) as client,
            AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://isolated-probe.invalid",
                headers={"Authorization": f"Bearer {staff}"},
                event_hooks={"response": [observe_staff]},
            ) as staff_client,
        ):
            yield _RouteCase(
                mode,
                maker,
                client,
                staff_client,
                {entry.faq_id: entry for entry in entries},
                provider,
                {"Authorization": f"Bearer {customer}"},
                replies,
                requests,
            )
    finally:
        await persister.drain()
        await engine.dispose()
        assert attempts == []


async def _durable_outcomes(case: _RouteCase) -> tuple[list[HandoffTicket], list[Message]]:
    async with case.maker() as db:
        tickets = list((await db.scalars(select(HandoffTicket).order_by(HandoffTicket.id))).all())
        messages = list((await db.scalars(select(Message).order_by(Message.id))).all())
    return tickets, messages


@pytest.mark.parametrize(
    "route_case", ["healthy", "ticket_rejected", "uncited_policy"], indirect=True
)
async def test_full_runner_reports_real_local_route_outcomes(route_case: _RouteCase) -> None:
    cases = await probe.run_probes(route_case.client)
    tickets, messages = await _durable_outcomes(route_case)
    assert len(cases) == 21
    failures = {case.name: case.failures for case in cases if case.failures}
    if route_case.mode == "healthy":
        assert failures == {}
        assert len(route_case.replies) == 24
        assert len(messages) == 48
        assert len(tickets) == 17
        by_id = {ticket.id: ticket for ticket in tickets}
        queued = [
            reply
            for reply in route_case.replies
            if reply["response"]["metadata"].get("handoff_outcome") == "queued"
        ]
        assert len(queued) == 17
        for reply in queued:
            ticket_id = reply["response"]["metadata"]["handoff_ticket_id"]
            ticket = by_id[ticket_id]
            assert ticket.session_id == reply["request"]["session_id"]
            assert ticket.status == "open"
            assert ticket.assigned_to is None and ticket.claimed_at is None
        knowledge = [
            reply for reply in route_case.replies if reply["request"]["message"] in _KNOWLEDGE_TURNS
        ]
        assert len(knowledge) == 3
        for reply in knowledge:
            body = reply["response"]
            assert len(body["citations"]) == 1
            citation = body["citations"][0]
            entry = route_case.entries[citation["faq_id"]]
            assert citation["source_type"] == "faq"
            assert citation["excerpt"] == body["content"] == entry.answer
        assert route_case.provider.calls == 0
        no_staff = await route_case.client.get("/api/v1/handoff/tickets")
        assert no_staff.status_code == 401
        observed = await route_case.staff.get("/api/v1/handoff/tickets")
        assert observed.status_code == 200
        assert {row["id"] for row in observed.json()["tickets"]} == set(by_id)
    elif route_case.mode == "ticket_rejected":
        assert len(failures) == 17
        assert tickets == []
        handoff_replies = [
            reply
            for reply in route_case.replies
            if reply["response"]["metadata"].get("handoff_outcome") == "failed"
        ]
        assert len(handoff_replies) == 17
        assert all(
            not row["response"]["metadata"].get("handoff_ticket_id") for row in handoff_replies
        )
        assert route_case.provider.calls == 0
    else:
        # The old harness passed numeric prose without observing any
        # reference. This is fresh local generation, not a real-model eval.
        assert "knowledge-zh" in failures
        assert "knowledge-en" in failures
        knowledge = [
            reply for reply in route_case.replies if reply["request"]["message"] in _KNOWLEDGE_TURNS
        ]
        assert len(knowledge) >= 2
        assert all(not row["response"]["citations"] for row in knowledge)
        assert route_case.provider.calls >= 2


@pytest.mark.parametrize("authorized_observer", [True, False])
async def test_owned_runner_reads_owner_history_and_staff_ticket_evidence(
    route_case: _RouteCase,
    monkeypatch: pytest.MonkeyPatch,
    authorized_observer: bool,
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    route_case.client.headers.update(route_case.customer_headers)
    cases = await probe.run_probes(
        route_case.client,
        authenticated_sessions=True,
        ticket_client=route_case.staff if authorized_observer else route_case.client,
    )
    tickets, messages = await _durable_outcomes(route_case)
    async with route_case.maker() as db:
        sessions = list((await db.scalars(select(ChatSession))).all())
    assert len(cases) == len(sessions) == 21
    assert len(tickets) == 17 and len(messages) == 48
    assert all(session.user_id == 1 for session in sessions)
    assert {case.session_id for case in cases} == {session.id for session in sessions}
    assert all(case.authenticated_session and case.history_verified for case in cases)
    assert all(case.ticket_evidence_requested for case in cases)
    assert route_case.requests.count(("customer", "GET", "/api/v1/auth/me")) == 1
    assert route_case.requests.count(("customer", "POST", "/api/v1/sessions")) == 21
    assert route_case.requests.count(("customer", "GET", "/api/v1/chat/history")) == 21
    assert route_case.provider.calls == 0
    assert all(ticket.user_id == 1 and ticket.status == "open" for ticket in tickets)
    assert all(ticket.assigned_to is None and ticket.claimed_at is None for ticket in tickets)
    if authorized_observer:
        assert all(not case.failures for case in cases)
        expected = {(ticket.id, ticket.session_id) for ticket in tickets}
        observed = {
            (ticket_id, case.session_id) for case in cases for ticket_id in case.verified_ticket_ids
        }
        assert observed == expected
        staff_requests = [row for row in route_case.requests if row[0] == "staff"]
        assert staff_requests == [("staff", "GET", "/api/v1/handoff/tickets")]
        report = probe.format_report(cases)
        assert "21/21 cases passed" in report
        assert "21/21 verified" in report
        assert "17/17 ticket/session pairs verified" in report
    else:
        failed = [case for case in cases if case.failures]
        assert len(failed) == 17
        assert all(case.ticket_ids for case in failed)
        assert all(case.failures == ["staff ticket observation: HTTP 403"] for case in failed)
        assert all(not case.verified_ticket_ids for case in cases)
        assert not any(row[0] == "staff" for row in route_case.requests)
        assert "0/17 ticket/session pairs verified" in probe.format_report(cases)


async def test_main_uses_authenticated_client_for_real_owned_route_run(
    route_case: _RouteCase, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    route_case.client.headers.update(route_case.customer_headers)
    code = await probe.main([], client=route_case.client, ticket_client=route_case.staff)
    report = capsys.readouterr().out
    assert code == 0
    assert "21/21 cases passed" in report
    assert "Owned session history readback: 21/21 verified" in report
    assert "17/17 ticket/session pairs verified" in report
    assert "not full semantic correctness or human acceptance" in report
    assert "Bearer" not in report and "probe-customer@example.com" not in report
    assert route_case.requests.count(("customer", "POST", "/api/v1/sessions")) == 21
    assert route_case.requests.count(("customer", "GET", "/api/v1/chat/history")) == 21
    tickets, messages = await _durable_outcomes(route_case)
    assert len(tickets) == 17 and len(messages) == 48
    assert all(ticket.status == "open" and ticket.assigned_to is None for ticket in tickets)
