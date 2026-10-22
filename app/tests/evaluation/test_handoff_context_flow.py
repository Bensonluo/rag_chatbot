"""Strict customer-to-agent lifecycle with real graph, JWTs and SQLite.

The first turn uses a reviewed FAQ and deterministic local embeddings; neither
turn has an LLM. Authentication, ticket creation, assignment, persistence and
the agent's scoped transcript all run through their production implementations.
"""

import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import MemorySaver
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_db
from app.api.v1 import chat as chat_api
from app.api.v1 import handoff as handoff_api
from app.config.settings import settings
from app.core.security import create_access_token
from app.middleware.rate_limiter_redis import EndpointRateLimiter, _RedisClientHolder
from app.models.database.base import Base
from app.models.database.message import Message
from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.models.database.user import User
from app.services.chat.chat_service import ChatService
from app.services.chat.persistence import ChatMessagePersister
from app.services.dialogue.graph import build_dialogue_graph
from app.services.dialogue.tools import ToolRegistry
from app.services.embeddings.base import EmbeddingResult
from app.services.faq.store import FAQEntry, FAQService
from app.services.handoff.service import HandoffService
from app.services.intent.rule_based import RuleBasedIntentDetector

FAQ = FAQEntry(
    faq_id="handoff_probe_returns",
    question="退货政策是什么",
    answer="支持 7 天无理由退货。",
)


class _LocalFAQEmbeddings:
    """One local FAQ; matching has no network or downloaded model dependency."""

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        return EmbeddingResult(
            embeddings=[[1.0, 0.0] for _ in texts],
            model="isolated-faq-fixture",
            dimensions=2,
            tokens_used=0,
        )


async def test_assigned_agent_receives_real_owner_chat_context_until_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "DEMO_MODE", False)
    monkeypatch.setattr(settings, "SECRET_KEY", "isolated-handoff-context-signing-key-32bytes")
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)

    def unavailable_redis() -> None:
        raise RedisConnectionError("isolated handoff probe uses local rate-limit fallback")

    monkeypatch.setattr(_RedisClientHolder, "get", unavailable_redis)
    monkeypatch.setattr(
        chat_api,
        "_chat_rate_limiter",
        EndpointRateLimiter(scope="handoff-context-probe", requests_per_minute=10),
    )
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/handoff.sqlite3")

    @event.listens_for(engine.sync_engine, "connect")
    def enable_foreign_keys(connection: Any, _record: Any) -> None:
        connection.execute("PRAGMA foreign_keys=ON")

    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(
                lambda conn: Base.metadata.create_all(
                    conn,
                    tables=[
                        Base.metadata.tables["users"],
                        Base.metadata.tables["chat_sessions"],
                        Base.metadata.tables["messages"],
                        Base.metadata.tables["handoff_tickets"],
                    ],
                )
            )
        async with maker() as session:
            session.add_all(
                [
                    User(id=1, email="owner@context.invalid", hashed_password="!"),
                    User(
                        id=2, email="assigned@context.invalid", hashed_password="!", is_admin=True
                    ),
                    User(id=3, email="other@context.invalid", hashed_password="!", is_admin=True),
                ]
            )
            await session.flush()
            session.add(ChatSession(id=1, user_id=1, title="Handoff context probe"))
            await session.commit()

        persister = ChatMessagePersister(maker)
        handoff = HandoffService(maker)
        chat = ChatService(
            graph=build_dialogue_graph(
                intent_detector=RuleBasedIntentDetector(),
                slot_filler=None,
                tool_registry=ToolRegistry(),
                retrieval_pipeline=None,
                llm_service=None,
                faq_service=FAQService(_LocalFAQEmbeddings(), [FAQ]),
                checkpointer=MemorySaver(),
                handoff_service=handoff,
            ),
            persister=persister,
        )

        async def isolated_db() -> AsyncGenerator[AsyncSession, None]:
            async with maker() as session:
                yield session

        app = FastAPI()
        app.include_router(chat_api.router, prefix="/api/v1")
        app.include_router(handoff_api.router, prefix="/api/v1")
        app.dependency_overrides[get_db] = isolated_db
        app.dependency_overrides[chat_api.get_chat_service] = lambda: chat
        app.dependency_overrides[handoff_api.get_handoff_service] = lambda: handoff
        headers = {
            user_id: {
                "Authorization": "Bearer " + create_access_token({"sub": email, "user_id": user_id})
            }
            for user_id, email in (
                (1, "owner@context.invalid"),
                (2, "assigned@context.invalid"),
                (3, "other@context.invalid"),
            )
        }
        statuses: dict[str, int] = {}
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://handoff-context.invalid"
        ) as client:
            answer = await client.post(
                "/api/v1/chat",
                headers=headers[1],
                json={"session_id": 1, "message": FAQ.question},
            )
            statuses["owner_faq_chat"] = answer.status_code
            assert answer.status_code == 200, answer.text
            answer_body = answer.json()
            assert answer_body["content"] == FAQ.answer
            assert answer_body["metadata"]["llm_calls"] == 0
            assert len(answer_body["citations"]) == 1
            assert answer_body["citations"][0]["faq_id"] == FAQ.faq_id

            escalation = await client.post(
                "/api/v1/chat",
                headers=headers[1],
                json={"session_id": 1, "message": "human agent"},
            )
            statuses["owner_handoff_chat"] = escalation.status_code
            assert escalation.status_code == 200, escalation.text
            escalation_body = escalation.json()
            assert escalation_body["intent"] == "handoff"
            assert escalation_body["metadata"]["handoff_outcome"] == "queued"
            assert escalation_body["metadata"]["llm_calls"] == 0
            ticket_id = escalation_body["metadata"]["handoff_ticket_id"]
            history_path = f"/api/v1/handoff/tickets/{ticket_id}/history"

            for label, auth, expected_status in (
                ("anonymous_history", {}, 401),
                ("invalid_token_history", {"Authorization": "Bearer invalid-probe-token"}, 401),
                ("customer_history", headers[1], 403),
                ("unclaimed_history", headers[2], 404),
            ):
                response = await client.get(history_path, headers=auth)
                statuses[label] = response.status_code
                assert response.status_code == expected_status, (label, response.text)

            claimed = await client.post(
                f"/api/v1/handoff/tickets/{ticket_id}/claim", headers=headers[2]
            )
            statuses["claim"] = claimed.status_code
            assert claimed.status_code == 200, claimed.text
            assert claimed.json()["assigned_to"] == 2
            assert claimed.json()["status"] == "claimed"

            received = await client.get(history_path, headers=headers[2])
            statuses["assigned_history"] = received.status_code
            assert received.status_code == 200, received.text
            history = received.json()
            assert history["ticket_id"] == ticket_id
            assert history["session_id"] == 1
            assert history["skip"] == 0
            assert history["total"] == 4
            messages = history["messages"]
            assert [row["role"] for row in messages] == ["user", "assistant"] * 2
            assert [row["content"] for row in messages] == [
                FAQ.question,
                answer_body["content"],
                "human agent",
                escalation_body["content"],
            ]
            assert messages[1]["citations"] == answer_body["citations"]
            assert messages[1]["sources"] == answer_body["sources"]
            assert all(row["timestamp"] for row in messages)
            assert all(not row["citations"] for row in (messages[0], messages[2], messages[3]))

            unrelated = await client.get(history_path, headers=headers[3])
            statuses["other_agent_history"] = unrelated.status_code
            assert unrelated.status_code == 404
            generic = await client.get(
                "/api/v1/chat/history", params={"session_id": 1}, headers=headers[2]
            )
            statuses["assigned_generic_history"] = generic.status_code
            assert generic.status_code == 404
            owner = await client.get(
                "/api/v1/chat/history", params={"session_id": 1}, headers=headers[1]
            )
            statuses["owner_generic_history"] = owner.status_code
            assert owner.status_code == 200
            assert owner.json()["messages"][1]["citations"] == messages[1]["citations"]

            resolved = await client.post(
                f"/api/v1/handoff/tickets/{ticket_id}/resolve", headers=headers[2]
            )
            statuses["resolve"] = resolved.status_code
            assert resolved.status_code == 200, resolved.text
            assert resolved.json()["status"] == "resolved"
            revoked = await client.get(history_path, headers=headers[2])
            statuses["resolved_history"] = revoked.status_code
            assert revoked.status_code == 404

        async with maker() as session:
            ticket = await session.get(HandoffTicket, ticket_id)
            assert ticket is not None
            assert ticket.session_id == 1
            assert ticket.user_id == 1
            assert ticket.assigned_to == 2
            assert ticket.status == "resolved"
            rows = (await session.scalars(select(Message).order_by(Message.id))).all()
            assert [row.id for row in rows] == [row["id"] for row in messages]
            assert [row.content for row in rows] == [row["content"] for row in messages]

        evidence = {
            "scope": "strict_customer_chat_to_assigned_agent_context_and_resolution",
            "database": "isolated_sqlite_foreign_keys_enabled",
            "authentication": "real_jwt_and_admin_dependency",
            "demo_mode": False,
            "llm_calls": 0,
            "statuses": statuses,
            "ticket": {"id": ticket_id, "user_id": 1, "assigned_to": 2, "status": "resolved"},
            "transcript": {
                "total": history["total"],
                "roles": [row["role"] for row in messages],
                "citation_count": sum(len(row["citations"]) for row in messages),
                "citation_source_type": messages[1]["citations"][0]["source_type"],
                "faq_id": messages[1]["citations"][0]["faq_id"],
                "citation_roundtrip_matches": True,
                "database_row_ids_match": True,
            },
        }
        (tmp_path / "handoff-context-flow.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        await engine.dispose()
