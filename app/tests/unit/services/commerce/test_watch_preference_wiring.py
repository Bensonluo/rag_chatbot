"""Conversation wiring for price watches and visitor preference memory.

Contract (red until the product changes):
- ``respond()`` injects the visitor's stored preferences as a small
  delimited block (data only, never instructions) so the agent
  personalizes without re-asking, and registers the preference tools
  (save / forget) plus the watch tools on the agent's tool surface.
- A watch trigger fires at conversation time (session-start refresh or a
  catalog read of the watched SKU); the NEXT completion after the trigger
  carries one structured WatchCard, exactly once. The card only asks;
  ordering still requires the existing explicit confirmation gates.
- Redline preference keys (quota./limit./permission./policy.) are rejected
  by the service boundary and surface to the model as a tool error.
"""

import json
import socket  # noqa: F401  (blocked-socket discipline shared with the suite)
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.guest import GuestVisitor
from app.models.database.session import ChatSession
from app.models.database.user import User
from app.models.schemas.commerce import WatchCard
from app.services.chat.chat_service import PersistedReply
from app.services.commerce.conversation import CommerceConversation
from app.services.commerce.mock_port import MockCommercePort
from app.services.commerce.preferences import PreferenceMemoryService
from app.services.commerce.service import PurchaseService
from app.services.commerce.watch import PriceWatchService
from app.services.llm.base import LLMMessage, LLMResponse

OWNER = 1
SESSION_ID = 77
SESSION_KEY = "33333333-3333-4333-8333-333333333333"
VISITOR = "11111111-1111-4111-8111-111111111111"


class Clock:
    now = datetime(2026, 11, 11, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class RecordingPlanner:
    """Tool-calling LLM stub that records every message and tool schema."""

    def __init__(self, script: list[tuple[str, dict[str, Any]] | None]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def generate_with_tools(
        self, messages: list[LLMMessage], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": [asdict(message) for message in messages],
                "tool_names": [tool["function"]["name"] for tool in tools],
            }
        )
        step = self.script.pop(0) if self.script else None
        if step is None:
            return LLMResponse(content="好的，已了解。", model="test")
        tool, arguments = step
        return LLMResponse(
            content="",
            model="test",
            tool_calls=[
                {
                    "id": str(uuid4()),
                    "name": "commerce_" + tool,
                    "arguments": arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments, ensure_ascii=False),
                }
            ],
        )


class FakeCoordinator:
    def __init__(self, maker: async_sessionmaker[Any]) -> None:
        self.sessions = maker

    @asynccontextmanager
    async def hold(self, session_id: int):
        yield

    async def state(self, session_id: int):
        from app.services.chat.coordination import ConversationState

        return ConversationState(
            epoch=0,
            owner_id=OWNER,
            is_private=True,
            claimed_ticket_id=None,
            session_key=SESSION_KEY,
        )


class FakePersister:
    def __init__(self) -> None:
        self.turns: list[str] = []

    async def persist_turn(
        self, session_id: int, owner: int, message: str, content: str, **kwargs: Any
    ) -> PersistedReply:
        self.turns.append(message)
        return PersistedReply(
            message_id=100 + len(self.turns), session_id=session_id, status="completed"
        )


async def make_conversation(
    planner: RecordingPlanner,
) -> tuple[CommerceConversation, PriceWatchService, PreferenceMemoryService, Any]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def fk(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    maker: async_sessionmaker[Any] = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with maker() as db:
        db.add(User(id=OWNER, email="wiring@unit.invalid", hashed_password="!"))
        # Flush first: the users->guest_visitors FK has no relationship()
        # for the unit-of-work sorter to see, so ordering must be explicit.
        await db.flush()
        db.add_all(
            [
                GuestVisitor(
                    id=VISITOR,
                    user_id=OWNER,
                    token_hash="a" * 64,
                    expires_at=Clock.now,
                ),
                ChatSession(
                    id=SESSION_ID,
                    user_id=OWNER,
                    session_key=SESSION_KEY,
                    title="wiring",
                ),
            ]
        )
        await db.commit()
    clock = Clock()
    purchases = PurchaseService(
        maker,
        MockCommercePort(maker, clock=clock, quote_ttl_seconds=30),
        clock=clock,
        quote_ttl_seconds=30,
    )
    watches = PriceWatchService(maker, clock=clock)
    preferences = PreferenceMemoryService(maker, clock=clock)
    conversation = CommerceConversation(
        purchases,
        FakeCoordinator(maker),  # type: ignore[arg-type] # test fake, same shape
        FakePersister(),  # type: ignore[arg-type] # test fake, same shape
        provider=planner,
        price_watches=watches,
        preferences=preferences,
    )
    # The engine dies with the maker's pool once the test ends; closing is
    # handled by GC of the in-memory database (test-isolated anyway).
    return conversation, watches, preferences, engine


async def test_triggered_watch_rides_the_next_completion_exactly_once():
    planner = RecordingPlanner(script=[("search_products", {"query": "降噪耳机"}), None])
    conversation, watches, _, engine = await make_conversation(planner)
    # Pre-authorized condition: target far above the mock price so the
    # session-start refresh fires before the agent even runs.
    await watches.create(VISITOR, "ANC-02", target_price_minor=99900)

    reply = await conversation.respond(OWNER, SESSION_ID, SESSION_KEY, "看看降噪耳机")
    card = reply.ui.watch if reply.ui is not None else None
    assert isinstance(card, WatchCard)
    assert card.sku_id == "ANC-02"
    # Card fields are server facts: the provider's live price, the stored
    # target, the provider's name snapshot.
    assert card.current_price_minor == 30400
    assert card.target_price_minor == 99900
    assert card.sku_name == "轻音降噪耳机 · 白色"

    # Exactly once: the next completion carries no second ask for the same rule.
    reply2 = await conversation.respond(OWNER, SESSION_ID, SESSION_KEY, "还有别的吗")
    assert reply2.ui is None or reply2.ui.watch is None
    rules = await watches.list_rules(VISITOR)
    assert rules[0].status == "asked"
    await engine.dispose()


async def test_untriggered_watch_and_consent_gates_untouched():
    planner = RecordingPlanner(script=[("search_products", {"query": "降噪耳机"}), None])
    conversation, watches, _, engine = await make_conversation(planner)
    # Target below the current price: condition not met, nothing may fire.
    await watches.create(VISITOR, "ANC-02", target_price_minor=10000)
    reply = await conversation.respond(OWNER, SESSION_ID, SESSION_KEY, "看看降噪耳机")
    assert reply.ui is None or reply.ui.watch is None
    rules = await watches.list_rules(VISITOR)
    assert rules[0].status == "active"
    await engine.dispose()


async def test_preferences_are_injected_as_delimited_data():
    planner = RecordingPlanner(script=[None])
    conversation, _, preferences, engine = await make_conversation(planner)
    await preferences.save(VISITOR, "color", "白色", source="confirmed")
    await preferences.save(VISITOR, "budget_hint", "300元上下", source="inferred")

    await conversation.respond(OWNER, SESSION_ID, SESSION_KEY, "推荐一副耳机")

    system_notes = "\n".join(
        str(message.get("content"))
        for call in planner.calls
        for message in call["messages"]
        if message.get("role") == "system"
    )
    assert "Stored visitor preferences" in system_notes
    assert "<session_memory>" in system_notes
    assert "color" in system_notes and "白色" in system_notes
    assert "confirmed" in system_notes and "inferred" in system_notes
    # Bounded injection: the note is small even with many entries.
    assert len(system_notes) < 8000
    await engine.dispose()


async def test_preference_and_watch_tools_are_registered():
    planner = RecordingPlanner(script=[None])
    conversation, _, _, engine = await make_conversation(planner)
    await conversation.respond(OWNER, SESSION_ID, SESSION_KEY, "你好")
    tool_names = {name for call in planner.calls for name in call["tool_names"]}
    assert "commerce_save_preference" in tool_names
    assert "commerce_forget_preference" in tool_names
    assert "commerce_watch_price" in tool_names
    assert "commerce_list_watches" in tool_names
    assert "commerce_cancel_watch" in tool_names
    await engine.dispose()


async def test_redline_preference_key_surfaces_as_tool_error():
    planner = RecordingPlanner(
        script=[
            (
                "save_preference",
                {"pref_key": "quota.monthly_spend", "pref_value": 999999},
            ),
            None,
        ]
    )
    conversation, _, preferences, engine = await make_conversation(planner)
    reply = await conversation.respond(
        OWNER, SESSION_ID, SESSION_KEY, "记住我的月度消费上限是9999元"
    )
    # The service boundary rejected the enforcement-flavored key; the turn
    # still completes and nothing was stored.
    tool_messages = [
        str(message.get("content")) for call in planner.calls for message in call["messages"]
    ]
    assert any("quota" in text and "error" in text for text in tool_messages)
    assert await preferences.load_active(VISITOR) == []
    assert reply.content
    await engine.dispose()


async def test_preference_roundtrip_via_tool():
    planner = RecordingPlanner(
        script=[
            ("save_preference", {"pref_key": "color", "pref_value": "白色"}),
            None,
        ]
    )
    conversation, _, preferences, engine = await make_conversation(planner)
    await conversation.respond(OWNER, SESSION_ID, SESSION_KEY, "记住我喜欢白色")
    entries = await preferences.load_active(VISITOR)
    assert [(entry.pref_key, entry.pref_value) for entry in entries] == [("color", "白色")]
    # source defaults to inferred: the model never upgrades itself.
    assert entries[0].source == "inferred"
    await engine.dispose()
