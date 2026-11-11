"""Semantic routing degrades to bounded rule routing, selection-only.

Contract (red until the product changes):
- The shopping/support LLM router failing (exception, garbage output, invalid
  route, invalid confidence) no longer fails the whole turn: a bounded keyword
  rule router takes over SELECTION only.
- Ambiguous rule input defaults to the support lane — missing a shopping turn
  is recoverable, steering a visitor into the transaction lane is not.
- Degraded routing never touches purchase state and never bypasses the
  confirmation gates: those live in PurchaseService/tools, untouched here.
- A healthy LLM router verdict still wins over the rules.
"""

from typing import Any
from unittest.mock import Mock

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.database.base import Base
from app.models.database.user import User
from app.services.commerce.conversation import CommerceConversation
from app.services.commerce.mock_port import MockCommercePort
from app.services.commerce.service import PurchaseService
from app.services.llm.base import LLMMessage, LLMResponse


class RouterStub:
    """Configurable router LLM: raise, emit garbage, or answer strict JSON."""

    def __init__(self, *, content: str | None = None, error: Exception | None = None) -> None:
        self.content = content
        self.error = error

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        del messages, kwargs
        if self.error is not None:
            raise self.error
        return LLMResponse(content=self.content or "", model="test")


@pytest.fixture
async def purchases():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with maker() as db:
        db.add(User(id=1, email="routing@unit.invalid", hashed_password="!"))
        await db.commit()
    service = PurchaseService(maker, MockCommercePort(maker))
    try:
        yield service
    finally:
        await engine.dispose()


def conversation(provider: RouterStub) -> CommerceConversation:
    return CommerceConversation(
        purchases=None,  # type: ignore[arg-type] # replaced per-test below
        coordinator=Mock(),
        persister=Mock(),
        provider=provider,
    )


async def _route(provider: RouterStub, purchases, message: str):
    talk = CommerceConversation(
        purchases=purchases, coordinator=Mock(), persister=Mock(), provider=provider
    )
    return await talk.route(1, message)


@pytest.mark.asyncio
async def test_router_exception_degrades_to_rule_routing_shopping(purchases):
    decision = await _route(
        RouterStub(error=RuntimeError("router LLM exploded")), purchases, "我想买降噪耳机"
    )
    assert decision.route == "shopping"
    assert decision.fallback is True
    assert 0.0 <= decision.confidence <= 1.0
    assert "fallback" in decision.reason
    # Selection only: no purchase state was created by routing.
    assert await purchases.active(1) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json at all",
        '{"route":"hacking","confidence":0.9}',
        '{"route":"shopping"}',
        '{"route":"shopping","confidence":"high"}',
        '{"route":"shopping","confidence":true}',
        '{"route":"shopping","confidence":7.5}',
        '{"route":123,"confidence":0.5}',
    ],
)
async def test_invalid_router_output_degrades_instead_of_failing(purchases, content):
    decision = await _route(RouterStub(content=content), purchases, "帮我推荐一款键盘")
    assert decision.route == "shopping"
    assert decision.fallback is True


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["我要退货", "怎么转人工客服", "where is my refund?"])
async def test_support_language_stays_support_under_fallback(purchases, message):
    decision = await _route(RouterStub(error=RuntimeError("down")), purchases, message)
    assert decision.route == "support"
    assert decision.fallback is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "你好",
        "今天天气怎么样",
        "买的东西坏了要退",  # 1 shopping term vs 1 support term: tie -> support
    ],
)
async def test_ambiguous_or_keywordless_language_defaults_to_support(purchases, message):
    decision = await _route(RouterStub(error=RuntimeError("down")), purchases, message)
    assert decision.route == "support"
    assert decision.fallback is True


@pytest.mark.asyncio
async def test_healthy_llm_verdict_wins_over_rules(purchases):
    # No shopping keyword at all: only the semantic LLM reading routes this
    # to shopping. The fallback must not shadow a healthy router.
    decision = await _route(
        RouterStub(content='{"route":"shopping","confidence":0.91,"reason":"gifting"}'),
        purchases,
        "帮朋友的新家凑一套开场装备",
    )
    assert decision.route == "shopping"
    assert decision.fallback is False
    assert decision.reason == "gifting"


@pytest.mark.asyncio
async def test_fallback_event_is_logged(purchases, caplog):
    provider = RouterStub(error=RuntimeError("router LLM exploded"))
    with caplog.at_level("WARNING", logger="app.services.commerce.conversation"):
        decision = await _route(provider, purchases, "来一副耳机")
    assert decision.route == "shopping"
    assert any("rule routing" in record.message for record in caplog.records)
