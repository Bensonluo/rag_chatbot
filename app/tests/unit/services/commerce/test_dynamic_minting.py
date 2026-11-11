"""Dynamic category minting across the agent tool chain and wire contract.

Contract:
- The fixed domain frame announced in the system context is pinned to the
  provider taxonomy's domains plus the general fallback (drift fails loudly).
- ``commerce_propose_category`` is a catalog-fact tool: no purchase view is
  staged, a normalized-name twin returns ``existing``, and a minted name may
  always be re-searched even after the zero-result budget is spent.
- A fresh mint fire-and-forgets an incremental graph upsert (semantic layer
  only); failures degrade silently to the next full rebuild.
- The external adapter's mint wire matches the documented provider contract.
"""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast
from uuid import uuid4

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.commerce_provider.catalog import default_taxonomy
from app.models.database.base import Base
from app.models.database.user import User
from app.models.schemas.commerce import CommerceView
from app.services.agent.service import AgentService
from app.services.commerce.conversation import _DOMAIN_FRAME, COMMERCE_PROMPT, domain_frame_note
from app.services.commerce.external_port import ExternalCommercePort
from app.services.commerce.graph_service import CommerceGraphService
from app.services.commerce.mock_port import MockCommercePort
from app.services.commerce.service import PurchaseService
from app.services.commerce.tools import CATALOG_DATA_TAGS, create_commerce_tools
from app.services.graph.base import GraphEntity, GraphRelation, GraphSearchResult
from app.services.llm.base import LLMMessage, LLMResponse

OWNER = 1


def _provider_facts(payload: dict[str, Any]) -> dict[str, Any]:
    """Tool payloads wrap provider facts in a delimited block; parse it out."""
    open_tag, close_tag = CATALOG_DATA_TAGS
    block = payload["catalog_data"]
    assert block.startswith(open_tag) and block.endswith(close_tag)
    return cast(dict[str, Any], json.loads(block[len(open_tag) : -len(close_tag)].strip()))


MINT_ARGS: dict[str, Any] = {
    "name": "糖果巧克力",
    "description": "可可制品甜食，含黑巧、牛奶与夹心口味。",
    "domain": "food",
    "attributes": [{"name": "类型", "values": ["黑巧", "牛奶", "白巧", "夹心"]}],
    "variant_key": "规格",
    "variant_values": ["45g", "100g", "250g", "礼盒"],
    "base_price_yuan": 15,
    "source_query": "我想吃巧克力",
}


class ScriptedPlanner:
    def __init__(self, script: list[tuple[str, dict[str, Any]] | None]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def generate_with_tools(
        self, messages: list[LLMMessage], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        self.calls.append({"tool_names": [tool["function"]["name"] for tool in tools]})
        step = self.script.pop(0) if self.script else None
        if step is None:
            return LLMResponse(content="已为您现场补充该类目并找到候选。", model="test")
        tool, arguments = step
        return LLMResponse(
            content="",
            model="test",
            tool_calls=[
                {
                    "id": str(uuid4()),
                    "name": "commerce_" + tool,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            ],
        )


class RecordingGraphClient:
    def __init__(self) -> None:
        self.entities: list[GraphEntity] = []
        self.relations: list[GraphRelation] = []

    async def add_entities(self, entities: list[GraphEntity]) -> list[str]:
        self.entities.extend(entities)
        return [entity.id for entity in entities]

    async def add_relations(self, relations: list[GraphRelation]) -> list[str]:
        self.relations.extend(relations)
        return [relation.id for relation in relations]

    async def replace_commerce_projection(
        self, entities: list[GraphEntity], relations: list[GraphRelation]
    ) -> dict[str, int]:
        self.entities = list(entities)
        self.relations = list(relations)
        return {"entities": len(entities), "relations": len(relations)}

    async def search_entities_by_embedding(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        entity_types: list[str] | None = None,
    ) -> list[GraphSearchResult]:
        del query_embedding, entity_types
        return [
            GraphSearchResult(
                content=entity.description or entity.name,
                entities=[entity],
                relations=[],
                score=0.9,
                source_type="graph_embedding",
            )
            for entity in self.entities[:top_k]
        ]

    async def execute_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        del query, params
        return []


class FailingGraphClient(RecordingGraphClient):
    async def add_entities(self, entities: list[GraphEntity]) -> list[str]:
        raise RuntimeError("graph unavailable")


class StaticEmbeddings:
    async def embed_single(self, text: str) -> list[float]:
        del text
        return [1.0]


@pytest.fixture(autouse=True)
def blocked_sockets(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Dynamic minting tests forbid external sockets")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    yield


@pytest.fixture
async def shop(tmp_path) -> AsyncIterator[tuple[PurchaseService, async_sessionmaker[AsyncSession]]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/minting.sqlite")
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with maker() as db:
        db.add(User(id=1, email="minting@unit.invalid", hashed_password="!"))
        await db.commit()
    yield PurchaseService(maker, MockCommercePort(maker)), maker
    await engine.dispose()


def test_domain_frame_is_pinned_to_the_provider_taxonomy() -> None:
    taxonomy_domains = {category.domain for category in default_taxonomy().categories}
    frame_ids = {identifier for identifier, _ in _DOMAIN_FRAME}
    assert frame_ids == taxonomy_domains | {"general"}
    assert len(_DOMAIN_FRAME) == len(frame_ids)  # no duplicate ids
    assert "general" in domain_frame_note()


async def test_mint_tool_is_a_catalog_fact_not_a_purchase_view(shop) -> None:
    service, maker = shop
    views: list[CommerceView] = []
    registry = create_commerce_tools(service, OWNER, views, None)
    result = await registry.execute("commerce:propose_category", MINT_ARGS, user_id=OWNER)
    assert result.success, result.message
    payload = _provider_facts(result.data)
    assert payload["status"] == "minted"
    assert payload["category"]["name"] == "糖果巧克力"
    assert views == []  # no staged purchase view


async def test_mint_tool_carries_brand_styles_to_the_provider(shop) -> None:
    """The naming matrix rides the mint wire so the provider's LLM-authored
    shelf replaces the static 恒锐/Astro grid; malformed matrices are
    refused at the tool boundary before any provider call."""

    service, _maker = shop
    registry = create_commerce_tools(service, OWNER, [], None)
    definition = registry.get_tool_by_name("commerce_propose_category")
    assert definition is not None
    # The LLM must be able to see the parameter.
    assert "brand_styles" in definition.parameters_schema["properties"]

    styles = [
        {"brand": "金穗坊", "series": "小榨原香"},
        {"brand": "澄香", "series": "家庭装"},
        {"brand": "鲁源坊", "series": "轻食"},
        {"brand": "金穗坊", "series": "礼享装"},
    ]
    result = await registry.execute(
        "commerce:propose_category", {**MINT_ARGS, "brand_styles": styles}, user_id=OWNER
    )
    assert result.success, result.message
    assert _provider_facts(result.data)["status"] == "minted"

    malformed = await registry.execute(
        "commerce:propose_category",
        {**MINT_ARGS, "brand_styles": [{"brand": "金穗坊"}]},
        user_id=OWNER,
    )
    assert not malformed.success
    assert "brand_styles" in malformed.message


async def test_mint_dedupes_normalized_names_to_existing(shop) -> None:
    service, _maker = shop
    registry = create_commerce_tools(service, OWNER, [], None)
    first = await registry.execute("commerce:propose_category", MINT_ARGS, user_id=OWNER)
    assert first.success
    twin_args = {**MINT_ARGS, "name": "糖果 巧克力", "base_price_yuan": 99}
    twin = await registry.execute("commerce:propose_category", twin_args, user_id=OWNER)
    assert twin.success
    payload = _provider_facts(twin.data)
    assert payload["status"] == "existing"
    assert payload["dynamic_category_count"] == 1
    assert payload["category"]["base_price_minor"] == 1500


async def test_minted_name_research_is_never_blocked_by_zero_result_budget(shop) -> None:
    service, _maker = shop
    registry = create_commerce_tools(service, OWNER, [], None)
    for query in ("保健食品", "母婴食品"):
        result = await registry.execute("commerce:search_products", {"query": query}, user_id=OWNER)
        assert result.success
    blocked = await registry.execute(
        "commerce:search_products", {"query": "任意其它类目"}, user_id=OWNER
    )
    assert not blocked.success  # budget spent for unrelated names

    minted = await registry.execute("commerce:propose_category", MINT_ARGS, user_id=OWNER)
    assert minted.success
    research = await registry.execute(
        "commerce:search_products", {"query": "糖果巧克力"}, user_id=OWNER
    )
    assert research.success


async def test_mint_tool_rejects_bad_shapes_client_side(shop) -> None:
    service, _maker = shop
    registry = create_commerce_tools(service, OWNER, [], None)
    bad_cases: list[dict[str, Any]] = [
        {**MINT_ARGS, "variant_values": ["45g", "100g", "250g"]},
        {**MINT_ARGS, "attributes": []},
        {**MINT_ARGS, "base_price_yuan": 0.5},
        {**MINT_ARGS, "domain": ""},
        {**MINT_ARGS, "name": "  "},
    ]
    for case in bad_cases:
        result = await registry.execute("commerce:propose_category", case, user_id=OWNER)
        assert not result.success, case


async def test_first_time_mint_journey_survives_extra_research_rounds(shop) -> None:
    """Regression for the silent 猪油 fallback (2026-11-01, session 8).

    The real journey ran search → discover → search → search (one wasted
    re-search) → propose → re-search: SEVEN model rounds, while max_steps=6
    truncated the final re-search burst into the apology that read as a
    client timeout. With the configured COMMERCE_AGENT_MAX_STEPS the same
    journey must finish with real prose, and a truncated outcome must
    never masquerade as a timeout again.
    """

    from app.config.settings import settings
    from app.services.agent.service import OUTCOME_STEPS_EXHAUSTED

    service, _maker = shop
    registry = create_commerce_tools(service, OWNER, [], None)
    planner = ScriptedPlanner(
        [
            ("search_products", {"query": "猪油"}),  # miss
            ("discover_catalog", {}),
            ("search_products", {"query": "猪油"}),  # re-search miss
            ("search_products", {"query": "猪油"}),  # wasted re-search
            ("propose_category", {**MINT_ARGS, "name": "猪油", "domain": "food"}),
            ("search_products", {"query": "糖果巧克力"}),  # post-mint research
            None,  # summary prose
        ]
    )
    assert len(planner.script) == 7, "the production journey was seven rounds"
    agent = AgentService(
        planner,
        registry,
        max_steps=settings.COMMERCE_AGENT_MAX_STEPS,
        system_prompt=COMMERCE_PROMPT,
    )
    result = await agent.run("有猪油卖吗", user_id=OWNER, session_id=1)

    assert "超时" not in result.response
    assert result.outcome != OUTCOME_STEPS_EXHAUSTED
    assert result.truncated is False


async def test_agent_discovers_mints_then_researches(shop) -> None:
    service, _maker = shop
    graph_client = RecordingGraphClient()
    graph = CommerceGraphService(graph_client, StaticEmbeddings())
    views: list[CommerceView] = []
    registry = create_commerce_tools(service, OWNER, views, None, commerce_graph=graph)
    planner = ScriptedPlanner(
        [
            ("discover_catalog", {}),
            ("propose_category", MINT_ARGS),
            ("search_products", {"query": "糖果巧克力"}),
        ]
    )
    agent = AgentService(planner, registry, max_steps=6, system_prompt=COMMERCE_PROMPT)
    result = await agent.run("我想吃巧克力", user_id=OWNER, session_id=1)
    assert result.response
    executed = [call["tool"] for call in planner.calls if call.get("tool")]
    del executed
    # The graph upsert is fire-and-forget: let the task loop drain it.
    for _ in range(10):
        await asyncio.sleep(0)
    category_nodes = [
        entity
        for entity in graph_client.entities
        if entity.type == "CommerceCategory"
        and entity.properties.get("category_id") == "mock_dyn_0001"
    ]
    assert category_nodes, "mint should upsert the category node into the graph"
    assert category_nodes[0].embedding is not None


async def test_mint_upsert_emits_background_trace_events(shop) -> None:
    import asyncio as asyncio_mod

    from app.services.observability.trace_events import TraceSink, trace_sink_active

    service, _maker = shop
    graph = CommerceGraphService(RecordingGraphClient(), StaticEmbeddings())
    registry = create_commerce_tools(service, OWNER, [], None, commerce_graph=graph)
    queue: asyncio_mod.Queue[Any] = asyncio_mod.Queue()
    sink = TraceSink(queue)
    with trace_sink_active(sink):
        minted = await registry.execute("commerce:propose_category", MINT_ARGS, user_id=OWNER)
        assert minted.success
        for _ in range(10):
            await asyncio_mod.sleep(0)  # drain the fire-and-forget task

    events = [queue.get_nowait() for _ in range(queue.qsize())]
    upsert_events = [
        event for event in events if event.detail.get("tool") == "commerce_graph_upsert"
    ]
    statuses = [event.status for event in upsert_events]
    assert statuses == ["start", "end"]
    end = upsert_events[-1]
    assert end.ms is not None and end.ms >= 0
    assert end.detail["ok"] is True
    assert end.detail["background"] is True


async def test_respond_bridges_background_traces_into_the_observer_channel(shop) -> None:
    """Regression pin: the guest shopping stream forwards events via
    ``on_trace`` only, so ``emit_trace`` calls from fire-and-forget tasks
    (the post-mint graph upsert) were silent no-ops there — the live
    waterfall never showed the graph write. respond() must bridge the
    ContextVar channel onto the observer it was given."""
    from app.models.database.session import ChatSession
    from app.services.chat.coordination import SessionCoordinator
    from app.services.chat.persistence import ChatMessagePersister
    from app.services.commerce.conversation import CommerceConversation
    from app.services.observability.trace_events import TraceEvent

    service, maker = shop
    async with maker() as db:
        session = ChatSession(id=1, user_id=OWNER, session_metadata={"access_mode": "private"})
        db.add(session)
        await db.commit()
        key = session.session_key
    graph = CommerceGraphService(RecordingGraphClient(), StaticEmbeddings())
    conversation = CommerceConversation(
        service,
        SessionCoordinator(maker),
        ChatMessagePersister(maker),
        provider=ScriptedPlanner([("propose_category", MINT_ARGS)]),
        commerce_graph_service=graph,
    )
    seen: list[TraceEvent] = []

    reply = await conversation.respond(OWNER, 1, key, "我想吃巧克力", on_trace=seen.append)

    assert reply.content
    # The fire-and-forget upsert runs concurrently with the final reply.
    for _ in range(10):
        await asyncio.sleep(0)
    upsert = [e for e in seen if e.detail.get("tool") == "commerce_graph_upsert"]
    assert [e.status for e in upsert] == ["start", "end"]
    assert upsert[-1].detail["ok"] is True
    assert upsert[-1].detail["background"] is True


async def test_status_turns_do_not_re_pitch_the_default_catalog_page(shop) -> None:
    """The candidate product list belongs to exactly one state: the customer
    said what they want, a search matched, and the reply invites selection
    and add-to-cart. An order/logistics question must never carry it — not
    even when an active shopping-stage purchase exists that older code
    refilled with a default catalog page."""
    from app.models.database.session import ChatSession
    from app.services.chat.coordination import SessionCoordinator
    from app.services.chat.persistence import ChatMessagePersister
    from app.services.commerce.conversation import CommerceConversation

    service, maker = shop
    async with maker() as db:
        session = ChatSession(id=2, user_id=OWNER, session_metadata={"access_mode": "private"})
        db.add(session)
        await db.commit()
        key = session.session_key
    planner = ScriptedPlanner(
        [
            ("search_products", {"query": "显示器"}),
            None,  # end turn 1 with prose
            ("purchase_status", {}),
            None,  # end turn 2 with prose
        ]
    )
    conversation = CommerceConversation(
        service,
        SessionCoordinator(maker),
        ChatMessagePersister(maker),
        provider=planner,
    )

    browse = await conversation.respond(OWNER, 2, key, "看看显示器")
    assert "商品事实为准" in browse.content and "¥" in browse.content, (
        "browse turn still publishes the candidate page"
    )

    # Production state at the bug report: an active shopping-stage purchase.
    await service.start(OWNER)
    status = await conversation.respond(OWNER, 2, key, "我要查一下我的订单物流")

    assert "商品事实为准" not in status.content
    assert "¥" not in status.content
    assert "请从上方展示商品" not in status.content
    # UI payload keeps panel continuity even when the reply text stays clean.
    assert status.ui is not None


async def test_existing_twin_does_not_upsert_again_and_failures_degrade(shop) -> None:
    service, _maker = shop
    graph_client = FailingGraphClient()
    graph = CommerceGraphService(graph_client, StaticEmbeddings())
    registry = create_commerce_tools(service, OWNER, [], None, commerce_graph=graph)
    first = await registry.execute("commerce:propose_category", MINT_ARGS, user_id=OWNER)
    assert first.success
    twin = await registry.execute("commerce:propose_category", MINT_ARGS, user_id=OWNER)
    assert twin.success
    for _ in range(10):
        await asyncio.sleep(0)
    assert _provider_facts(twin.data)["status"] == "existing"
    # The failing graph never breaks the mint path; nothing crashed above.


async def test_mock_port_publishes_minted_categories_in_discovery(shop) -> None:
    service, _maker = shop
    registry = create_commerce_tools(service, OWNER, [], None)
    minted = await registry.execute("commerce:propose_category", MINT_ARGS, user_id=OWNER)
    assert minted.success
    index = await registry.execute("commerce:discover_catalog", {}, user_id=OWNER)
    assert index.success
    payload = _provider_facts(index.data)
    names = {item["name"] for item in payload["catalog_category_index"]}
    assert "糖果巧克力" in names


async def test_external_adapter_mint_matches_the_documented_wire() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "status": "minted",
                "category": {
                    "category_id": "dyn_0001",
                    "name": "糖果巧克力",
                    "description": "可可制品甜食",
                    "domain": "food",
                    "attributes": [{"name": "类型", "allowed_values": ["黑巧", "牛奶"]}],
                    "variant_key": "规格",
                    "variant_values": ["45g", "100g", "250g", "礼盒"],
                    "base_price_minor": 1500,
                },
                "dynamic_category_count": 1,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        port = ExternalCommercePort("http://commerce.internal:8000", token="secret", client=client)
        outcome = await port.propose_category(
            OWNER,
            request_key="mint-1",
            name="糖果巧克力",
            description="可可制品甜食",
            domain="food",
            attributes=[("类型", ["黑巧", "牛奶"])],
            variant_key="规格",
            variant_values=["45g", "100g", "250g", "礼盒"],
            base_price_minor=1500,
            source_query="我想吃巧克力",
        )
    assert captured["method"] == "POST"
    assert captured["path"] == "/commerce/categories/dynamic"
    body = captured["body"]
    assert body["request_key"] == "mint-1"
    assert body["attributes"] == [{"name": "类型", "allowed_values": ["黑巧", "牛奶"]}]
    assert body["variant_values"] == ["45g", "100g", "250g", "礼盒"]
    assert outcome.status == "minted"
    assert outcome.category.category_id == "dyn_0001"


async def test_external_adapter_mint_refusal_is_definitive() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422, json={"error": {"code": "rejected", "message": "该商品属于平台不支持的类目范围"}}
        )

    from app.services.commerce.port import ProviderRejected

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        port = ExternalCommercePort("http://commerce.internal:8000", token="secret", client=client)
        with pytest.raises(ProviderRejected, match="不支持"):
            await port.propose_category(
                OWNER,
                request_key="mint-1",
                name="处方药",
                description="违规",
                domain="health",
                attributes=[("规格", ["1盒"])],
                variant_key="颜色",
                variant_values=["红", "蓝", "白", "黑"],
                base_price_minor=1500,
            )


async def test_recovered_tool_refusal_does_not_claim_the_turn_stalled(shop) -> None:
    """A refused-then-retried mint DID advance the turn.

    The mint tool refuses a malformed proposal (provider shape rules);
    the agent corrects and retries successfully. The reply must not be
    prefixed with "本次请求未推进" — the candle turn (dyn_0008) was
    stickered exactly this way while its shelf was being minted.
    """

    from app.models.database.session import ChatSession
    from app.services.chat.coordination import SessionCoordinator
    from app.services.chat.persistence import ChatMessagePersister
    from app.services.commerce.conversation import CommerceConversation

    service, maker = shop
    async with maker() as db:
        db.add(ChatSession(id=3, user_id=OWNER, session_metadata={"access_mode": "private"}))
        await db.commit()
        row = await db.get(ChatSession, 3)
        key = row.session_key if row else ""
    bad_styles = [{"brand": "只有一个"}]
    planner = ScriptedPlanner(
        [
            ("propose_category", {**MINT_ARGS, "brand_styles": bad_styles}),
            ("propose_category", MINT_ARGS),
            ("search_products", {"query": "糖果巧克力"}),
            None,
        ]
    )
    conversation = CommerceConversation(
        service,
        SessionCoordinator(maker),
        ChatMessagePersister(maker),
        provider=planner,
    )
    reply = await conversation.respond(OWNER, 3, key, "我想吃巧克力")
    assert "本次请求未推进" not in reply.content
    assert reply.content.strip(), "the retried mint still produced an answer"


async def test_terminal_tool_failure_still_flags_the_unadvanced_turn(shop) -> None:
    """An intent that ENDS in failure (no successful retry) keeps the
    '未推进' notice: the mock state genuinely did not change."""

    from app.models.database.session import ChatSession
    from app.services.chat.coordination import SessionCoordinator
    from app.services.chat.persistence import ChatMessagePersister
    from app.services.commerce.conversation import CommerceConversation

    service, maker = shop
    async with maker() as db:
        db.add(ChatSession(id=4, user_id=OWNER, session_metadata={"access_mode": "private"}))
        await db.commit()
        row = await db.get(ChatSession, 4)
        key = row.session_key if row else ""
    planner = ScriptedPlanner(
        [
            (
                "update_cart_lines",
                {
                    "lines": [{"sku_id": "SKU-00000001", "quantity": 1}],
                    "purchase_id": "no-such-purchase",
                    "expected_version": 0,
                },
            ),
            None,
        ]
    )
    conversation = CommerceConversation(
        service,
        SessionCoordinator(maker),
        ChatMessagePersister(maker),
        provider=planner,
    )
    reply = await conversation.respond(OWNER, 4, key, "帮我加购这个")
    assert "本次请求未推进" in reply.content
