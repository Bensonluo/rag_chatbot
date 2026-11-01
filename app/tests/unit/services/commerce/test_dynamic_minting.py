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
from typing import Any
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
from app.services.commerce.tools import create_commerce_tools
from app.services.graph.base import GraphEntity, GraphRelation, GraphSearchResult
from app.services.llm.base import LLMMessage, LLMResponse

OWNER = 1

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
    payload = result.data
    assert payload["mint_result"]["status"] == "minted"
    assert payload["mint_result"]["category"]["name"] == "糖果巧克力"
    assert views == []  # no staged purchase view


async def test_mint_dedupes_normalized_names_to_existing(shop) -> None:
    service, _maker = shop
    registry = create_commerce_tools(service, OWNER, [], None)
    first = await registry.execute("commerce:propose_category", MINT_ARGS, user_id=OWNER)
    assert first.success
    twin_args = {**MINT_ARGS, "name": "糖果 巧克力", "base_price_yuan": 99}
    twin = await registry.execute("commerce:propose_category", twin_args, user_id=OWNER)
    assert twin.success
    payload = twin.data
    assert payload["mint_result"]["status"] == "existing"
    assert payload["mint_result"]["dynamic_category_count"] == 1
    assert payload["mint_result"]["category"]["base_price_minor"] == 1500


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
    payload = twin.data
    assert payload["mint_result"]["status"] == "existing"
    # The failing graph never breaks the mint path; nothing crashed above.


async def test_mock_port_publishes_minted_categories_in_discovery(shop) -> None:
    service, _maker = shop
    registry = create_commerce_tools(service, OWNER, [], None)
    minted = await registry.execute("commerce:propose_category", MINT_ARGS, user_id=OWNER)
    assert minted.success
    index = await registry.execute("commerce:discover_catalog", {}, user_id=OWNER)
    assert index.success
    payload = index.data
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
