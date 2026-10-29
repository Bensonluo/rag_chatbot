"""Provider-owned catalog discovery across ports, tools and the Agent loop.

Contract:
- Both provider modes answer one typed ``catalog_categories`` method with the
  same taxonomy schema; the external wire follows the documented contract and
  undecodable payloads stay unknown outcomes, never local guesses.
- ``discover_catalog`` is a read-only Agent tool: structured category facts
  for the model, no staged purchase view, no purchase session created.
- Open-language needs stay LLM-owned: scripted model decisions pick standard
  category names from the discovered taxonomy and recover from empty results.
  No lexical alias table exists in the semantic layer.
"""

from __future__ import annotations

import json
import socket
from collections.abc import AsyncIterator, Iterator
from typing import Any
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.commerce_provider.api import create_app
from app.commerce_provider.settings import CommerceProviderSettings
from app.models.database.base import Base
from app.models.database.commerce import (
    MockCommerceObject,
    MockCommerceReceipt,
    PurchaseSession,
)
from app.models.database.user import User
from app.models.schemas.commerce import CatalogBlock, CatalogTaxonomy, CommerceView
from app.services.agent.service import AgentService
from app.services.commerce.conversation import COMMERCE_PROMPT, describe
from app.services.commerce.external_port import ExternalCommercePort
from app.services.commerce.graph_service import CommerceGraphService
from app.services.commerce.mock_port import CATALOG, MockCommercePort
from app.services.commerce.port import ProviderRejected, ProviderUnknown
from app.services.commerce.service import PurchaseService
from app.services.commerce.tools import create_commerce_tools
from app.services.dialogue.tools import ToolResult
from app.services.graph.base import GraphEntity, GraphRelation, GraphSearchResult
from app.services.llm.base import LLMMessage, LLMResponse

BASE_URL = "http://commerce.internal:8000"
TOKEN = "internal-secret"

TAXONOMY_JSON = {
    "categories": [
        {
            "category_id": "mouse-ergonomic",
            "name": "人体工学鼠标",
            "description": "模拟商品：垂直握持、可调 dpi；用于演示购物流程。",
            "attributes": [
                {"name": "grip", "allowed_values": ["右手型", "左手型", "中型", "大型", "紧凑型"]}
            ],
        },
        {
            "category_id": "keyboard-mechanical",
            "name": "机械键盘",
            "description": "模拟商品：有线连接、可热插拔轴体；用于演示购物流程。",
            "attributes": [
                {"name": "layout", "allowed_values": ["68键", "87键", "98键", "104键", "双模87键"]}
            ],
        },
    ]
}


class ScriptedPlanner:
    """Explicit model decisions; production maps needs with the LLM itself."""

    def __init__(self, script: list[tuple[str, dict[str, Any]] | None]) -> None:
        self.script = list(script)
        self.seen_tools: list[list[str]] = []

    async def generate_with_tools(
        self, messages: list[LLMMessage], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        self.seen_tools.append([tool["function"]["name"] for tool in tools])
        step = self.script.pop(0) if self.script else None
        if step is None:
            return LLMResponse(content="已按标准类目为您找到候选。", model="test")
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
        del query_embedding
        matches = [
            entity
            for entity in self.entities
            if entity.type in (entity_types or [entity.type])
            and (
                entity.type == "CommerceUseCase"
                or "音箱" in str(entity.properties.get("provider_name", ""))
            )
        ]
        matches.sort(key=lambda entity: entity.type != "CommerceUseCase")
        return [
            GraphSearchResult(
                content=entity.description or entity.name,
                entities=[entity],
                relations=[],
                score=0.9,
                source_type="graph_embedding",
            )
            for entity in matches[:top_k]
        ]

    async def execute_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        del query
        parameters = params or {}
        if "use_case_id" in parameters:
            use_case = next(
                entity for entity in self.entities if entity.id == parameters["use_case_id"]
            )
            category_ids = [
                relation.source_entity_id
                for relation in self.relations
                if relation.target_entity_id == use_case.id
            ]
            return [
                {
                    "use_case_name": use_case.properties["use_case_name"],
                    "use_case_description": use_case.description,
                    "category_id": category_id,
                }
                for category_id in category_ids
            ]

        entity = next(entity for entity in self.entities if entity.id == parameters["category_id"])
        attribute_entities = [
            other
            for other in self.entities
            if other.type == "CommerceAttribute"
            and other.properties["category_id"] == entity.properties["category_id"]
        ]
        rows = [
            {
                "name": entity.properties["provider_name"],
                "category_id": entity.properties["category_id"],
                "description": entity.description,
                "attribute_name": attribute.properties["attribute_name"],
                "allowed_values": attribute.properties["allowed_values"],
            }
            for attribute in attribute_entities
        ]
        return rows or [
            {
                "name": entity.properties["provider_name"],
                "category_id": entity.properties["category_id"],
                "description": entity.description,
                "attribute_name": None,
                "allowed_values": None,
            }
        ]


class StaticEmbeddings:
    async def embed_single(self, text: str) -> list[float]:
        del text
        return [1.0]


class ScriptUseCasePlanner:
    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        del messages, kwargs
        return LLMResponse(
            content=json.dumps(
                {
                    "use_cases": [
                        {
                            "name": "家庭影音",
                            "description": "用显示、声音和佩戴设备组成观影与听歌场景。",
                            "category_ids": [
                                "monitor-4k",
                                "speaker-conference",
                                "headphones-anc",
                            ],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            model="test",
        )


@pytest.fixture(autouse=True)
def blocked_sockets(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Catalog discovery tests forbid external sockets")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    yield


@pytest.fixture
async def shop(tmp_path) -> AsyncIterator[tuple[PurchaseService, async_sessionmaker[AsyncSession]]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/discovery.sqlite")
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with maker() as db:
        db.add(User(id=1, email="discovery@unit.invalid", hashed_password="!"))
        await db.commit()
    port = MockCommercePort(maker)
    yield PurchaseService(maker, port), maker
    await engine.dispose()


async def _count(maker: async_sessionmaker[AsyncSession], model: Any) -> int:
    async with maker() as db:
        value = await db.scalar(select(func.count()).select_from(model))
        assert value is not None
        return int(value)


# --- provider modes answer one typed method -------------------------------


async def test_mock_port_publishes_the_family_taxonomy(shop) -> None:
    service, _maker = shop
    taxonomy = await service.discover_catalog(1)
    assert isinstance(taxonomy, CatalogTaxonomy)
    family_ids = {str(item["product_id"]) for item in CATALOG}
    assert {category.category_id for category in taxonomy.categories} == family_ids
    for category in taxonomy.categories:
        assert category.description
        (attribute,) = category.attributes
        assert attribute.allowed_values
        # Typed facts only — no user-language alias list rides the taxonomy.
        assert set(category.model_dump()) == {
            "category_id",
            "name",
            "description",
            "attributes",
        }


async def test_mock_port_rejects_invalid_owner(shop) -> None:
    _service, maker = shop
    port = MockCommercePort(maker)
    with pytest.raises(ProviderRejected):
        await port.catalog_categories(0)


async def test_external_adapter_matches_the_documented_wire() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=TAXONOMY_JSON)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        port = ExternalCommercePort(BASE_URL, token=TOKEN, client=client)
        taxonomy = await port.catalog_categories(7)
    assert seen[0].method == "GET"
    assert seen[0].url.path == "/commerce/categories"
    assert seen[0].url.params["owner_user_id"] == "7"
    assert seen[0].headers["authorization"] == f"Bearer {TOKEN}"
    assert [category.name for category in taxonomy.categories] == ["人体工学鼠标", "机械键盘"]
    assert taxonomy.categories[0].attributes[0].allowed_values[:2] == ["右手型", "左手型"]


async def test_external_adapter_keeps_the_wire_typed() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"categories": []}))
    ) as client:
        port = ExternalCommercePort(BASE_URL, client=client)
        with pytest.raises(ProviderUnknown):
            await port.catalog_categories(7)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text="ok"))
    ) as client:
        port = ExternalCommercePort(BASE_URL, client=client)
        with pytest.raises(ProviderUnknown):
            await port.catalog_categories(7)


def test_external_wire_contract_matches_the_provider_service(tmp_path) -> None:
    """The adapter's documented path is served verbatim by the provider app."""
    with TestClient(
        create_app(
            CommerceProviderSettings(
                database_path=str(tmp_path / "provider.sqlite3"), auth_token=""
            )
        )
    ) as client:
        response = client.get("/commerce/categories", params={"owner_user_id": 1})
        assert response.status_code == 200
        body = response.json()
        assert body["categories"]
        assert CatalogTaxonomy.model_validate(body).categories[0].name


# --- discovery is read-only ------------------------------------------------


async def test_discovery_never_creates_or_seeds_state(shop) -> None:
    service, maker = shop
    taxonomy = await service.discover_catalog(1)
    assert taxonomy.categories
    assert await _count(maker, PurchaseSession) == 0
    assert await _count(maker, MockCommerceReceipt) == 0
    assert await _count(maker, MockCommerceObject) == 0
    assert await service.active(1) is None


async def test_discover_tool_stages_no_purchase_view(shop) -> None:
    service, maker = shop
    views: list[CommerceView] = []
    registry = create_commerce_tools(service, 1, views, None)
    result = await registry.execute("commerce:discover_catalog", {}, user_id=1)
    assert isinstance(result, ToolResult)
    assert result.success
    categories = result.data["catalog_categories"]["categories"]
    assert categories and categories[0]["name"]
    assert views == []
    assert await _count(maker, PurchaseSession) == 0


async def test_discover_tool_is_exported_with_an_empty_schema(shop) -> None:
    service, _maker = shop
    registry = create_commerce_tools(service, 1, [], None)
    exported = {
        schema["function"]["name"]: schema["function"] for schema in registry.to_function_schemas()
    }
    discover = exported["commerce_discover_catalog"]
    assert discover["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    assert "authoritative" in discover["description"] and "Read-only" in discover["description"]


async def test_product_graph_is_semantic_context_not_commerce_facts(shop) -> None:
    """The existing graph holds category semantics; provider facts stay external."""

    service, maker = shop
    taxonomy = await service.discover_catalog(1)
    graph = RecordingGraphClient()
    product_graph = CommerceGraphService(graph, StaticEmbeddings())

    projected = await product_graph.refresh_projection(taxonomy)
    assert projected == {
        "categories": len(taxonomy.categories),
        "attributes": sum(len(category.attributes) for category in taxonomy.categories),
        "use_cases": 0,
        "relations": sum(len(category.attributes) for category in taxonomy.categories),
    }
    category_entities = [entity for entity in graph.entities if entity.type == "CommerceCategory"]
    assert len(category_entities) == len(taxonomy.categories)
    assert all(entity.embedding for entity in category_entities)
    projection_json = json.dumps(
        [entity.properties for entity in graph.entities], ensure_ascii=False, default=str
    )
    assert "alias" not in projection_json.lower()
    assert "sku" not in projection_json.lower()
    assert "price" not in projection_json.lower()

    facts = await product_graph.search("家庭影音", top_k=1)
    assert facts and "音箱" in facts[0].name
    assert facts[0].attributes

    views: list[CommerceView] = []
    registry = create_commerce_tools(service, 1, views, None, commerce_graph=product_graph)
    exported = {
        schema["function"]["name"]: schema["function"] for schema in registry.to_function_schemas()
    }
    assert "commerce_graph_context" in exported
    result = await registry.execute(
        "commerce:graph_context", {"query": "家庭影音", "top_k": 1}, user_id=1
    )
    assert isinstance(result, ToolResult)
    assert result.success
    assert result.data["commerce_graph"]["semantic_only"] is True
    assert "音箱" in result.data["commerce_graph"]["facts"][0]["name"]
    assert views == []
    assert await _count(maker, PurchaseSession) == 0


async def test_llm_projects_product_use_case_communities(shop) -> None:
    """The LLM infers graph use-case edges; provider categories remain nodes."""

    service, _maker = shop
    taxonomy = await service.discover_catalog(1)
    graph = RecordingGraphClient()
    product_graph = CommerceGraphService(
        graph,
        StaticEmbeddings(),
        semantic_planner=ScriptUseCasePlanner(),
    )
    projected = await product_graph.refresh_projection(taxonomy)
    attribute_count = sum(len(category.attributes) for category in taxonomy.categories)
    assert projected == {
        "categories": len(taxonomy.categories),
        "attributes": attribute_count,
        "use_cases": 1,
        "relations": attribute_count + 3,
    }

    facts = await product_graph.search("家庭影音", top_k=1)
    assert {fact.name for fact in facts} == {
        "4K显示器",
        "会议音箱",
        "轻音降噪耳机",
    }
    assert all(fact.matched_use_cases for fact in facts)
    assert all(fact.attributes for fact in facts)


def test_graph_tool_is_absent_without_wired_graph(shop) -> None:
    service, _maker = shop
    registry = create_commerce_tools(service, 1, [], None)
    exported = {schema["function"]["name"] for schema in registry.to_function_schemas()}
    assert "commerce_graph_context" not in exported


# --- the LLM owns open-need mapping ----------------------------------------


async def test_open_need_discovers_then_maps_a_standard_category(shop) -> None:
    service, _maker = shop
    views: list[CommerceView] = []
    registry = create_commerce_tools(service, 1, views, None)
    planner = ScriptedPlanner(
        [
            ("discover_catalog", {}),
            # The model's own semantic pick from the taxonomy — no alias table.
            ("search_products", {"query": "人体工学鼠标"}),
        ]
    )
    agent = AgentService(planner, registry, max_steps=6, system_prompt=COMMERCE_PROMPT)
    result = await agent.run("天天码字手腕越来越疼，想换个舒服点的", user_id=1)
    assert result.executed_tools == ["commerce_discover_catalog", "commerce_search_products"]
    assert all("commerce_discover_catalog" in names for names in planner.seen_tools[:2])
    catalog = next(block for block in views[-1].blocks if isinstance(block, CatalogBlock))
    assert catalog.items
    assert all("人体工学鼠标" in item.name for item in catalog.items)


async def test_zero_result_need_recovers_through_discovery(shop) -> None:
    service, _maker = shop
    views: list[CommerceView] = []
    registry = create_commerce_tools(service, 1, views, None)
    planner = ScriptedPlanner(
        [
            # A novel open phrasing: lexical search finds nothing.
            ("search_products", {"query": "桌面上一堆线乱成一团"}),
            ("discover_catalog", {}),
            # The model remaps the need onto a standard category name.
            ("search_products", {"query": "桌面拓展坞"}),
        ]
    )
    agent = AgentService(planner, registry, max_steps=6, system_prompt=COMMERCE_PROMPT)
    result = await agent.run("桌面上一堆线乱成一团，有办法吗", user_id=1)
    assert result.executed_tools == [
        "commerce_search_products",
        "commerce_discover_catalog",
        "commerce_search_products",
    ]
    first, last = views[0], views[-1]
    assert next(b for b in first.blocks if isinstance(b, CatalogBlock)).items == []
    assert next(b for b in last.blocks if isinstance(b, CatalogBlock)).items


async def test_multi_category_need_keeps_every_searched_category(shop) -> None:
    """An open setup need keeps category diversity within five visible SKUs."""
    service, _maker = shop
    views: list[CommerceView] = []
    registry = create_commerce_tools(service, 1, views, None)
    planner = ScriptedPlanner(
        [
            ("discover_catalog", {}),
            ("search_products", {"query": "4K显示器", "page_size": 5}),
            ("search_products", {"query": "机械键盘", "page_size": 5}),
            ("search_products", {"query": "人体工学鼠标", "page_size": 5}),
        ]
    )
    agent = AgentService(planner, registry, max_steps=6, system_prompt=COMMERCE_PROMPT)
    await agent.run("帮我配一个远程办公角落", user_id=1)
    catalog = next(block for block in views[-1].blocks if isinstance(block, CatalogBlock))
    names = "\n".join(item.name for item in catalog.items)
    assert all(term in names for term in ("4K显示器", "机械键盘", "人体工学鼠标"))
    assert len(catalog.items) == 5
    assert len({item.sku_id for item in catalog.items}) == len(catalog.items)
    assert catalog.total_estimate > len(catalog.items)
    projection = describe(views[-1])
    numbered_products = [
        line
        for line in projection.splitlines()
        if len(line) > 3 and line[0].isdigit() and line[1:3] == ". "
    ]
    assert len(numbered_products) == 5
    assert "本轮最多展示5个候选" in projection


async def test_same_query_pagination_keeps_provider_order(shop) -> None:
    """Paging one category must not interleave page one with page two."""

    service, _maker = shop
    views: list[CommerceView] = []
    registry = create_commerce_tools(service, 1, views, None)
    first_call = await registry.execute(
        "commerce:search_products", {"query": "音箱", "page_size": 3}, user_id=1
    )
    assert isinstance(first_call, ToolResult)
    assert first_call.success
    first = next(block for block in views[-1].blocks if isinstance(block, CatalogBlock))
    assert first.next_page_token is not None

    second_call = await registry.execute(
        "commerce:search_products",
        {"query": "音箱", "page_size": 3, "page_token": first.next_page_token},
        user_id=1,
    )
    assert isinstance(second_call, ToolResult)
    assert second_call.success
    merged = next(block for block in views[-1].blocks if isinstance(block, CatalogBlock))
    assert 3 < len(merged.items) <= 5
    assert [item.sku_id for item in merged.items[: len(first.items)]] == [
        item.sku_id for item in first.items
    ]
    assert merged.next_page_token is None


def test_semantic_layer_has_no_lexical_alias_table() -> None:
    """The taxonomy schema is facts-only; no alias field can carry routing."""
    schema = CatalogTaxonomy.model_json_schema()
    category_fields = set(schema["$defs"]["CatalogCategory"]["properties"])
    assert category_fields == {"category_id", "name", "description", "attributes"}
    attribute_fields = set(schema["$defs"]["CategoryAttribute"]["properties"])
    assert attribute_fields == {"name", "allowed_values"}
