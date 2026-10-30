"""Regression tests for the dynamic commerce taxonomy architecture."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.v1 import commerce_admin
from app.commerce_provider.catalog import VirtualCatalog, default_taxonomy
from app.core.exceptions import ConflictError
from app.models.database.base import Base
from app.models.database.user import User
from app.models.schemas.commerce import CatalogItem, CatalogPage
from app.models.schemas.commerce_taxonomy import (
    CoverageReviewDecisionRequest,
    RegistryCategory,
    RegistryTaxonomy,
    TaxonomyVersionCreateRequest,
)
from app.services.commerce.coverage import SqlCommerceCoverageRecorder
from app.services.commerce.graph_service import CommerceGraphService
from app.services.commerce.port import CommercePort
from app.services.commerce.router import RouterCommercePort
from app.services.commerce.taxonomy import CommerceTaxonomyRegistry
from app.services.graph.base import GraphEntity, GraphRelation


class FakeGraph:
    def __init__(self) -> None:
        self.entities: list[GraphEntity] = []
        self.relations: list[GraphRelation] = []

    async def replace_commerce_projection(
        self,
        entities: list[GraphEntity],
        relations: list[GraphRelation],
    ) -> dict[str, int]:
        self.entities = entities
        self.relations = relations
        return {"entities": len(entities), "relations": len(relations)}


class FakePort:
    def __init__(self, *, mode: str, items: list[CatalogItem]) -> None:
        self.mode = mode
        self.items = items
        self.queries: list[str] = []

    async def search_products(
        self,
        owner_user_id: int,
        query: str,
        **kwargs: Any,
    ) -> CatalogPage:
        self.queries.append(query)
        return CatalogPage(items=self.items, next_page_token=None, total_estimate=len(self.items))


def _category(
    key: str,
    name: str,
    *,
    policy: str = "generated",
    aliases: tuple[str, ...] = (),
    base_price: int = 10_000,
) -> RegistryCategory:
    return RegistryCategory(
        category_id=key,
        name=name,
        description=f"{name} 类目",
        attributes=[],
        domain="gift",
        provider_policy=policy,
        status="active",
        provider_aliases=list(aliases),
        base_price_minor=base_price,
    )


def _taxonomy(version: str, categories: list[RegistryCategory]) -> RegistryTaxonomy:
    return RegistryTaxonomy(taxonomy_version=version, categories=categories)


@pytest.fixture
async def registry(tmp_path: Any) -> AsyncIterator[CommerceTaxonomyRegistry]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/taxonomy.sqlite")
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield CommerceTaxonomyRegistry(maker)
    await engine.dispose()


def test_registry_category_is_internal_and_public_wire_schema_stays_facts_only() -> None:
    public_schema = default_taxonomy().to_catalog_taxonomy().model_json_schema()
    public_fields = set(public_schema["$defs"]["CatalogCategory"]["properties"])
    assert public_fields == {"category_id", "name", "description", "attributes"}


async def test_taxonomy_registry_versions_and_generated_provider_snapshot(
    registry: CommerceTaxonomyRegistry,
) -> None:
    fallback = await registry.load_active()
    assert fallback.taxonomy_version == "static-v1"

    taxonomy = _taxonomy(
        "gift-v1",
        [
            _category("generated-gift", "生成礼品", aliases=("generated",)),
            _category("external-gift", "外部礼品", policy="external"),
            _category("blocked-domain", "不支持领域", policy="unsupported"),
        ],
    )
    await registry.publish(version_id="gift-v1", taxonomy=taxonomy, activate=True)

    active = await registry.load_active()
    assert {category.category_id for category in active.categories} == {
        "generated-gift",
        "external-gift",
        "blocked-domain",
    }
    generated = await registry.load_active(provider_policy="generated")
    assert [category.category_id for category in generated.categories] == ["generated-gift"]

    catalog = VirtualCatalog(generated)
    public_taxonomy = catalog.taxonomy()
    assert public_taxonomy.model_json_schema()["$defs"]["CatalogCategory"]["properties"].keys() == {
        "category_id",
        "name",
        "description",
        "attributes",
    }
    assert catalog.search(query="生成礼品", page_size=1).items

    with pytest.raises(ConflictError):
        await registry.publish(version_id="gift-v1", taxonomy=taxonomy)


async def test_zero_result_alias_review_projects_semantic_metadata(
    registry: CommerceTaxonomyRegistry,
) -> None:
    await registry.publish(
        version_id="gift-v1",
        taxonomy=_taxonomy("gift-v1", [_category("generated-gift", "礼品")]),
        activate=True,
    )
    event = await registry.record_coverage(query="鲜花礼品", result_count=0, owner_hash="owner-7")
    assert event is not None
    assert event.status == "zero_result"
    assert event.nearest_category_ids == ["generated-gift"]

    review = await registry.create_review(
        event_id=event.id,
        proposed_type="alias",
        proposal_payload={"alias": "鲜花", "category_key": "generated-gift", "confidence": 0.9},
    )
    await registry.decide_review(
        review_id=review.id,
        approve=True,
        reviewer="taxonomy-admin",
        review_note="approved by regression test",
    )

    semantics = await registry.load_semantics()
    assert any(
        node.node_type == "CommerceAlias" and node.name == "鲜花" for node in semantics.nodes
    )
    assert any(
        relation.relation_type == "ALIAS_OF" and relation.target_ref == "category:generated-gift"
        for relation in semantics.relations
    )

    graph = FakeGraph()
    graph_service = CommerceGraphService(graph, embedding_service=None, semantic_planner=None)
    projection = await graph_service.refresh_projection(
        await registry.load_active(),
        semantics,
    )
    assert projection["relations"] >= 1
    assert any(entity.type == "CommerceAlias" for entity in graph.entities)
    assert not any("unit_price_minor" in entity.properties for entity in graph.entities)


async def test_category_review_publishes_new_active_taxonomy_version(
    registry: CommerceTaxonomyRegistry,
) -> None:
    await registry.publish(
        version_id="base-v1",
        taxonomy=_taxonomy("base-v1", [_category("generated-gift", "礼品")]),
        activate=True,
    )
    review = await registry.create_review(
        proposed_type="category",
        proposal_payload={"category": _category("gift-box", "礼品盒").model_dump()},
    )
    await registry.decide_review(
        review_id=review.id,
        approve=True,
        reviewer="taxonomy-admin",
    )

    active = await registry.load_active()
    assert {category.category_id for category in active.categories} == {
        "generated-gift",
        "gift-box",
    }
    assert len(await registry.list_versions()) == 2
    metrics = await registry.coverage_metrics()
    assert metrics["approved_reviews"] == 1


async def test_router_routes_generated_external_and_unsupported_categories(
    registry: CommerceTaxonomyRegistry,
) -> None:
    generated_item = CatalogItem(
        product_id="PRD-0-00000",
        sku_id="SKU-00000000",
        name="生成礼品",
        description="generated",
        attributes={},
        unit_price_minor=1000,
        available_quantity=1,
        max_order_quantity=1,
    )
    external_item = CatalogItem(
        product_id="EXT-1",
        sku_id="EXT-SKU-1",
        name="外部礼品",
        description="external",
        attributes={},
        unit_price_minor=2000,
        available_quantity=1,
        max_order_quantity=1,
    )
    generated_fake = FakePort(mode="generated", items=[generated_item])
    external_fake = FakePort(mode="external", items=[external_item])
    generated = cast(CommercePort, generated_fake)
    external = cast(CommercePort, external_fake)

    await registry.publish(
        version_id="router-v1",
        taxonomy=_taxonomy(
            "router-v1",
            [
                _category("generated-gift", "生成礼品"),
                _category("external-gift", "外部礼品", policy="external"),
                _category("blocked-domain", "不支持领域", policy="unsupported"),
            ],
        ),
        activate=True,
    )
    await registry.add_external_mapping(
        provider="external",
        internal_category_key="external-gift",
        external_category_id="ext-gift-1",
        external_path="Gifts > External",
        confidence=0.98,
    )

    router = RouterCommercePort(
        generated=generated,
        external=external,
        registry=registry,
        external_provider="external",
    )

    external_page = await router.search_products(1, "外部礼品")
    assert [item.sku_id for item in external_page.items] == ["EXT-SKU-1"]
    assert external_fake.queries[-1] == "ext-gift-1"

    unsupported_page = await router.search_products(1, "不支持领域")
    assert unsupported_page.items == []
    assert unsupported_page.total_estimate == 0

    merged_page = await router.search_products(1, "未知礼品")
    assert [item.sku_id for item in merged_page.items] == ["SKU-00000000", "EXT-SKU-1"]


async def test_sql_coverage_recorder_marks_exact_category_matches(
    registry: CommerceTaxonomyRegistry,
) -> None:
    await registry.publish(
        version_id="recorder-v1",
        taxonomy=_taxonomy(
            "recorder-v1",
            [_category("headphone", "耳机", aliases=("headset",))],
        ),
        activate=True,
    )
    recorder = SqlCommerceCoverageRecorder(registry)
    await recorder.record_search(query="耳机", result_count=0, owner=42)

    events = await registry.list_coverage_events(status="zero_result")
    assert len(events) == 1
    assert events[0].matched_category_ids == ["headphone"]
    assert events[0].owner_hash


async def test_admin_taxonomy_endpoint_writes_registry(
    registry: CommerceTaxonomyRegistry,
) -> None:
    admin = User(id=1, email="admin@example.com", hashed_password="x", is_admin=True)
    request = TaxonomyVersionCreateRequest(
        version_id="admin-v1",
        taxonomy=_taxonomy(
            "admin-v1",
            [_category("generated-gift", "礼品", aliases=("generated",))],
        ),
        activate=True,
    )
    response = await commerce_admin.create_taxonomy_version(
        request=request,
        admin=admin,
        registry=registry,
    )
    assert response.version_id == "admin-v1"
    assert response.status == "active"
    assert response.category_count == 1

    exported = await commerce_admin.export_active_taxonomy(admin=admin, registry=registry)
    assert exported.taxonomy_version == "admin-v1"
    assert exported.categories[0].provider_aliases == ["generated"]
    assert exported.categories[0].base_price_minor == 10_000

    reviews = await registry.create_review(
        proposed_type="use_case",
        proposal_payload={
            "name": "送女友礼物",
            "description": "礼物场景",
            "category_keys": ["generated-gift"],
        },
    )
    approved = await commerce_admin.approve_coverage_review(
        review_id=reviews.id,
        request=CoverageReviewDecisionRequest(reviewer="admin@example.com"),
        admin=admin,
        registry=registry,
    )
    assert approved.status == "approved"
    semantics = await registry.load_semantics()
    assert any(node.node_type == "CommerceUseCase" for node in semantics.nodes)
