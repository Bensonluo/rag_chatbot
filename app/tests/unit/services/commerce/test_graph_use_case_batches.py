"""Use-case inference batching and outage degradation (incident 2026-11-01).

A 151-category single-shot inference outlived the 120s GLM client timeout on
the flash tier (three dead providers + one timeout ladder burned ~7 minutes
and voided the whole projection). These tests pin the two guarantees:

1. Categories are inferred in <=40-category parallel batches.
2. A dead planner chain degrades to zero use-case edges while categories,
   attributes and relations still project.
"""

import json
from typing import Any

from app.core.exceptions import ExternalServiceError
from app.models.schemas.commerce import (
    CatalogCategory,
    CatalogTaxonomy,
    CategoryAttribute,
)
from app.services.graph.base import GraphEntity, GraphRelation
from app.services.llm.base import LLMMessage, LLMResponse


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


class StaticEmbeddings:
    async def embed_single(self, text: str) -> list[float]:
        del text
        return [1.0]


class CountingPlanner:
    """Records per-call payloads; returns one use case per two batch ids."""

    def __init__(self, *, per_call: int = 1, fail: bool = False) -> None:
        self.per_call = per_call
        self.fail = fail
        self.payloads: list[list[dict[str, Any]]] = []
        self.prompts: list[str] = []

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        del kwargs
        if self.fail:
            raise ExternalServiceError(
                service="GLM", message="timed out after 120s", status_code=504
            )
        user_content = messages[-1].content
        self.prompts.append(user_content)
        marker = "Categories:\n"
        payload = json.loads(user_content[user_content.rindex(marker) + len(marker) :])
        self.payloads.append(payload)
        ids = [item["category_id"] for item in payload]
        use_cases = [
            {
                "name": f"用途{i}-{len(self.payloads)}",
                "description": "把两个类目组合进同一场景。",
                "category_ids": ids[:2],
            }
            for i in range(self.per_call)
        ]
        return LLMResponse(
            content=json.dumps({"use_cases": use_cases}, ensure_ascii=False),
            model="fake-planner",
        )


def _taxonomy(count: int) -> CatalogTaxonomy:
    return CatalogTaxonomy(
        categories=[
            CatalogCategory(
                category_id=f"cat-{i:03d}",
                name=f"类目{i}",
                description=f"类目{i}的描述",
                attributes=[CategoryAttribute(name="规格", allowed_values=["标准"])],
            )
            for i in range(count)
        ]
    )


async def test_large_taxonomy_infers_in_parallel_batches() -> None:
    """85 categories split into 3 batches of <=40; results are merged."""
    from app.services.commerce.graph_service import (
        _USE_CASE_BATCH_SIZE,
        CommerceGraphService,
    )

    planner = CountingPlanner()
    graph = RecordingGraphClient()
    service = CommerceGraphService(graph, StaticEmbeddings(), semantic_planner=planner)

    projected = await service.refresh_projection(_taxonomy(85))

    assert [len(payload) for payload in planner.payloads] == [40, 40, 5]
    assert all(len(payload) <= _USE_CASE_BATCH_SIZE for payload in planner.payloads)
    # Multi-batch runs ask for compact per-batch communities.
    assert all("4 to 10" in prompt for prompt in planner.prompts)
    assert projected["categories"] == 85
    assert projected["attributes"] == 85
    assert projected["use_cases"] == 3
    # 85 attribute edges plus 2 linked-category edges per use case.
    assert projected["relations"] == 85 + 3 * 2


async def test_single_batch_keeps_whole_catalog_count_hint() -> None:
    from app.services.commerce.graph_service import CommerceGraphService

    planner = CountingPlanner()
    service = CommerceGraphService(
        RecordingGraphClient(), StaticEmbeddings(), semantic_planner=planner
    )

    projected = await service.refresh_projection(_taxonomy(25))

    assert len(planner.payloads) == 1
    assert "20 to 40" in planner.prompts[0]
    assert projected["use_cases"] == 1


async def test_merged_use_cases_capped_at_40() -> None:
    from app.services.commerce.graph_service import CommerceGraphService

    planner = CountingPlanner(per_call=15)  # 3 batches * 15 = 45 -> capped
    service = CommerceGraphService(
        RecordingGraphClient(), StaticEmbeddings(), semantic_planner=planner
    )

    projected = await service.refresh_projection(_taxonomy(100))

    assert projected["use_cases"] == 40


async def test_planner_outage_projects_categories_without_use_cases() -> None:
    """A dead LLM chain must not void the category/attribute projection."""
    from app.services.commerce.graph_service import CommerceGraphService

    planner = CountingPlanner(fail=True)
    graph = RecordingGraphClient()
    service = CommerceGraphService(graph, StaticEmbeddings(), semantic_planner=planner)

    projected = await service.refresh_projection(_taxonomy(151))

    assert projected == {
        "categories": 151,
        "attributes": 151,
        "use_cases": 0,
        "relations": 151,
    }
    assert len(graph.entities) == 151 + 151  # categories + attributes
