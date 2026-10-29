"""Semantic product graph projection over the existing Neo4j infrastructure.

The commerce provider remains the only authority for SKU, price, stock and
transaction facts. This projection publishes its authoritative category and
attribute vocabulary as a graph, then asks the LLM to infer high-level product
use-case communities. The graph therefore carries product semantics while the
provider carries every purchasable fact.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field, replace
from typing import Any

from app.models.schemas.commerce import CatalogTaxonomy
from app.services.graph.base import GraphEntity, GraphRelation
from app.services.llm.base import LLMMessage

logger = logging.getLogger(__name__)

_COMMERCE_CATEGORY_TYPE = "CommerceCategory"
_COMMERCE_ATTRIBUTE_TYPE = "CommerceAttribute"
_COMMERCE_USE_CASE_TYPE = "CommerceUseCase"
_ATTRIBUTE_RELATION = "HAS_COMMERCE_ATTRIBUTE"
_USE_CASE_RELATION = "SERVES_COMMERCE_USE_CASE"

_USE_CASE_PROMPT = """You maintain a product knowledge graph from provider-owned category facts.
Infer generic shopping use-case communities from the categories below. Use semantic judgement;
do not emit a phrase alias table.

Return strict JSON:
{{"use_cases":[{{"name":"...","description":"...","category_ids":["..."]}}]}}

Rules:
- 6 to 10 use cases; each links 2 to 8 category_ids that all exist in the input.
- Names describe generic purposes (for example workspace setup, home AV, mobile work), not SKUs.
- Descriptions explain why those categories belong together in one short sentence; mention no price, stock, order state or SKU.
- category_ids must come from the input. Do not include any other key.

Categories:
{categories}"""


@dataclass(frozen=True)
class CommerceGraphFact:
    """One category node plus its connected attribute vocabulary."""

    category_id: str
    name: str
    description: str
    attributes: list[dict[str, Any]]
    matched_use_cases: list[str] = field(default_factory=list)


class CommerceGraphService:
    """Read-only semantic lookup over provider-owned category facts."""

    def __init__(
        self,
        graph_client: Any,
        embedding_service: Any,
        semantic_planner: Any = None,
    ) -> None:
        self._graph = graph_client
        self._embeddings = embedding_service
        self._semantic_planner = semantic_planner

    async def refresh_projection(self, taxonomy: CatalogTaxonomy) -> dict[str, int]:
        """Project provider category facts and LLM use-case communities.

        This is an idempotent semantic index, not a second commerce fact
        source. No user-language alias, SKU, price or stock is written.
        """

        entities: list[GraphEntity] = []
        relations: list[GraphRelation] = []
        category_ids = {category.category_id for category in taxonomy.categories}
        for category in taxonomy.categories:
            category_node_id = f"commerce-category:{category.category_id}"
            attribute_names = [attribute.name for attribute in category.attributes]
            embedding_text = (
                f"{category.name} {category.description} {' '.join(attribute_names)}"
            ).strip()
            entities.append(
                GraphEntity(
                    id=category_node_id,
                    name=f"商品类目：{category.name}",
                    type=_COMMERCE_CATEGORY_TYPE,
                    description=category.description,
                    properties={
                        "category_id": category.category_id,
                        "provider_name": category.name,
                        "attribute_names": attribute_names,
                        "commerce_projection": True,
                    },
                    embedding=(
                        await self._embeddings.embed_single(embedding_text)
                        if self._embeddings is not None
                        else None
                    ),
                )
            )
            for attribute in category.attributes:
                attribute_node_id = f"commerce-attribute:{category.category_id}:{attribute.name}"
                entities.append(
                    GraphEntity(
                        id=attribute_node_id,
                        name=f"{category.name}属性：{attribute.name}",
                        type=_COMMERCE_ATTRIBUTE_TYPE,
                        description=(
                            f"{category.name}的{attribute.name}可取值："
                            f"{'、'.join(attribute.allowed_values)}"
                        ),
                        properties={
                            "category_id": category.category_id,
                            "attribute_name": attribute.name,
                            "allowed_values": attribute.allowed_values,
                            "commerce_projection": True,
                        },
                    )
                )
                relations.append(
                    GraphRelation(
                        id=f"{_ATTRIBUTE_RELATION}:{category.category_id}:{attribute.name}",
                        source_entity_id=category_node_id,
                        target_entity_id=attribute_node_id,
                        relation_type=_ATTRIBUTE_RELATION,
                        description=f"{category.name}拥有商品属性{attribute.name}",
                    )
                )

        use_cases = await self._infer_use_cases(taxonomy)
        for use_case in use_cases:
            linked_ids = [
                category_id
                for category_id in use_case.get("category_ids", [])
                if category_id in category_ids
            ]
            if len(linked_ids) < 2:
                continue
            name = str(use_case.get("name", "")).strip()
            description = str(use_case.get("description", "")).strip()
            if not name or not description:
                continue
            stable = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
            use_case_node_id = f"commerce-use-case:{stable}"
            entities.append(
                GraphEntity(
                    id=use_case_node_id,
                    name=f"商品用途：{name}",
                    type=_COMMERCE_USE_CASE_TYPE,
                    description=description,
                    properties={
                        "use_case_name": name,
                        "commerce_projection": True,
                    },
                    embedding=(
                        await self._embeddings.embed_single(f"{name} {description}")
                        if self._embeddings is not None
                        else None
                    ),
                )
            )
            for category_id in linked_ids:
                relations.append(
                    GraphRelation(
                        id=(
                            f"{_USE_CASE_RELATION}:{stable}:"
                            f"{hashlib.sha256(category_id.encode('utf-8')).hexdigest()[:8]}"
                        ),
                        source_entity_id=f"commerce-category:{category_id}",
                        target_entity_id=use_case_node_id,
                        relation_type=_USE_CASE_RELATION,
                        description=f"该商品类目可服务于{name}",
                    )
                )

        await self._graph.replace_commerce_projection(entities, relations)
        use_case_count = sum(entity.type == _COMMERCE_USE_CASE_TYPE for entity in entities)
        return {
            "categories": len(taxonomy.categories),
            "attributes": max(len(entities) - len(taxonomy.categories) - use_case_count, 0),
            "use_cases": use_case_count,
            "relations": len(relations),
        }

    async def _infer_use_cases(self, taxonomy: CatalogTaxonomy) -> list[dict[str, Any]]:
        """Let the LLM infer generic use-case edges from provider categories."""

        if self._semantic_planner is None:
            return []
        category_payload = [
            {
                "category_id": category.category_id,
                "name": category.name,
                "description": category.description,
                "attributes": [
                    {"name": attribute.name, "allowed_values": attribute.allowed_values}
                    for attribute in category.attributes
                ],
            }
            for category in taxonomy.categories
        ]
        response = await self._semantic_planner.generate(
            messages=[
                LLMMessage(
                    role="system",
                    content="Output only valid compact JSON. Do not use reasoning tags or markdown.",
                ),
                LLMMessage(
                    role="user",
                    content=_USE_CASE_PROMPT.format(
                        categories=json.dumps(category_payload, ensure_ascii=False)
                    ),
                ),
            ],
            max_tokens=4096,
        )
        try:
            content = str(response.content)
            decision = json.loads(content[content.index("{") : content.rindex("}") + 1])
            raw = decision["use_cases"]
            if not isinstance(raw, list):
                raise TypeError("use_cases must be an array")
            return [
                item
                for item in raw
                if isinstance(item, dict) and isinstance(item.get("category_ids"), list)
            ][:10]
        except (AttributeError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            logger.warning(
                "Commerce graph use-case inference returned invalid JSON (length=%s tail=%r)",
                len(str(response.content)),
                str(response.content)[-200:],
            )
            return []

    async def search(self, query: str, top_k: int = 3) -> list[CommerceGraphFact]:
        """Find commerce semantics, then expand provider category vocabulary."""

        if self._embeddings is None:
            return []
        query_embedding = await self._embeddings.embed_single(query)
        hits = await self._graph.search_entities_by_embedding(
            query_embedding,
            top_k=top_k,
            entity_types=[_COMMERCE_USE_CASE_TYPE, _COMMERCE_CATEGORY_TYPE],
        )
        facts: dict[str, CommerceGraphFact] = {}
        for hit in hits:
            entity = hit.entities[0] if hit.entities else None
            if entity is None:
                continue
            if entity.type == _COMMERCE_USE_CASE_TYPE:
                rows = await self._graph.execute_cypher(
                    """
                    MATCH (use_case:Entity {id: $use_case_id})
                          <-[:SERVES_COMMERCE_USE_CASE]-(category:Entity)
                    RETURN use_case.name AS use_case_name,
                           use_case.description AS use_case_description,
                           category.id AS category_id
                    """,
                    {"use_case_id": entity.id},
                )
                for row in rows:
                    fact = await self._fact_for_category(str(row["category_id"]))
                    if fact is None:
                        continue
                    matched = f"{row['use_case_name']}：{row['use_case_description']}"
                    facts[fact.category_id] = self._merge_matched_use_case(fact, matched, facts)
            else:
                fact = await self._fact_for_category(entity.id)
                if fact is not None:
                    facts[fact.category_id] = self._merge_matched_use_case(fact, "", facts)
        return list(facts.values())[:10]

    @staticmethod
    def _merge_matched_use_case(
        fact: CommerceGraphFact, matched: str, existing: dict[str, CommerceGraphFact]
    ) -> CommerceGraphFact:
        previous = existing.get(fact.category_id)
        base = previous or fact
        cases = list(base.matched_use_cases)
        if matched and matched not in cases:
            cases.append(matched)
        return replace(base, matched_use_cases=cases)

    async def _fact_for_category(self, category_node_id: str) -> CommerceGraphFact | None:
        rows = await self._graph.execute_cypher(
            """
            MATCH (category:Entity {id: $category_id})
            OPTIONAL MATCH (category)-[:HAS_COMMERCE_ATTRIBUTE]->(attribute:Entity)
            RETURN category.provider_name AS name,
                   category.category_id AS category_id,
                   category.description AS description,
                   attribute.attribute_name AS attribute_name,
                   attribute.allowed_values AS allowed_values
            """,
            {"category_id": category_node_id},
        )
        if not rows:
            return None
        attributes = [
            {
                "name": row["attribute_name"],
                "allowed_values": row.get("allowed_values") or [],
            }
            for row in rows
            if row.get("attribute_name") is not None
        ]
        first = rows[0]
        return CommerceGraphFact(
            category_id=first.get("category_id") or "",
            name=first.get("name") or "",
            description=first.get("description") or "",
            attributes=attributes,
        )
