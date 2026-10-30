"""Versioned commerce taxonomy and coverage registry service.

The registry owns category lifecycle and semantic-review metadata. It does not
own SKU facts: product search, price, stock, carts, quotes, orders and payments
remain behind ``CommercePort`` implementations.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.commerce_provider.catalog import default_taxonomy
from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.models.database.commerce_taxonomy import (
    CommerceCoverageEvent,
    CommerceCoverageReview,
    CommerceExternalCategoryMapping,
    CommerceSemanticNode,
    CommerceSemanticRelation,
    CommerceTaxonomyAttribute,
    CommerceTaxonomyCategory,
    CommerceTaxonomyVersion,
)
from app.models.schemas.commerce_taxonomy import RegistryCategory, RegistryTaxonomy

_NORMALIZE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class SemanticGraphData:
    """Approved semantic nodes and relations projected into Neo4j."""

    nodes: list[CommerceSemanticNode]
    relations: list[CommerceSemanticRelation]


class CommerceTaxonomyRegistry:
    """Database-backed registry with a static default for fresh installs."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        fallback_taxonomy: RegistryTaxonomy | None = None,
    ) -> None:
        self._sessions = sessions
        self._fallback = fallback_taxonomy or default_taxonomy()

    # ------------------------------------------------------------------
    # Taxonomy versions
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_taxonomy(taxonomy: RegistryTaxonomy) -> None:
        seen: set[str] = set()
        for category in taxonomy.categories:
            key = category.category_id.strip()
            if not key:
                raise ValidationError("Category ids must be non-empty")
            if key in seen:
                raise ValidationError(f"Duplicate category id: {key}")
            seen.add(key)
            attribute_names = [attribute.name.strip() for attribute in category.attributes]
            if any(not name for name in attribute_names):
                raise ValidationError("Attribute names must be non-empty")
            if len(attribute_names) != len(set(attribute_names)):
                raise ValidationError(f"Duplicate attribute names in category: {key}")
        for category in taxonomy.categories:
            if category.parent_id is not None and category.parent_id not in seen:
                raise ValidationError(
                    f"Category {category.category_id} has unknown parent {category.parent_id}"
                )

    async def load_active(self, *, provider_policy: str | None = None) -> RegistryTaxonomy:
        """Return the active taxonomy, or the built-in default when none exists."""

        async with self._sessions() as session:
            version = await session.scalar(
                select(CommerceTaxonomyVersion).where(CommerceTaxonomyVersion.status == "active")
            )
            if version is None:
                return self._fallback
            categories = (
                await session.scalars(
                    select(CommerceTaxonomyCategory)
                    .where(CommerceTaxonomyCategory.version_id == version.id)
                    .order_by(CommerceTaxonomyCategory.position, CommerceTaxonomyCategory.id)
                )
            ).all()
            if not categories:
                return self._fallback
            rows = list(categories)
            attributes = (
                await session.scalars(
                    select(CommerceTaxonomyAttribute).where(
                        CommerceTaxonomyAttribute.version_id == version.id
                    )
                )
            ).all()
            by_category: dict[str, list[Any]] = {row.category_key: [] for row in rows}
            for attribute in attributes:
                if attribute.category_key in by_category:
                    by_category[attribute.category_key].append(attribute)
            for values in by_category.values():
                values.sort(key=lambda item: (item.position, item.id))

            category_models = [
                RegistryCategory(
                    category_id=row.category_key,
                    name=row.name,
                    description=row.description,
                    attributes=[
                        {
                            "name": attribute.name,
                            "allowed_values": list(attribute.allowed_values),
                        }
                        for attribute in by_category[row.category_key]
                    ],
                    parent_id=row.parent_key,
                    domain=row.domain,
                    provider_policy=row.provider_policy,
                    status=row.status,
                    provider_aliases=list(row.provider_aliases),
                    base_price_minor=row.base_price_minor,
                )
                for row in rows
                if row.status != "deprecated"
            ]
        if provider_policy is not None:
            category_models = [
                category
                for category in category_models
                if category.provider_policy == provider_policy and category.status == "active"
            ]
        if not category_models:
            raise NotFoundError("No matching active taxonomy categories")
        return RegistryTaxonomy(
            taxonomy_version=version.id
            if provider_policy is None
            else f"{version.id}:{provider_policy}",
            categories=category_models,
        )

    async def publish(
        self,
        *,
        version_id: str,
        taxonomy: RegistryTaxonomy,
        activate: bool = False,
    ) -> None:
        """Create an immutable taxonomy version and optionally activate it."""

        version_id = version_id.strip()
        if not version_id:
            raise ValidationError("Taxonomy version id is required")
        self._validate_taxonomy(taxonomy)
        async with self._sessions.begin() as session:
            exists = await session.scalar(
                select(CommerceTaxonomyVersion.id).where(CommerceTaxonomyVersion.id == version_id)
            )
            if exists is not None:
                raise ConflictError("Taxonomy version already exists")
            if activate:
                await session.execute(
                    update(CommerceTaxonomyVersion)
                    .where(CommerceTaxonomyVersion.status == "active")
                    .values(status="archived")
                )
            session.add(
                CommerceTaxonomyVersion(
                    id=version_id,
                    status="active" if activate else "draft",
                    activated_at=datetime.now(UTC) if activate else None,
                )
            )
            for position, category in enumerate(taxonomy.categories):
                session.add(
                    CommerceTaxonomyCategory(
                        version_id=version_id,
                        category_key=category.category_id,
                        parent_key=category.parent_id,
                        name=category.name,
                        description=category.description,
                        domain=category.domain,
                        provider_policy=category.provider_policy,
                        status=category.status,
                        provider_aliases=list(category.provider_aliases),
                        base_price_minor=category.base_price_minor,
                        position=position,
                    )
                )
                for attribute_position, attribute in enumerate(category.attributes):
                    session.add(
                        CommerceTaxonomyAttribute(
                            version_id=version_id,
                            category_key=category.category_id,
                            name=attribute.name,
                            allowed_values=list(attribute.allowed_values),
                            position=attribute_position,
                        )
                    )

    async def activate(self, *, version_id: str) -> None:
        async with self._sessions.begin() as session:
            version = await session.get(CommerceTaxonomyVersion, version_id)
            if version is None:
                raise NotFoundError("Taxonomy version")
            await session.execute(
                update(CommerceTaxonomyVersion)
                .where(CommerceTaxonomyVersion.status == "active")
                .values(status="archived")
            )
            version.status = "active"
            version.activated_at = datetime.now(UTC)

    async def list_versions(self) -> list[CommerceTaxonomyVersion]:
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(CommerceTaxonomyVersion).order_by(
                        CommerceTaxonomyVersion.activated_at.desc(),
                        CommerceTaxonomyVersion.created_at.desc(),
                    )
                )
            ).all()
            return list(rows)

    async def version_summary(self, version_id: str) -> dict[str, Any]:
        async with self._sessions() as session:
            version = await session.get(CommerceTaxonomyVersion, version_id)
            if version is None:
                raise NotFoundError("Taxonomy version")
            category_count = (
                await session.scalar(
                    select(func.count(CommerceTaxonomyCategory.id)).where(
                        CommerceTaxonomyCategory.version_id == version_id
                    )
                )
                or 0
            )
            attribute_count = (
                await session.scalar(
                    select(func.count(CommerceTaxonomyAttribute.id)).where(
                        CommerceTaxonomyAttribute.version_id == version_id
                    )
                )
                or 0
            )
        return {
            "version_id": version.id,
            "status": version.status,
            "category_count": category_count,
            "attribute_count": attribute_count,
            "activated_at": version.activated_at,
        }

    # ------------------------------------------------------------------
    # External mappings
    # ------------------------------------------------------------------
    async def add_external_mapping(
        self,
        *,
        provider: str,
        internal_category_key: str,
        external_category_id: str,
        external_path: str = "",
        confidence: float = 0.0,
        status: str = "active",
    ) -> None:
        taxonomy = await self.load_active()
        active_keys = {category.category_id for category in taxonomy.categories}
        if internal_category_key not in active_keys:
            raise NotFoundError("Internal taxonomy category")
        version_id = taxonomy.taxonomy_version or "fallback"
        version_id = version_id.split(":", 1)[0]
        async with self._sessions.begin() as session:
            session.add(
                CommerceExternalCategoryMapping(
                    version_id=version_id,
                    internal_category_key=internal_category_key,
                    provider=provider,
                    external_category_id=external_category_id,
                    external_path=external_path,
                    confidence=confidence,
                    status=status,
                )
            )

    async def find_external_mapping(
        self,
        *,
        provider: str,
        internal_category_key: str,
    ) -> CommerceExternalCategoryMapping | None:
        taxonomy = await self.load_active()
        version_id = (taxonomy.taxonomy_version or "fallback").split(":", 1)[0]
        async with self._sessions() as session:
            return await session.scalar(
                select(CommerceExternalCategoryMapping)
                .where(
                    CommerceExternalCategoryMapping.version_id == version_id,
                    CommerceExternalCategoryMapping.provider == provider,
                    CommerceExternalCategoryMapping.internal_category_key == internal_category_key,
                    CommerceExternalCategoryMapping.status == "active",
                )
                .order_by(CommerceExternalCategoryMapping.confidence.desc())
            )

    # ------------------------------------------------------------------
    # Semantic graph metadata
    # ------------------------------------------------------------------
    async def load_semantics(self) -> SemanticGraphData:
        async with self._sessions() as session:
            nodes = (
                await session.scalars(
                    select(CommerceSemanticNode).where(CommerceSemanticNode.status == "approved")
                )
            ).all()
            relations = (
                await session.scalars(
                    select(CommerceSemanticRelation).where(
                        CommerceSemanticRelation.status == "approved"
                    )
                )
            ).all()
            return SemanticGraphData(nodes=list(nodes), relations=list(relations))

    @staticmethod
    def _semantic_id(prefix: str, value: str) -> str:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}:{digest}"

    async def _upsert_semantic_node(
        self,
        session: AsyncSession,
        *,
        node_type: str,
        name: str,
        description: str,
        source: str,
        confidence: float,
        properties: dict[str, Any],
    ) -> CommerceSemanticNode:
        node_id = self._semantic_id(node_type, name)
        node = await session.get(CommerceSemanticNode, node_id)
        if node is None:
            node = CommerceSemanticNode(
                id=node_id,
                node_type=node_type,
                name=name,
                description=description,
                properties=properties,
                source=source,
                confidence=confidence,
                status="approved",
            )
            session.add(node)
        else:
            node.description = description or node.description
            node.properties = {**node.properties, **properties}
            node.source = source
            node.confidence = confidence
            node.status = "approved"
        return node

    async def _upsert_semantic_relation(
        self,
        session: AsyncSession,
        *,
        source_ref: str,
        target_ref: str,
        relation_type: str,
        description: str,
        source: str,
        confidence: float,
        properties: dict[str, Any],
    ) -> CommerceSemanticRelation:
        relation_id = self._semantic_id(
            relation_type,
            f"{source_ref}->{target_ref}:{description}",
        )
        relation = await session.get(CommerceSemanticRelation, relation_id)
        if relation is None:
            relation = CommerceSemanticRelation(
                id=relation_id,
                source_ref=source_ref,
                target_ref=target_ref,
                relation_type=relation_type,
                description=description,
                properties=properties,
                source=source,
                confidence=confidence,
                status="approved",
            )
            session.add(relation)
        else:
            relation.description = description or relation.description
            relation.properties = {**relation.properties, **properties}
            relation.source = source
            relation.confidence = confidence
            relation.status = "approved"
        return relation

    # ------------------------------------------------------------------
    # Coverage capture and review
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_query(query: str) -> str:
        return _NORMALIZE_PATTERN.sub(" ", query.strip().lower())[:500]

    async def record_coverage(
        self,
        *,
        query: str,
        result_count: int,
        matched_category_ids: list[str] | None = None,
        owner_hash: str = "",
    ) -> CommerceCoverageEvent | None:
        """Record zero/low-result shopping searches for taxonomy review."""

        status = "zero_result" if result_count == 0 else "low_result" if result_count < 3 else ""
        if not status:
            return None
        normalized = self._normalize_query(query)
        try:
            taxonomy = await self.load_active()
            names = {category.name: category.category_id for category in taxonomy.categories}
            nearest_names = difflib.get_close_matches(query.strip(), list(names), n=3, cutoff=0.35)
            nearest = [names[name] for name in nearest_names]
        except NotFoundError:
            nearest = []
        event = CommerceCoverageEvent(
            query=query[:500],
            normalized_query=normalized,
            owner_hash=owner_hash[:64],
            matched_category_ids=matched_category_ids or [],
            nearest_category_ids=nearest,
            result_count=result_count,
            status=status,
        )
        async with self._sessions.begin() as session:
            session.add(event)
            await session.flush()
            return CommerceCoverageEvent(
                id=event.id,
                query=event.query,
                normalized_query=event.normalized_query,
                owner_hash=event.owner_hash,
                matched_category_ids=list(event.matched_category_ids),
                nearest_category_ids=list(event.nearest_category_ids),
                result_count=event.result_count,
                status=event.status,
                created_at=event.created_at,
            )

    async def list_coverage_events(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
    ) -> list[CommerceCoverageEvent]:
        async with self._sessions() as session:
            statement = select(CommerceCoverageEvent).order_by(
                CommerceCoverageEvent.created_at.desc()
            )
            if status is not None:
                statement = statement.where(CommerceCoverageEvent.status == status)
            rows = (await session.scalars(statement.limit(limit))).all()
            return list(rows)

    async def create_review(
        self,
        *,
        proposed_type: str,
        proposal_payload: dict[str, Any],
        event_id: int | None = None,
    ) -> CommerceCoverageReview:
        if proposed_type not in {"category", "alias", "use_case", "relation"}:
            raise ValidationError("Unsupported coverage proposal type")
        async with self._sessions.begin() as session:
            review = CommerceCoverageReview(
                event_id=event_id,
                proposed_type=proposed_type,
                proposal_payload=proposal_payload,
                status="pending",
            )
            session.add(review)
            await session.flush()
            return CommerceCoverageReview(
                id=review.id,
                event_id=review.event_id,
                proposed_type=review.proposed_type,
                proposal_payload=dict(review.proposal_payload),
                status=review.status,
                reviewer=review.reviewer,
                review_note=review.review_note,
                created_at=review.created_at,
                updated_at=review.updated_at,
            )

    async def _get_review(self, review_id: int) -> CommerceCoverageReview:
        async with self._sessions() as session:
            review = await session.get(CommerceCoverageReview, review_id)
            if review is None:
                raise NotFoundError("Coverage review")
            return CommerceCoverageReview(
                id=review.id,
                event_id=review.event_id,
                proposed_type=review.proposed_type,
                proposal_payload=dict(review.proposal_payload),
                status=review.status,
                reviewer=review.reviewer,
                review_note=review.review_note,
                created_at=review.created_at,
                updated_at=review.updated_at,
            )

    async def _publish_category_proposal(self, payload: dict[str, Any]) -> None:
        raw_category = payload.get("category")
        if not isinstance(raw_category, dict):
            raise ValidationError("Category proposals require a category object")
        category = RegistryCategory.model_validate(raw_category)
        active = await self.load_active()
        keys = {item.category_id for item in active.categories}
        if category.category_id in keys:
            raise ConflictError("Category already exists in active taxonomy")
        next_version = f"coverage-{uuid4().hex[:12]}"
        taxonomy = RegistryTaxonomy(
            taxonomy_version=next_version,
            categories=[*active.categories, category],
        )
        await self.publish(version_id=next_version, taxonomy=taxonomy, activate=True)

    async def decide_review(
        self,
        *,
        review_id: int,
        approve: bool,
        reviewer: str,
        review_note: str = "",
    ) -> None:
        reviewer = reviewer.strip()
        if not reviewer:
            raise ValidationError("Reviewer is required")
        review = await self._get_review(review_id)
        if approve and review.proposed_type == "category":
            await self._publish_category_proposal(review.proposal_payload)
            async with self._sessions.begin() as session:
                stored = await session.get(CommerceCoverageReview, review_id)
                if stored is None:
                    raise NotFoundError("Coverage review")
                stored.status = "approved"
                stored.reviewer = reviewer
                stored.review_note = review_note
            return
        async with self._sessions.begin() as session:
            stored = await session.get(CommerceCoverageReview, review_id)
            if stored is None:
                raise NotFoundError("Coverage review")
            stored.status = "approved" if approve else "rejected"
            stored.reviewer = reviewer
            stored.review_note = review_note
            if not approve:
                return
            await self._apply_approved_review(session, stored)

    async def _apply_approved_review(
        self,
        session: AsyncSession,
        review: CommerceCoverageReview,
    ) -> None:
        payload = dict(review.proposal_payload)
        source = "coverage_review"
        if review.proposed_type == "alias":
            alias = str(payload.get("alias", "")).strip()
            category_key = str(payload.get("category_key", "")).strip()
            if not alias or not category_key:
                raise ValidationError("Alias proposals require alias and category_key")
            node = await self._upsert_semantic_node(
                session,
                node_type="CommerceAlias",
                name=alias,
                description=str(payload.get("description", "")),
                source=source,
                confidence=float(payload.get("confidence", 0.8)),
                properties={"alias": alias},
            )
            await self._upsert_semantic_relation(
                session,
                source_ref=f"node:{node.id}",
                target_ref=f"category:{category_key}",
                relation_type="ALIAS_OF",
                description=f"{alias}是{category_key}的用户语言表达",
                source=source,
                confidence=float(payload.get("confidence", 0.8)),
                properties={"review_id": review.id},
            )
            return
        if review.proposed_type == "use_case":
            name = str(payload.get("name", "")).strip()
            category_keys = [str(key) for key in payload.get("category_keys", [])]
            if not name or not category_keys:
                raise ValidationError("Use-case proposals require name and category_keys")
            node = await self._upsert_semantic_node(
                session,
                node_type="CommerceUseCase",
                name=name,
                description=str(payload.get("description", "")),
                source=source,
                confidence=float(payload.get("confidence", 0.8)),
                properties={"use_case_name": name},
            )
            for category_key in category_keys:
                await self._upsert_semantic_relation(
                    session,
                    source_ref=f"node:{node.id}",
                    target_ref=f"category:{category_key}",
                    relation_type="SERVED_BY",
                    description=f"{name}场景由{category_key}类目承接",
                    source=source,
                    confidence=float(payload.get("confidence", 0.8)),
                    properties={"review_id": review.id},
                )
            return
        if review.proposed_type == "relation":
            source_ref = str(payload.get("source_ref", "")).strip()
            target_ref = str(payload.get("target_ref", "")).strip()
            relation_type = str(payload.get("relation_type", "")).strip()
            if not source_ref or not target_ref or not relation_type:
                raise ValidationError(
                    "Relation proposals require source_ref, target_ref and relation_type"
                )
            await self._upsert_semantic_relation(
                session,
                source_ref=source_ref,
                target_ref=target_ref,
                relation_type=relation_type,
                description=str(payload.get("description", "")),
                source=source,
                confidence=float(payload.get("confidence", 0.8)),
                properties={"review_id": review.id},
            )
            return
        raise ValidationError("Unsupported approved review type")

    async def list_reviews(
        self, *, status: str | None = None, limit: int = 100
    ) -> list[CommerceCoverageReview]:
        async with self._sessions() as session:
            statement = select(CommerceCoverageReview).order_by(
                CommerceCoverageReview.updated_at.desc()
            )
            if status is not None:
                statement = statement.where(CommerceCoverageReview.status == status)
            rows = (await session.scalars(statement.limit(limit))).all()
            return list(rows)

    async def coverage_metrics(self) -> dict[str, int]:
        async with self._sessions() as session:
            total = await session.scalar(select(func.count(CommerceCoverageEvent.id))) or 0
            zero = (
                await session.scalar(
                    select(func.count(CommerceCoverageEvent.id)).where(
                        CommerceCoverageEvent.status == "zero_result"
                    )
                )
                or 0
            )
            low = (
                await session.scalar(
                    select(func.count(CommerceCoverageEvent.id)).where(
                        CommerceCoverageEvent.status == "low_result"
                    )
                )
                or 0
            )
            pending = (
                await session.scalar(
                    select(func.count(CommerceCoverageReview.id)).where(
                        CommerceCoverageReview.status == "pending"
                    )
                )
                or 0
            )
            approved = (
                await session.scalar(
                    select(func.count(CommerceCoverageReview.id)).where(
                        CommerceCoverageReview.status == "approved"
                    )
                )
                or 0
            )
            rejected = (
                await session.scalar(
                    select(func.count(CommerceCoverageReview.id)).where(
                        CommerceCoverageReview.status == "rejected"
                    )
                )
                or 0
            )
        return {
            "total_events": total,
            "zero_result_events": zero,
            "low_result_events": low,
            "pending_reviews": pending,
            "approved_reviews": approved,
            "rejected_reviews": rejected,
        }
