"""Versioned commerce taxonomy and coverage storage.

These tables are operational metadata for the shopping architecture. They do
not replace the commerce provider: SKU, price, stock, carts, quotes, orders
and payments remain provider-owned facts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database.base import Base, TimestampMixin


class CommerceTaxonomyVersion(Base, TimestampMixin):
    """An immutable taxonomy snapshot that can be activated or archived."""

    __tablename__ = "commerce_taxonomy_versions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('draft','active','archived')", name="ck_taxonomy_version_status"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CommerceTaxonomyCategory(Base, TimestampMixin):
    """One category in a taxonomy version."""

    __tablename__ = "commerce_taxonomy_categories"
    __table_args__ = (
        UniqueConstraint("version_id", "category_key", name="uq_taxonomy_category_key"),
        CheckConstraint(
            "provider_policy IN ('generated','external','unsupported')", name="ck_taxonomy_policy"
        ),
        CheckConstraint(
            "status IN ('active','planned','deprecated')", name="ck_taxonomy_category_status"
        ),
        Index("ix_commerce_taxonomy_categories_version_id", "version_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("commerce_taxonomy_versions.id", ondelete="CASCADE"), nullable=False
    )
    category_key: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    domain: Mapped[str] = mapped_column(String(64), nullable=False, default="general")
    provider_policy: Mapped[str] = mapped_column(String(16), nullable=False, default="generated")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    provider_aliases: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    base_price_minor: Mapped[int | None] = mapped_column(Integer, nullable=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class CommerceTaxonomyAttribute(Base, TimestampMixin):
    """One allowed attribute vocabulary for a taxonomy category."""

    __tablename__ = "commerce_taxonomy_attributes"
    __table_args__ = (
        UniqueConstraint("version_id", "category_key", "name", name="uq_taxonomy_attribute_key"),
        Index("ix_commerce_taxonomy_attributes_version_id", "version_id"),
        Index("ix_commerce_taxonomy_attributes_category_key", "category_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("commerce_taxonomy_versions.id", ondelete="CASCADE"), nullable=False
    )
    category_key: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    allowed_values: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    required: Mapped[bool] = mapped_column(default=False, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class CommerceExternalCategoryMapping(Base, TimestampMixin):
    """Map an internal category to a provider-specific category tree."""

    __tablename__ = "commerce_external_category_mappings"
    __table_args__ = (
        UniqueConstraint(
            "version_id",
            "provider",
            "external_category_id",
            name="uq_external_category_mapping",
        ),
        CheckConstraint(
            "status IN ('active','planned','deprecated')", name="ck_external_mapping_status"
        ),
        Index("ix_commerce_external_category_mappings_version_id", "version_id"),
        Index(
            "ix_commerce_external_category_mappings_internal_category_key", "internal_category_key"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("commerce_taxonomy_versions.id", ondelete="CASCADE"), nullable=False
    )
    internal_category_key: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    external_category_id: Mapped[str] = mapped_column(String(128), nullable=False)
    external_path: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")


class CommerceSemanticNode(Base, TimestampMixin):
    """Approved alias/use-case/concept node used in the commerce projection."""

    __tablename__ = "commerce_semantic_nodes"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','approved','rejected')", name="ck_semantic_node_status"
        ),
        Index("ix_commerce_semantic_nodes_node_type", "node_type"),
        Index("ix_commerce_semantic_nodes_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    node_type: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    properties: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="approved")
    version: Mapped[str] = mapped_column(String(64), nullable=False, default="semantic-v1")


class CommerceSemanticRelation(Base, TimestampMixin):
    """Approved relation between semantic nodes and/or standard categories."""

    __tablename__ = "commerce_semantic_relations"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','approved','rejected')", name="ck_semantic_relation_status"
        ),
        Index("ix_commerce_semantic_relations_relation_type", "relation_type"),
        Index("ix_commerce_semantic_relations_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source_ref: Mapped[str] = mapped_column(String(128), nullable=False)
    target_ref: Mapped[str] = mapped_column(String(128), nullable=False)
    relation_type: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    properties: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="approved")
    version: Mapped[str] = mapped_column(String(64), nullable=False, default="semantic-v1")


class CommerceCoverageEvent(Base):
    """A shopping query whose category coverage deserves review."""

    __tablename__ = "commerce_coverage_events"
    __table_args__ = (
        CheckConstraint(
            "status IN ('zero_result','low_result','wrong_result')", name="ck_coverage_status"
        ),
        Index("ix_commerce_coverage_events_normalized_query", "normalized_query"),
        Index("ix_commerce_coverage_events_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(String(500), nullable=False)
    normalized_query: Mapped[str] = mapped_column(String(500), nullable=False)
    owner_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    matched_category_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    nearest_category_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    result_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="zero_result")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class CommerceCoverageReview(Base, TimestampMixin):
    """Review decision for a coverage event or generated semantic proposal."""

    __tablename__ = "commerce_coverage_reviews"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','approved','rejected')", name="ck_coverage_review_status"
        ),
        CheckConstraint(
            "proposed_type IN ('category','alias','use_case','relation')",
            name="ck_coverage_proposed_type",
        ),
        Index("ix_commerce_coverage_reviews_event_id", "event_id"),
        Index("ix_commerce_coverage_reviews_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("commerce_coverage_events.id", ondelete="SET NULL"),
        nullable=True,
    )
    proposed_type: Mapped[str] = mapped_column(String(16), nullable=False)
    proposal_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    reviewer: Mapped[str | None] = mapped_column(String(128), nullable=True)
    review_note: Mapped[str] = mapped_column(Text, nullable=False, default="")
