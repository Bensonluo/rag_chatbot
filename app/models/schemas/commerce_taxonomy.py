"""Admin-facing schemas for commerce taxonomy and coverage workflows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.schemas.commerce import CatalogCategory, CatalogTaxonomy


class RegistryCategory(CatalogCategory):
    """Internal category lifecycle metadata not exposed on provider wires."""

    model_config = ConfigDict(extra="forbid")

    parent_id: str | None = Field(default=None, max_length=64)
    domain: str = Field(default="general", min_length=1, max_length=64)
    provider_policy: Literal["generated", "external", "unsupported"] = "generated"
    status: Literal["active", "planned", "deprecated"] = "active"
    provider_aliases: list[str] = Field(default_factory=list, max_length=32)
    base_price_minor: int | None = Field(default=None, ge=0)


class RegistryTaxonomy(BaseModel):
    """Versioned registry taxonomy consumed by generated providers and graph."""

    model_config = ConfigDict(extra="forbid")

    taxonomy_version: str | None = Field(default=None, min_length=1, max_length=64)
    categories: list[RegistryCategory] = Field(min_length=1)

    def to_catalog_taxonomy(self) -> CatalogTaxonomy:
        """Project the public provider wire model without registry metadata."""

        return CatalogTaxonomy(categories=self.categories)


class TaxonomyVersionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version_id: str = Field(min_length=1, max_length=64)
    taxonomy: RegistryTaxonomy
    activate: bool = False


class TaxonomyVersionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version_id: str
    status: Literal["draft", "active", "archived"]
    category_count: int
    attribute_count: int
    activated_at: datetime | None = None


class ExternalCategoryMappingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    internal_category_key: str = Field(min_length=1, max_length=64)
    provider: str = Field(min_length=1, max_length=64)
    external_category_id: str = Field(min_length=1, max_length=128)
    external_path: str = Field(default="", max_length=500)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: Literal["active", "planned", "deprecated"] = "active"


class CoverageEventResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    query: str
    normalized_query: str
    owner_hash: str
    matched_category_ids: list[str]
    nearest_category_ids: list[str]
    result_count: int
    status: Literal["zero_result", "low_result", "wrong_result"]
    created_at: datetime


class CoverageReviewCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: int | None = None
    proposed_type: Literal["category", "alias", "use_case", "relation"]
    proposal_payload: dict[str, Any] = Field(default_factory=dict)


class CoverageReviewDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: str = Field(min_length=1, max_length=128)
    review_note: str = Field(default="", max_length=2000)


class CoverageReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    event_id: int | None
    proposed_type: Literal["category", "alias", "use_case", "relation"]
    proposal_payload: dict[str, Any]
    status: Literal["pending", "approved", "rejected"]
    reviewer: str | None
    review_note: str


class CoverageMetricsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_events: int
    zero_result_events: int
    low_result_events: int
    pending_reviews: int
    approved_reviews: int
    rejected_reviews: int
