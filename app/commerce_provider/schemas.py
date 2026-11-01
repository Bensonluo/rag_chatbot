"""Request bodies for the provider's internal HTTP API.

Success bodies are the shared commerce models from
``app.models.schemas.commerce`` (validated with ``extra="forbid"``), so these
request models intentionally mirror the wire contract documented in
``app/services/commerce/external_port.py``. Optional keys are omitted when
unset, exactly as the adapter sends them.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from app.commerce_provider.catalog import CommerceSort
from app.models.schemas.commerce import (
    CartLineSelection,
    CatalogFilters,
    CategoryAttribute,
    CommerceModel,
)

FaultKind = Literal[
    "update_cart_lines",
    "prepare_checkout",
    "create_order",
    "create_payment",
    "mock_pay",
]

FaultMode = Literal["reject", "unknown_after_commit", "processing_payment"]


class CatalogSearchRequest(CommerceModel):
    owner_user_id: int = Field(ge=1)
    query: str = Field(default="", max_length=500)
    sort: CommerceSort = "relevance"
    filters: CatalogFilters | None = None
    page_size: int = Field(default=10, ge=1, le=50)
    page_token: str | None = None


class UpdateCartLinesRequest(CommerceModel):
    owner_user_id: int = Field(ge=1)
    cart_id: str | None = Field(default=None, min_length=1, max_length=64)
    expected_cart_version: int | None = Field(default=None, ge=1)
    lines: list[CartLineSelection] = Field(default_factory=list, max_length=50)
    request_key: str = Field(min_length=1, max_length=128)


class PrepareCheckoutRequest(CommerceModel):
    owner_user_id: int = Field(ge=1)
    cart_id: str = Field(min_length=1, max_length=64)
    expected_cart_version: int = Field(ge=1)
    request_key: str = Field(min_length=1, max_length=128)


class CreateOrderRequest(CommerceModel):
    owner_user_id: int = Field(ge=1)
    quote_id: str = Field(min_length=1, max_length=64)
    request_key: str = Field(min_length=1, max_length=128)


class CreatePaymentRequest(CommerceModel):
    owner_user_id: int = Field(ge=1)
    order_id: str = Field(min_length=1, max_length=64)
    request_key: str = Field(min_length=1, max_length=128)


class SandboxPayRequest(CommerceModel):
    owner_user_id: int = Field(ge=1)
    request_key: str = Field(min_length=1, max_length=128)


class MintCategoryAttributeInput(CommerceModel):
    """One attribute axis proposed for a runtime-minted category."""

    name: str = Field(min_length=1, max_length=32)
    allowed_values: list[str] = Field(min_length=1, max_length=8)


class MintCategoryRequest(CommerceModel):
    """Proposal for minting a dynamic category (agent-driven, auditable)."""

    owner_user_id: int = Field(ge=1)
    request_key: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=300)
    domain: str = Field(min_length=1, max_length=64)
    attributes: list[MintCategoryAttributeInput] = Field(min_length=1, max_length=8)
    variant_key: str = Field(min_length=1, max_length=32)
    variant_values: list[str] = Field(min_length=4, max_length=4)
    base_price_minor: int = Field(ge=1, le=100_000_000)
    source_query: str = Field(default="", max_length=300)


class MintedCategoryView(CommerceModel):
    category_id: str
    name: str
    description: str
    domain: str
    attributes: list[CategoryAttribute]
    variant_key: str
    variant_values: list[str]
    base_price_minor: int | None = None


class MintCategoryResult(CommerceModel):
    """``minted`` created a new category; ``existing`` returned the twin."""

    status: Literal["minted", "existing"]
    category: MintedCategoryView
    dynamic_category_count: int = Field(ge=0)


class FaultPlanRequest(CommerceModel):
    kind: FaultKind
    mode: FaultMode
    target: str | None = Field(default=None, min_length=1, max_length=64)
    times: int = Field(default=1, ge=1, le=100)


class SettlePaymentRequest(CommerceModel):
    owner_user_id: int = Field(ge=1)
    payment_id: str = Field(min_length=1, max_length=64)
    paid: bool = True


OwnerParam = Annotated[int, Field(ge=1)]

__all__ = [
    "CatalogSearchRequest",
    "CreateOrderRequest",
    "CreatePaymentRequest",
    "FaultPlanRequest",
    "MintCategoryAttributeInput",
    "MintCategoryRequest",
    "MintCategoryResult",
    "MintedCategoryView",
    "OwnerParam",
    "PrepareCheckoutRequest",
    "SandboxPayRequest",
    "SettlePaymentRequest",
    "UpdateCartLinesRequest",
]
