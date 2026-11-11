"""Typed mock-commerce facts projected into assistant chat text."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

CommerceStage = Literal[
    "shopping",
    "cart",
    "quoted",
    "submitting",
    "unknown",
    "awaiting_payment",
    "processing",
    "paid",
    "refunded",
    "failed",
]


class CommerceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CatalogItem(CommerceModel):
    product_id: str
    sku_id: str
    name: str
    description: str
    attributes: dict[str, str]
    unit_price_minor: int = Field(ge=0)
    currency: Literal["CNY"] = "CNY"
    available_quantity: int = Field(ge=0)
    max_order_quantity: int = Field(ge=1)


class CatalogFilters(CommerceModel):
    min_price_minor: int | None = Field(default=None, ge=0)
    max_price_minor: int | None = Field(default=None, ge=0)
    attributes: dict[str, list[str]] = Field(default_factory=dict)
    in_stock: bool = True


class CatalogPage(CommerceModel):
    items: list[CatalogItem]
    next_page_token: str | None = None
    total_estimate: int = Field(ge=0)


class CategoryAttribute(CommerceModel):
    """One filterable catalog attribute with the provider's allowed values."""

    name: str = Field(min_length=1, max_length=64)
    allowed_values: list[str] = Field(min_length=1, max_length=32)


class CatalogCategory(CommerceModel):
    """Provider-owned product category; authoritative, not user language.

    Deliberately carries no alias list: mapping an open user need onto
    standard categories is the LLM Agent's semantic job, never a
    provider-published phrase table.
    """

    category_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=128)
    description: str = Field(default="", max_length=1000)
    attributes: list[CategoryAttribute] = Field(default_factory=list, max_length=16)


class CatalogTaxonomy(CommerceModel):
    """The full authoritative category taxonomy of the provider catalog."""

    categories: list[CatalogCategory] = Field(min_length=1)


class MintBrandStyle(CommerceModel):
    """One LLM-authored (brand, series) pairing on a minted category."""

    brand: str = Field(min_length=1, max_length=16)
    series: str = Field(min_length=1, max_length=16)


class MintedCategory(CommerceModel):
    """A runtime-minted dynamic category (or its existing twin) as minted.

    Mirrors the provider's mint wire: registry-internal fields (domain,
    variant axis, base price) ride along so the agent can immediately re-search
    with the right vocabulary; the public category view stays ``CatalogCategory``.
    """

    category_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=300)
    domain: str = Field(min_length=1, max_length=64)
    attributes: list[CategoryAttribute] = Field(default_factory=list, max_length=8)
    variant_key: str = Field(min_length=1, max_length=32)
    variant_values: list[str] = Field(min_length=4, max_length=4)
    base_price_minor: int | None = Field(default=None, ge=0)
    brand_styles: list[MintBrandStyle] | None = Field(default=None, max_length=8)


class MintCategoryOutcome(CommerceModel):
    """Result of proposing a dynamic category.

    ``minted`` created and registered a new category; ``existing`` means a
    normalized-name twin already existed and was returned unchanged. Either
    way the category is immediately searchable and purchasable.
    """

    status: Literal["minted", "existing"]
    category: MintedCategory
    dynamic_category_count: int = Field(ge=0)


class CartLineSelection(CommerceModel):
    sku_id: str = Field(min_length=1, max_length=64)
    quantity: int = Field(strict=True, ge=1)
    provider: Literal["generated", "external"] | None = Field(default=None, exclude=True)


class CartLine(CommerceModel):
    sku_id: str
    name: str
    attributes: dict[str, str]
    quantity: int = Field(ge=1)
    unit_price_minor: int = Field(ge=0)
    subtotal_minor: int = Field(ge=0)


class CartSnapshot(CommerceModel):
    cart_id: str
    cart_version: int = Field(ge=1)
    lines: list[CartLine]
    subtotal_minor: int = Field(ge=0)
    currency: Literal["CNY"] = "CNY"


class QuoteSnapshot(CommerceModel):
    quote_id: str
    cart_id: str
    cart_version: int
    lines: list[CartLine]
    subtotal_minor: int = Field(ge=0)
    shipping_minor: int = Field(ge=0)
    total_minor: int = Field(ge=0)
    currency: Literal["CNY"] = "CNY"
    expires_at: datetime
    fulfillment: str = "模拟配送：无需真实地址"


class OrderSnapshot(CommerceModel):
    order_id: str
    quote_id: str
    total_minor: int = Field(ge=0)
    currency: Literal["CNY"] = "CNY"
    status: Literal["awaiting_payment", "processing", "paid", "refunded", "failed"]


class PaymentSnapshot(CommerceModel):
    payment_id: str
    order_id: str
    total_minor: int = Field(ge=0)
    currency: Literal["CNY"] = "CNY"
    status: Literal["awaiting_payment", "processing", "paid", "refunded", "failed"]


class RefundSnapshot(CommerceModel):
    """Authoritative refund receipt; deterministic id keyed to the order.

    ``refund_id`` is derived from the order id in both ports, so a retry
    with a fresh request key after an unknown outcome replays the SAME
    receipt instead of creating a second refund — refunds are exactly-once
    per order by construction, like payments.
    """

    refund_id: str
    order_id: str
    payment_id: str
    total_minor: int = Field(ge=0)
    currency: Literal["CNY"] = "CNY"
    status: Literal["refunded"] = "refunded"


class CatalogBlock(CommerceModel):
    type: Literal["catalog"] = "catalog"
    title: str = "模拟商品目录"
    items: list[CatalogItem]
    next_page_token: str | None = None
    total_estimate: int = Field(default=0, ge=0)


class CartBlock(CommerceModel):
    type: Literal["cart"] = "cart"
    title: str = "模拟购物车"
    cart: CartSnapshot


class QuoteBlock(CommerceModel):
    type: Literal["quote"] = "quote"
    title: str = "模拟结算确认（不会真实扣款）"
    confirmation_id: str
    quote: QuoteSnapshot


class PaymentBlock(CommerceModel):
    type: Literal["payment"] = "payment"
    title: str = "模拟付款（不会真实扣款）"
    order_id: str
    payment_id: str | None
    status: str
    total_minor: int
    currency: Literal["CNY"] = "CNY"


class StatusBlock(CommerceModel):
    type: Literal["status"] = "status"
    title: str = "模拟购买进度"
    stage: CommerceStage
    message: str
    operation_id: str | None = None


class SpendLimitBlock(CommerceModel):
    """Structured refusal when a scoped-delegation spend limit blocks an order.

    The block is a server fact: it carries the configured limit, the
    attempted amount and the handoff ticket that a human can resolve to
    delegate the purchase. No field is model-generated.
    """

    type: Literal["spend_limit"] = "spend_limit"
    title: str = "消费限额保护"
    limit_kind: Literal["single", "daily"]
    limit_minor: int = Field(ge=0)
    attempted_minor: int = Field(ge=0)
    window_paid_minor: int | None = Field(default=None, ge=0)
    ticket_id: int | None = None
    message: str


class ApprovalLine(CommerceModel):
    """One orderable line of the confirmation card, copied from the quote."""

    sku_id: str
    name: str
    quantity: int = Field(ge=1)
    unit_price_minor: int = Field(ge=0)
    subtotal_minor: int = Field(ge=0)


class ApprovalCard(CommerceModel):
    """Structured confirmation card for the irreversible next step.

    Every field is copied from the durable PurchaseConfirmation snapshot
    and the purchase row it belongs to — the LLM never authors card data,
    and the card total is the amount an order must match. ``confirm_order``
    cards are bound to the quote TTL: once ``expires_at`` passes the card
    disappears because the order it describes can no longer be created.
    ``confirm_payment`` cards describe an already-created order, so the
    quote window is informational there, not a payment deadline.
    """

    purchase_id: str
    stage: Literal["quoted", "awaiting_payment"]
    action: Literal["confirm_order", "confirm_payment"]
    action_label: Literal["确认下单", "确认支付"]
    lines: list[ApprovalLine] = Field(min_length=1, max_length=50)
    total_minor: int = Field(ge=0)
    currency: Literal["CNY"] = "CNY"
    quote_hash: str
    expires_at: datetime
    purchase_version: int = Field(ge=0)
    generation: int = Field(ge=0)


class WatchCard(CommerceModel):
    """Structured ask card for a fired price-watch trigger.

    Every field is copied from the durable watch rule row and the
    provider price fact that satisfied it — the LLM never authors card
    data. The card only ASKS whether to order now ("pre-authorized
    condition + trigger + confirm"): ordering still goes through the
    existing quote → order → payment confirmation state machine, and a
    watch rule itself never creates cart, order or payment state.
    ``expires_at`` is the rule's lifetime bound, informational only —
    unlike an order card it carries no deadline semantics.
    """

    watch_id: str
    sku_id: str
    sku_name: str
    target_price_minor: int = Field(ge=1)
    current_price_minor: int = Field(ge=0)
    currency: Literal["CNY"] = "CNY"
    condition: Literal["at_or_below"] = "at_or_below"
    action: Literal["ask_order"] = "ask_order"
    action_label: Literal["现在下单"] = "现在下单"
    triggered_at: datetime
    expires_at: datetime


CommerceBlock = Annotated[
    CatalogBlock | CartBlock | QuoteBlock | PaymentBlock | StatusBlock | SpendLimitBlock,
    Field(discriminator="type"),
]


class CommerceView(CommerceModel):
    schema_version: Literal[1] = 1
    purchase_id: str
    version: int
    mode: Literal["mock"] = "mock"
    stage: CommerceStage
    snapshot_at: datetime
    blocks: list[CommerceBlock]
    approval: ApprovalCard | None = None
    watch: WatchCard | None = None
