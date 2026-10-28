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


class CartLine(CommerceModel):
    sku_id: str
    name: str
    attributes: dict[str, str]
    quantity: int = Field(ge=1, le=10)
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
    status: Literal["awaiting_payment", "processing", "paid", "failed"]


class PaymentSnapshot(CommerceModel):
    payment_id: str
    order_id: str
    total_minor: int = Field(ge=0)
    currency: Literal["CNY"] = "CNY"
    status: Literal["awaiting_payment", "processing", "paid", "failed"]


class CatalogBlock(CommerceModel):
    type: Literal["catalog"] = "catalog"
    title: str = "模拟商品目录"
    items: list[CatalogItem]


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


CommerceBlock = Annotated[
    CatalogBlock | CartBlock | QuoteBlock | PaymentBlock | StatusBlock, Field(discriminator="type")
]


class CommerceView(CommerceModel):
    schema_version: Literal[1] = 1
    purchase_id: str
    version: int
    mode: Literal["mock"] = "mock"
    stage: CommerceStage
    snapshot_at: datetime
    blocks: list[CommerceBlock]
