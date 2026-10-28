"""Purchase orchestration and separately committed mock-provider facts."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database.base import Base, TimestampMixin


class PurchaseSession(Base, TimestampMixin):
    __tablename__ = "commerce_purchases"
    __table_args__ = (
        CheckConstraint("version >= 0 AND generation >= 0", name="ck_commerce_purchase_versions"),
        UniqueConstraint("active_slot", name="uq_commerce_active_slot"),
        CheckConstraint("quantity >= 0 AND quantity <= 10", name="ck_commerce_purchase_quantity"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), index=True
    )
    active_slot: Mapped[str | None] = mapped_column(String(96))
    mode: Mapped[str] = mapped_column(String(16), default="mock")
    merchant_id: Mapped[str] = mapped_column(String(64), default="demo-store")
    currency: Mapped[str] = mapped_column(String(3), default="CNY")
    stage: Mapped[str] = mapped_column(String(32), default="shopping")
    version: Mapped[int] = mapped_column(Integer, default=0)
    generation: Mapped[int] = mapped_column(Integer, default=0)
    sku_id: Mapped[str | None] = mapped_column(String(64))
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    cart_id: Mapped[str | None] = mapped_column(String(64))
    cart_version: Mapped[int | None] = mapped_column(Integer)
    cart_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    quote_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    confirmation_id: Mapped[str | None] = mapped_column(String(36))
    order_id: Mapped[str | None] = mapped_column(String(64))
    payment_id: Mapped[str | None] = mapped_column(String(64))
    payment_status: Mapped[str | None] = mapped_column(String(32))
    active_operation_id: Mapped[str | None] = mapped_column(String(36))
    error: Mapped[str | None] = mapped_column(String(255))


class PurchaseOperation(Base, TimestampMixin):
    __tablename__ = "commerce_operations"
    __table_args__ = (
        UniqueConstraint(
            "purchase_id", "kind", "generation", name="uq_commerce_operation_generation"
        ),
        CheckConstraint(
            "status IN ('reserved','unknown','completed','failed')",
            name="ck_commerce_operation_status",
        ),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    purchase_id: Mapped[str] = mapped_column(
        ForeignKey("commerce_purchases.id", ondelete="RESTRICT"), index=True
    )
    kind: Mapped[str] = mapped_column(String(32))
    generation: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="reserved")
    request_hash: Mapped[str] = mapped_column(String(64))
    request: Mapped[dict[str, Any]] = mapped_column(JSON)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True))
    error: Mapped[str | None] = mapped_column(String(255))


class PurchaseConfirmation(Base, TimestampMixin):
    __tablename__ = "commerce_confirmations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    purchase_id: Mapped[str] = mapped_column(
        ForeignKey("commerce_purchases.id", ondelete="RESTRICT"), index=True
    )
    purchase_version: Mapped[int] = mapped_column(Integer)
    generation: Mapped[int] = mapped_column(Integer)
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSON)
    snapshot_hash: Mapped[str] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    operation_id: Mapped[str | None] = mapped_column(String(36))


class PurchaseAction(Base, TimestampMixin):
    __tablename__ = "commerce_actions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    purchase_id: Mapped[str] = mapped_column(
        ForeignKey("commerce_purchases.id", ondelete="RESTRICT"), index=True
    )
    kind: Mapped[str] = mapped_column(String(32))
    expected_version: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class MockCommerceObject(Base, TimestampMixin):
    """External emulator storage: deliberately has no purchase/chat/user foreign key."""

    __tablename__ = "commerce_mock_objects"
    __table_args__ = (UniqueConstraint("kind", "parent_id", name="uq_commerce_mock_parent"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)
    owner_user_id: Mapped[int] = mapped_column(Integer)
    parent_id: Mapped[str | None] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer, default=0)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class MockCommerceReceipt(Base, TimestampMixin):
    __tablename__ = "commerce_mock_receipts"
    request_key: Mapped[str] = mapped_column(String(96), primary_key=True)
    owner_user_id: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(32))
    request_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    result: Mapped[dict[str, Any]] = mapped_column(JSON)
