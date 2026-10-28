"""External commerce boundary: typed business calls, no model credentials."""

from typing import Any, Literal, Protocol

from app.models.schemas.commerce import (
    CartSnapshot,
    CatalogItem,
    OrderSnapshot,
    PaymentSnapshot,
    QuoteSnapshot,
)


class ProviderRejected(Exception):
    """Definitive refusal; no successful external operation is implied."""


class ProviderUnknown(Exception):
    """A call may have committed. Reconcile its original request key."""


class CommercePort(Protocol):
    mode: Literal["mock"]

    async def search_products(self, owner_user_id: int, query: str) -> list[CatalogItem]: ...
    async def get_product(self, owner_user_id: int, sku_id: str) -> CatalogItem: ...
    async def set_cart(
        self,
        owner_user_id: int,
        cart_id: str,
        sku_id: str,
        quantity: int,
        expected_cart_version: int | None,
        request_key: str,
    ) -> CartSnapshot: ...
    async def get_cart(self, owner_user_id: int, cart_id: str) -> CartSnapshot: ...
    async def prepare_checkout(
        self, owner_user_id: int, cart_id: str, expected_cart_version: int, request_key: str
    ) -> QuoteSnapshot: ...
    async def create_order(
        self, owner_user_id: int, quote_id: str, request_key: str
    ) -> OrderSnapshot: ...
    async def create_payment(
        self, owner_user_id: int, order_id: str, request_key: str
    ) -> PaymentSnapshot: ...
    async def mock_pay(
        self, owner_user_id: int, payment_id: str, request_key: str
    ) -> PaymentSnapshot: ...
    async def get_status(self, owner_user_id: int, payment_id: str) -> PaymentSnapshot: ...
    async def lookup_operation(
        self, owner_user_id: int, request_key: str
    ) -> dict[str, Any] | None: ...
