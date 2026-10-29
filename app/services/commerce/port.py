"""Typed external commerce boundary: facts come only from the provider.

The protocol accepts structured search/pagination and multi-line cart
mutations. It never parses user language: the LLM Agent constructs typed
arguments, while owner binding and idempotency remain local, non-semantic
orchestration concerns.
"""

from collections.abc import Awaitable
from typing import Any, Literal, Protocol, TypedDict

from app.models.schemas.commerce import (
    CartLineSelection,
    CartSnapshot,
    CatalogFilters,
    CatalogItem,
    CatalogPage,
    CatalogTaxonomy,
    OrderSnapshot,
    PaymentSnapshot,
    QuoteSnapshot,
)

CommerceSort = Literal["relevance", "price_asc", "price_desc", "newest"]


class ProviderOperationReceipt(TypedDict):
    """Normalized provider operation receipt used for unknown reconciliation."""

    kind: str
    status: Literal["completed", "failed", "unknown"]
    data: dict[str, Any]


class ProviderRejected(Exception):
    """Definitive refusal; no successful external operation is implied."""


class ProviderUnknown(Exception):
    """A call may have committed. Reconcile its original request key."""


class CommercePort(Protocol):
    mode: str

    def search_products(
        self,
        owner_user_id: int,
        query: str,
        *,
        filters: CatalogFilters | None = None,
        sort: CommerceSort = "relevance",
        page_size: int = 10,
        page_token: str | None = None,
    ) -> Awaitable[CatalogPage]: ...

    def catalog_categories(self, owner_user_id: int) -> Awaitable[CatalogTaxonomy]: ...

    def get_product(self, owner_user_id: int, sku_id: str) -> Awaitable[CatalogItem]: ...

    def update_cart_lines(
        self,
        owner_user_id: int,
        cart_id: str | None,
        expected_cart_version: int | None,
        lines: list[CartLineSelection],
        request_key: str,
    ) -> Awaitable[CartSnapshot]: ...

    def get_cart(self, owner_user_id: int, cart_id: str) -> Awaitable[CartSnapshot]: ...

    def prepare_checkout(
        self,
        owner_user_id: int,
        cart_id: str,
        expected_cart_version: int,
        request_key: str,
    ) -> Awaitable[QuoteSnapshot]: ...

    def create_order(
        self, owner_user_id: int, quote_id: str, request_key: str
    ) -> Awaitable[OrderSnapshot]: ...

    def create_payment(
        self, owner_user_id: int, order_id: str, request_key: str
    ) -> Awaitable[PaymentSnapshot]: ...

    def mock_pay(
        self, owner_user_id: int, payment_id: str, request_key: str
    ) -> Awaitable[PaymentSnapshot]: ...

    def get_order(self, owner_user_id: int, order_id: str) -> Awaitable[OrderSnapshot]: ...

    def get_status(self, owner_user_id: int, payment_id: str) -> Awaitable[PaymentSnapshot]: ...

    def lookup_operation(
        self, owner_user_id: int, request_key: str
    ) -> Awaitable[ProviderOperationReceipt | None]: ...


__all__ = [
    "CommercePort",
    "CommerceSort",
    "ProviderOperationReceipt",
    "ProviderRejected",
    "ProviderUnknown",
]
