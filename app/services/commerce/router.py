"""Route commerce calls across generated and external provider adapters."""

from __future__ import annotations

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
from app.services.commerce.port import (
    CommercePort,
    CommerceSort,
    ProviderOperationReceipt,
    ProviderRejected,
)
from app.services.commerce.taxonomy import CommerceTaxonomyRegistry


class RouterCommercePort:
    """Provider router driven by the active taxonomy registry.

    Category-level routing is explicit: generated categories stay on the
    generated provider, externally mapped categories go to the configured
    external adapter, and unsupported domains return an empty typed page rather
    than inventing products.
    """

    mode = "router"

    def __init__(
        self,
        *,
        generated: CommercePort,
        external: CommercePort,
        registry: CommerceTaxonomyRegistry,
        external_provider: str = "external",
    ) -> None:
        self._generated = generated
        self._external = external
        self._registry = registry
        self._external_provider = external_provider

    async def catalog_categories(self, owner_user_id: int) -> CatalogTaxonomy:
        if owner_user_id <= 0:
            raise ProviderRejected("Invalid commerce owner")
        return (await self._registry.load_active()).to_catalog_taxonomy()

    async def _route_for_query(self, query: str) -> str | None:
        taxonomy = await self._registry.load_active()
        normalized = " ".join(query.strip().lower().split())
        for category in taxonomy.categories:
            aliases = {
                category.name.lower(),
                *(alias.lower() for alias in category.provider_aliases),
            }
            if normalized and normalized in aliases:
                return category.provider_policy
        return None

    async def search_products(
        self,
        owner_user_id: int,
        query: str,
        *,
        filters: CatalogFilters | None = None,
        sort: CommerceSort = "relevance",
        page_size: int = 10,
        page_token: str | None = None,
    ) -> CatalogPage:
        route = await self._route_for_query(query)
        if route == "unsupported":
            return CatalogPage(items=[], next_page_token=None, total_estimate=0)
        if route == "external":
            external_query = query
            taxonomy = await self._registry.load_active()
            category_key = next(
                (
                    category.category_id
                    for category in taxonomy.categories
                    if category.name.lower() == " ".join(query.strip().lower().split())
                    or " ".join(query.strip().lower().split())
                    in {alias.lower() for alias in category.provider_aliases}
                ),
                None,
            )
            if category_key is not None:
                mapping = await self._registry.find_external_mapping(
                    provider=self._external_provider,
                    internal_category_key=category_key,
                )
                if mapping is not None:
                    external_query = mapping.external_category_id
            return await self._external.search_products(
                owner_user_id,
                external_query,
                filters=filters,
                sort=sort,
                page_size=page_size,
                page_token=page_token,
            )
        if route == "generated":
            return await self._generated.search_products(
                owner_user_id,
                query,
                filters=filters,
                sort=sort,
                page_size=page_size,
                page_token=page_token,
            )

        generated_page = await self._generated.search_products(
            owner_user_id,
            query,
            filters=filters,
            sort=sort,
            page_size=page_size,
            page_token=None,
        )
        external_page = await self._external.search_products(
            owner_user_id,
            query,
            filters=filters,
            sort=sort,
            page_size=page_size,
            page_token=None,
        )
        merged: list[CatalogItem] = []
        left = list(generated_page.items)
        right = list(external_page.items)
        while left or right:
            if left:
                merged.append(left.pop(0))
            if right:
                merged.append(right.pop(0))
        return CatalogPage(
            items=merged[:page_size],
            next_page_token=(
                generated_page.next_page_token
                if external_page.next_page_token is None and generated_page.next_page_token
                else None
            ),
            total_estimate=generated_page.total_estimate + external_page.total_estimate,
        )

    async def get_product(self, owner_user_id: int, sku_id: str) -> CatalogItem:
        try:
            return await self._generated.get_product(owner_user_id, sku_id)
        except Exception:
            return await self._external.get_product(owner_user_id, sku_id)

    def _group_lines(self, lines: list[CartLineSelection]) -> str:
        providers = {line.provider or "generated" for line in lines}
        if len(providers) != 1:
            raise ProviderRejected("Mixed-provider carts are not supported in one mutation")
        return providers.pop()

    async def update_cart_lines(
        self,
        owner_user_id: int,
        cart_id: str | None,
        expected_cart_version: int | None,
        lines: list[CartLineSelection],
        request_key: str,
    ) -> CartSnapshot:
        provider = self._group_lines(lines)
        target = self._external if provider == "external" else self._generated
        return await target.update_cart_lines(
            owner_user_id,
            cart_id,
            expected_cart_version,
            lines,
            request_key,
        )

    async def get_cart(self, owner_user_id: int, cart_id: str) -> CartSnapshot:
        try:
            return await self._generated.get_cart(owner_user_id, cart_id)
        except Exception:
            return await self._external.get_cart(owner_user_id, cart_id)

    async def prepare_checkout(
        self,
        owner_user_id: int,
        cart_id: str,
        expected_cart_version: int,
        request_key: str,
    ) -> QuoteSnapshot:
        try:
            return await self._generated.prepare_checkout(
                owner_user_id,
                cart_id,
                expected_cart_version,
                request_key,
            )
        except Exception:
            return await self._external.prepare_checkout(
                owner_user_id,
                cart_id,
                expected_cart_version,
                request_key,
            )

    async def create_order(
        self, owner_user_id: int, quote_id: str, request_key: str
    ) -> OrderSnapshot:
        try:
            return await self._generated.create_order(owner_user_id, quote_id, request_key)
        except Exception:
            return await self._external.create_order(owner_user_id, quote_id, request_key)

    async def create_payment(
        self, owner_user_id: int, order_id: str, request_key: str
    ) -> PaymentSnapshot:
        try:
            return await self._generated.create_payment(owner_user_id, order_id, request_key)
        except Exception:
            return await self._external.create_payment(owner_user_id, order_id, request_key)

    async def mock_pay(
        self, owner_user_id: int, payment_id: str, request_key: str
    ) -> PaymentSnapshot:
        try:
            return await self._generated.mock_pay(owner_user_id, payment_id, request_key)
        except Exception:
            return await self._external.mock_pay(owner_user_id, payment_id, request_key)

    async def get_order(self, owner_user_id: int, order_id: str) -> OrderSnapshot:
        try:
            return await self._generated.get_order(owner_user_id, order_id)
        except Exception:
            return await self._external.get_order(owner_user_id, order_id)

    async def get_status(self, owner_user_id: int, payment_id: str) -> PaymentSnapshot:
        try:
            return await self._generated.get_status(owner_user_id, payment_id)
        except Exception:
            return await self._external.get_status(owner_user_id, payment_id)

    async def lookup_operation(
        self, owner_user_id: int, request_key: str
    ) -> ProviderOperationReceipt | None:
        receipt = await self._generated.lookup_operation(owner_user_id, request_key)
        if receipt is not None:
            return receipt
        return await self._external.lookup_operation(owner_user_id, request_key)
