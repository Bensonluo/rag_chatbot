"""ExternalCommercePort adapter contract: typed facts, explicit failures, no fallback.

Every test drives the adapter through ``httpx.MockTransport`` — no sockets. The
provider wire paths and JSON fields asserted here are the documented contract at
the top of ``app/services/commerce/external_port.py``: the internal provider
service must match them. Provider refusals (4xx except 408/429) are definitive; timeouts,
connection failures, 5xx, and untyped bodies are unknown outcomes that must
never be answered by a local commerce implementation.
"""

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config.settings import Settings
from app.models.schemas.commerce import (
    CartLineSelection,
    CartSnapshot,
    CatalogFilters,
    CatalogItem,
    CatalogPage,
    OrderSnapshot,
    PaymentSnapshot,
    QuoteSnapshot,
)
from app.services.commerce import create_commerce_port
from app.services.commerce.external_port import ExternalCommercePort
from app.services.commerce.mock_port import MockCommercePort
from app.services.commerce.port import ProviderRejected, ProviderUnknown

BASE_URL = "http://commerce.internal:8000"
TOKEN = "internal-secret"
TIMEOUT_SECONDS = 2.5

CATALOG_ITEM_JSON = {
    "product_id": "headphones-anc",
    "sku_id": "ANC-BLACK",
    "name": "ANC 降噪耳机",
    "description": "主动降噪头戴耳机",
    "attributes": {"color": "黑色"},
    "unit_price_minor": 29900,
    "available_quantity": 5,
    "max_order_quantity": 2,
}

CART_JSON = {
    "cart_id": "cart-1",
    "cart_version": 3,
    "lines": [
        {
            "sku_id": "ANC-BLACK",
            "name": "ANC 降噪耳机",
            "attributes": {"color": "黑色"},
            "quantity": 2,
            "unit_price_minor": 29900,
            "subtotal_minor": 59800,
        }
    ],
    "subtotal_minor": 59800,
}

QUOTE_JSON = {
    "quote_id": "quote-1",
    "cart_id": "cart-1",
    "cart_version": 3,
    "lines": CART_JSON["lines"],
    "subtotal_minor": 59800,
    "shipping_minor": 0,
    "total_minor": 59800,
    "expires_at": "2026-10-29T12:00:00+00:00",
}

ORDER_JSON = {
    "order_id": "order-1",
    "quote_id": "quote-1",
    "total_minor": 59800,
    "status": "awaiting_payment",
}

PAYMENT_JSON = {
    "payment_id": "pay-1",
    "order_id": "order-1",
    "total_minor": 59800,
    "status": "awaiting_payment",
}

CATALOG_PAGE_JSON = {
    "items": [CATALOG_ITEM_JSON],
    "next_page_token": "page-2",
    "total_estimate": 3,
}

Handler = Callable[[httpx.Request], httpx.Response]


@asynccontextmanager
async def external_port(
    handler: Handler,
) -> AsyncIterator[tuple[ExternalCommercePort, list[httpx.Request]]]:
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(recording)) as client:
        yield (
            ExternalCommercePort(
                BASE_URL, token=TOKEN, timeout_seconds=TIMEOUT_SECONDS, client=client
            ),
            seen,
        )


def body_of(request: httpx.Request) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(request.content.decode())
    return body


def null_body(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=b"null", headers={"content-type": "application/json"})


class TestTypedSuccess:
    async def test_search_sends_structured_arguments_and_decodes_typed_page(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/commerce/search"
            return httpx.Response(200, json=CATALOG_PAGE_JSON)

        async with external_port(handler) as (port, seen):
            page = await port.search_products(
                7,
                "降噪耳机",
                filters=CatalogFilters(min_price_minor=10000, attributes={"color": ["黑色"]}),
                sort="price_asc",
                page_size=5,
                page_token="cursor-9",
            )

        assert isinstance(page, CatalogPage)
        assert page.total_estimate == 3
        assert page.next_page_token == "page-2"
        assert isinstance(page.items[0], CatalogItem)
        assert page.items[0].sku_id == "ANC-BLACK"

        request = seen[0]
        assert request.method == "POST"
        assert body_of(request) == {
            "owner_user_id": 7,
            "query": "降噪耳机",
            "sort": "price_asc",
            "page_size": 5,
            "filters": {
                "min_price_minor": 10000,
                "max_price_minor": None,
                "attributes": {"color": ["黑色"]},
                "in_stock": True,
            },
            "page_token": "cursor-9",
        }

    async def test_reads_use_stable_get_paths_with_owner_parameter(self):
        routes = {
            "/commerce/products/ANC-BLACK": CATALOG_ITEM_JSON,
            "/commerce/products/ANC BLACK/1": CATALOG_ITEM_JSON,  # url.path is decoded
            "/commerce/carts/cart-1": CART_JSON,
            "/commerce/orders/order-1": ORDER_JSON,
            "/commerce/payments/pay-1": PAYMENT_JSON,
        }

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=routes[request.url.path])

        async with external_port(handler) as (port, seen):
            assert isinstance(await port.get_product(7, "ANC-BLACK"), CatalogItem)
            await port.get_product(7, "ANC BLACK/1")  # identifiers are path-quoted
            assert isinstance(await port.get_cart(7, "cart-1"), CartSnapshot)
            assert isinstance(await port.get_order(7, "order-1"), OrderSnapshot)
            assert isinstance(await port.get_status(7, "pay-1"), PaymentSnapshot)

        assert [request.url.path for request in seen] == list(routes)
        # The wire carries the quoted form; url.path merely displays it decoded.
        assert seen[1].url.raw_path.decode().split("?")[0] == "/commerce/products/ANC%20BLACK%2F1"
        assert all(request.method == "GET" for request in seen)
        assert all(request.url.params.get("owner_user_id") == "7" for request in seen)

    async def test_update_cart_lines_sends_selections_and_idempotency_key(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/commerce/cart/update"
            return httpx.Response(200, json=CART_JSON)

        async with external_port(handler) as (port, seen):
            created = await port.update_cart_lines(
                7, None, None, [CartLineSelection(sku_id="ANC-BLACK", quantity=2)], "rk-1"
            )
            replaced = await port.update_cart_lines(
                7, "cart-1", 3, [CartLineSelection(sku_id="ANC-BLACK", quantity=1)], "rk-2"
            )

        assert isinstance(created, CartSnapshot) and created.cart_version == 3
        assert isinstance(replaced, CartSnapshot)
        assert body_of(seen[0]) == {
            "owner_user_id": 7,
            "lines": [{"sku_id": "ANC-BLACK", "quantity": 2}],
            "request_key": "rk-1",
        }
        assert body_of(seen[1]) == {
            "owner_user_id": 7,
            "cart_id": "cart-1",
            "expected_cart_version": 3,
            "lines": [{"sku_id": "ANC-BLACK", "quantity": 1}],
            "request_key": "rk-2",
        }

    async def test_checkout_order_payment_and_operation_lookup_paths_are_stable(self):
        routes: dict[str, object] = {
            "/commerce/checkout": QUOTE_JSON,
            "/commerce/orders": ORDER_JSON,
            "/commerce/payments": PAYMENT_JSON,
            "/commerce/payments/pay-1/sandbox-pay": {**PAYMENT_JSON, "status": "processing"},
        }

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/commerce/operations/rk-1":
                return httpx.Response(
                    200,
                    json={"kind": "order", "status": "completed", "result": ORDER_JSON},
                )
            if request.url.path == "/commerce/operations/rk-none":
                return null_body(request)
            return httpx.Response(200, json=routes[request.url.path])

        async with external_port(handler) as (port, seen):
            quote = await port.prepare_checkout(7, "cart-1", 3, "rk-q")
            order = await port.create_order(7, "quote-1", "rk-1")
            payment = await port.create_payment(7, "order-1", "rk-2")
            sandbox = await port.mock_pay(7, "pay-1", "rk-3")
            receipt = await port.lookup_operation(7, "rk-1")
            missing = await port.lookup_operation(7, "rk-none")

        assert isinstance(quote, QuoteSnapshot) and quote.quote_id == "quote-1"
        assert isinstance(order, OrderSnapshot) and order.status == "awaiting_payment"
        assert isinstance(payment, PaymentSnapshot)
        assert isinstance(sandbox, PaymentSnapshot) and sandbox.status == "processing"
        assert receipt == {"kind": "order", "status": "completed", "data": ORDER_JSON}
        assert missing is None
        assert [request.url.path for request in seen] == [
            "/commerce/checkout",
            "/commerce/orders",
            "/commerce/payments",
            "/commerce/payments/pay-1/sandbox-pay",
            "/commerce/operations/rk-1",
            "/commerce/operations/rk-none",
        ]
        assert all(body_of(request)["request_key"] for request in seen if request.method == "POST")

    async def test_requests_carry_bearer_token_and_bounded_timeout(self):
        async with external_port(lambda request: httpx.Response(200, json=CATALOG_ITEM_JSON)) as (
            port,
            seen,
        ):
            await port.get_product(7, "ANC-BLACK")

        request = seen[0]
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        assert request.headers["accept"] == "application/json"
        assert request.extensions.get("timeout") == {
            "connect": TIMEOUT_SECONDS,
            "read": TIMEOUT_SECONDS,
            "write": TIMEOUT_SECONDS,
            "pool": TIMEOUT_SECONDS,
        }

    async def test_no_token_configuration_omits_the_authorization_header(self):
        seen: list[httpx.Request] = []

        def record(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=CART_JSON)

        async with httpx.AsyncClient(transport=httpx.MockTransport(record)) as client:
            port = ExternalCommercePort(BASE_URL, client=client)
            await port.get_cart(7, "cart-1")
        assert "authorization" not in seen[0].headers


class TestProviderFailures:
    async def test_refusals_map_to_provider_rejected_with_provider_message(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.startswith("/commerce/products"):
                return httpx.Response(
                    404, json={"error": {"code": "not_found", "message": "商品不存在"}}
                )
            if request.url.path == "/commerce/search":
                return httpx.Response(403, text="forbidden")
            return httpx.Response(422, json={"detail": "参数无效"})

        async with external_port(handler) as (port, _):
            with pytest.raises(ProviderRejected, match="商品不存在"):
                await port.get_product(7, "MISSING")
            with pytest.raises(ProviderRejected, match="403"):
                await port.search_products(7, "耳机")
            with pytest.raises(ProviderRejected, match="参数无效"):
                await port.update_cart_lines(7, None, None, [], "rk-1")

    async def test_idempotency_key_mismatch_is_a_definitive_refusal(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                409,
                json={"error": {"code": "idempotency_mismatch", "message": "请求键已绑定其他参数"}},
            )

        async with external_port(handler) as (port, seen):
            with pytest.raises(ProviderRejected, match="请求键已绑定其他参数"):
                await port.create_order(7, "quote-1", "rk-replayed")

        assert body_of(seen[0])["request_key"] == "rk-replayed"

    async def test_timeouts_and_connection_failures_map_to_provider_unknown(self):
        def times_out(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("provider too slow")

        def refuses_connections(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        async with external_port(times_out) as (port, _):
            with pytest.raises(ProviderUnknown, match="超时"):
                await port.prepare_checkout(7, "cart-1", 3, "rk-q")

        async with external_port(refuses_connections) as (port, _):
            with pytest.raises(ProviderUnknown, match="暂不可用"):
                await port.get_cart(7, "cart-1")

    async def test_server_errors_and_untyped_bodies_map_to_provider_unknown(self):
        async with external_port(lambda request: httpx.Response(503, json={"error": "down"})) as (
            port,
            _,
        ):
            with pytest.raises(ProviderUnknown, match="HTTP 503"):
                await port.create_payment(7, "order-1", "rk-2")

        async with external_port(lambda request: httpx.Response(408, json={"error": "retry"})) as (
            port,
            _,
        ):
            with pytest.raises(ProviderUnknown, match="请按原请求核实"):
                await port.create_payment(7, "order-1", "rk-retry-408")

        async with external_port(lambda request: httpx.Response(429, json={"error": "retry"})) as (
            port,
            _,
        ):
            with pytest.raises(ProviderUnknown, match="请按原请求核实"):
                await port.create_payment(7, "order-1", "rk-retry-429")

        async with external_port(
            lambda request: httpx.Response(200, json={"items": "not-a-list"})
        ) as (
            port,
            _,
        ):
            with pytest.raises(ProviderUnknown, match="类型约定"):
                await port.search_products(7, "耳机")

        async with external_port(lambda request: httpx.Response(200, text="ok")) as (port, _):
            with pytest.raises(ProviderUnknown, match="不可解析"):
                await port.get_order(7, "order-1")

        async with external_port(
            lambda request: httpx.Response(200, json=["not", "an", "object"])
        ) as (
            port,
            _,
        ):
            with pytest.raises(ProviderUnknown, match="类型约定"):
                await port.lookup_operation(7, "rk-1")

    async def test_unavailable_provider_never_answers_with_local_substitutes(self):
        def unreachable(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        async with external_port(unreachable) as (port, _):
            assert port.mode == "external"
            assert not isinstance(port, MockCommercePort)
            with pytest.raises(ProviderUnknown):
                await port.search_products(7, "耳机")
            with pytest.raises(ProviderUnknown):
                await port.get_product(7, "ANC-BLACK")
            with pytest.raises(ProviderUnknown):
                await port.update_cart_lines(
                    7, None, None, [CartLineSelection(sku_id="ANC-BLACK", quantity=1)], "rk-1"
                )
            with pytest.raises(ProviderUnknown):
                await port.get_cart(7, "cart-1")
            with pytest.raises(ProviderUnknown):
                await port.prepare_checkout(7, "cart-1", 3, "rk-2")
            with pytest.raises(ProviderUnknown):
                await port.create_order(7, "quote-1", "rk-3")
            with pytest.raises(ProviderUnknown):
                await port.create_payment(7, "order-1", "rk-4")
            with pytest.raises(ProviderUnknown):
                await port.mock_pay(7, "pay-1", "rk-5")
            with pytest.raises(ProviderUnknown):
                await port.get_order(7, "order-1")
            with pytest.raises(ProviderUnknown):
                await port.get_status(7, "pay-1")
            with pytest.raises(ProviderUnknown):
                await port.lookup_operation(7, "rk-1")


class TestExplicitComposition:
    async def test_port_selection_follows_settings_mode_without_fallback(self):
        engine = create_async_engine("sqlite+aiosqlite://")
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        try:
            mock_config = Settings(_env_file=None)
            assert mock_config.COMMERCE_PROVIDER_MODE == "mock"
            assert isinstance(create_commerce_port(sessions, config=mock_config), MockCommercePort)

            external_config = Settings(
                _env_file=None,
                COMMERCE_PROVIDER_MODE="external",
                COMMERCE_PROVIDER_URL=BASE_URL,
                COMMERCE_PROVIDER_TOKEN=TOKEN,
            )
            selected = create_commerce_port(sessions, config=external_config)
            assert isinstance(selected, ExternalCommercePort)
            assert selected is create_commerce_port(sessions, config=external_config)
            # The provider stays the only fact source: the selected adapter is
            # external even when nothing has answered yet — no mock is consulted.
            assert not isinstance(selected, MockCommercePort)
        finally:
            await engine.dispose()

    def test_mode_is_an_explicit_choice_defaulting_to_mock_for_tests(self):
        assert Settings(_env_file=None).COMMERCE_PROVIDER_MODE == "mock"
        with pytest.raises(ValidationError, match="COMMERCE_PROVIDER_MODE"):
            Settings(_env_file=None, COMMERCE_PROVIDER_MODE="duck")

    def test_external_mode_requires_an_absolute_http_url(self):
        with pytest.raises(ValidationError, match="COMMERCE_PROVIDER_URL is required"):
            Settings(_env_file=None, COMMERCE_PROVIDER_MODE="external")
        with pytest.raises(ValidationError, match="http"):
            Settings(
                _env_file=None,
                COMMERCE_PROVIDER_MODE="external",
                COMMERCE_PROVIDER_URL="ftp://commerce.internal",
            )
        ok = Settings(
            _env_file=None,
            COMMERCE_PROVIDER_MODE="external",
            COMMERCE_PROVIDER_URL=BASE_URL,
        )
        assert ok.COMMERCE_PROVIDER_TIMEOUT_SECONDS == 5.0
