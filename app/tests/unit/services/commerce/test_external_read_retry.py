"""Bounded read retry for the external commerce provider adapter.

Contract:
- Read-only provider calls (search, categories, product, cart read, order
  read, payment status, operation lookup) survive transient ProviderUnknown
  failures (timeouts, connection errors, 5xx, undecodable bodies) with at
  most 2 retries: capped exponential backoff with jitter, total worst-case
  wall clock bounded by timeout_seconds * 3.
- ProviderRejected is definitive — never retried.
- Write operations (cart update, checkout, order, payment, sandbox-pay,
  dynamic category mint) are NEVER auto-retried: the call may have committed,
  so callers reconcile through the original request_key instead.
"""

from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest

from app.services.commerce.external_port import ExternalCommercePort
from app.services.commerce.port import ProviderRejected, ProviderUnknown

VALID_ITEM = {
    "product_id": "headphones-anc",
    "sku_id": "ANC-01",
    "name": "轻音降噪耳机 ANC-01",
    "description": "模拟商品：无线连接、主动降噪。",
    "attributes": {"color": "黑色"},
    "unit_price_minor": 29900,
    "currency": "CNY",
    "available_quantity": 5,
    "max_order_quantity": 3,
}
VALID_PAGE = {"items": [VALID_ITEM], "next_page_token": None, "total_estimate": 1}


class FlakyTransport:
    """Returns provider failures for the first N requests, then a valid page."""

    def __init__(self, *, failures: int, status: int = 500) -> None:
        self.failures = failures
        self.status = status
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if len(self.requests) <= self.failures:
            return httpx.Response(
                self.status, json={"error": {"code": "unavailable", "message": "暂不可用"}}
            )
        return httpx.Response(200, json=VALID_PAGE)


class SleepRecorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def _port(transport: httpx.MockTransport) -> tuple[ExternalCommercePort, SleepRecorder]:
    port = ExternalCommercePort(
        "http://provider.invalid",
        client=httpx.AsyncClient(transport=transport),
    )
    sleeper = SleepRecorder()
    port._sleep = sleeper  # test seam: no real waiting
    return port, sleeper


async def test_transient_500_on_search_retries_and_recovers() -> None:
    transport = FlakyTransport(failures=1)
    port, sleeper = _port(httpx.MockTransport(transport))
    page = await port.search_products(1, "降噪耳机")
    assert page.items[0].sku_id == "ANC-01"
    assert len(transport.requests) == 2  # 1 failure + 1 retry
    assert len(sleeper.delays) == 1
    await port.aclose()


async def test_persistent_500_fails_after_exactly_three_attempts() -> None:
    transport = FlakyTransport(failures=99)
    port, sleeper = _port(httpx.MockTransport(transport))
    with pytest.raises(ProviderUnknown):
        await port.search_products(1, "降噪耳机")
    assert len(transport.requests) == 3  # 1 initial + 2 bounded retries
    assert len(sleeper.delays) == 2
    # Capped exponential backoff with jitter: first delay in [0.05, 0.075],
    # second in [0.10, 0.15] — bounded, never exploding.
    assert 0.05 <= sleeper.delays[0] <= 0.075
    assert 0.10 <= sleeper.delays[1] <= 0.15
    await port.aclose()


async def test_connection_error_on_read_retries() -> None:
    calls: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(200, json=VALID_PAGE)

    port, sleeper = _port(httpx.MockTransport(handler))
    page = await port.search_products(1, "降噪耳机")
    assert page.items and len(calls) == 2 and len(sleeper.delays) == 1
    await port.aclose()


async def test_provider_rejection_is_never_retried() -> None:
    transport = FlakyTransport(failures=99, status=400)
    port, sleeper = _port(httpx.MockTransport(transport))
    with pytest.raises(ProviderRejected):
        await port.search_products(1, "降噪耳机")
    assert len(transport.requests) == 1 and sleeper.delays == []
    await port.aclose()


async def test_other_read_endpoints_retry_transient_failures() -> None:
    payment = {
        "payment_id": "pay-1",
        "order_id": "order-1",
        "total_minor": 100,
        "currency": "CNY",
        "status": "paid",
    }
    calls: list[httpx.Request] = []

    async def payment_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) <= 2:
            return httpx.Response(503, json={"error": {"code": "x", "message": "慢"}})
        return httpx.Response(200, json=payment)

    port, _ = _port(httpx.MockTransport(payment_handler))
    snapshot = await port.get_status(1, "pay-1")
    assert snapshot.status == "paid"
    assert len(calls) == 3  # read path retried to recovery
    await port.aclose()


async def test_within_retry_budget_boundaries() -> None:
    fits = ExternalCommercePort._within_retry_budget
    assert fits(elapsed=0.0, timeout_seconds=5.0, next_delay=0.05) is True
    assert fits(elapsed=9.9, timeout_seconds=5.0, next_delay=0.05) is True
    assert fits(elapsed=10.0, timeout_seconds=5.0, next_delay=0.05) is False


async def test_write_operations_are_never_auto_retried() -> None:
    from app.models.schemas.commerce import CartLineSelection

    writes: list[tuple[str, Callable[[ExternalCommercePort], Awaitable[Any]]]] = [
        (
            "update_cart_lines",
            lambda port: port.update_cart_lines(
                1,
                None,
                None,
                [CartLineSelection(sku_id="ANC-01", quantity=1)],
                request_key="w-1",
            ),
        ),
        (
            "prepare_checkout",
            lambda port: port.prepare_checkout(1, "cart-1", 1, request_key="w-2"),
        ),
        ("create_order", lambda port: port.create_order(1, "quote-1", "w-3")),
        ("create_payment", lambda port: port.create_payment(1, "order-1", "w-4")),
        ("mock_pay", lambda port: port.mock_pay(1, "pay-1", "w-5")),
        (
            "propose_category",
            lambda port: port.propose_category(
                1,
                request_key="w-6",
                name="糖果巧克力",
                description="可可制品。",
                domain="food",
                attributes=[("类型", ["黑巧"])],
                variant_key="规格",
                variant_values=["45g", "100g", "250g", "礼盒"],
                base_price_minor=1500,
            ),
        ),
    ]
    for name, call in writes:
        transport = FlakyTransport(failures=99)
        port, sleeper = _port(httpx.MockTransport(transport))
        with pytest.raises(ProviderUnknown, match="未确认"):
            await call(port)
        assert len(transport.requests) == 1, f"{name} issued more than one request"
        assert sleeper.delays == [], f"{name} slept between attempts"
        await port.aclose()
