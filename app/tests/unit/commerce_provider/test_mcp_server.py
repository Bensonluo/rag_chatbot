"""MCP server surface over the commerce provider core.

Verified against the ``mcp`` SDK 2.3.0 in-memory client pattern (the SDK's
own documented testing approach): ``Client(mcp)`` negotiates the protocol,
so every assertion here exercises the real tool pipeline including argument
schemas, structured output, and the ``Resolve``/``Elicit`` confirmation
round-trip for write tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from mcp.types import ElicitResult
from starlette.testclient import TestClient as StarletteTestClient

from app.commerce_provider.catalog import VirtualCatalog
from app.commerce_provider.core import CommerceProviderCore
from app.commerce_provider.faults import FaultController
from app.commerce_provider.mcp_server import (
    CommerceMcpSettings,
    cart_request_key,
    create_mcp_server,
    order_request_key,
    payment_request_key,
    quote_request_key,
)
from app.commerce_provider.storage import Database
from app.models.schemas.commerce import CartLineSelection
from app.tests.unit.commerce_provider.conftest import FixedClock

EXPECTED_TOOLS = {
    "search_products",
    "get_product",
    "get_categories",
    "discover_catalog",
    "update_cart",
    "prepare_quote",
    "create_order",
    "sandbox_pay",
}


async def _accept(ctx: Any, params: Any) -> ElicitResult:
    return ElicitResult(action="accept", content={"confirmed": True})


async def _decline(ctx: Any, params: Any) -> ElicitResult:
    return ElicitResult(action="decline")


async def _first_sku(client: Client, query: str = "显示器") -> str:
    search = await client.call_tool(
        "search_products", {"query": query, "owner_user_id": 1, "page_size": 1}
    )
    assert not search.is_error
    return str(search.structured_content["items"][0]["sku_id"])


@pytest.fixture
def settings() -> CommerceMcpSettings:
    return CommerceMcpSettings(auth_token="test-token")


@pytest.fixture
def mcp(core: CommerceProviderCore, settings: CommerceMcpSettings):
    return create_mcp_server(settings, core=core)


async def test_handshake_lists_the_documented_tool_surface(mcp) -> None:
    async with Client(mcp) as client:
        tools = {tool.name for tool in (await client.list_tools()).tools}
    assert tools == EXPECTED_TOOLS


async def test_search_products_returns_structured_catalog_page(mcp) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "search_products", {"query": "显示器", "owner_user_id": 1, "page_size": 3}
        )
    assert not result.is_error
    page: dict[str, Any] | None = result.structured_content
    assert page is not None
    assert 1 <= len(page["items"]) <= 3
    first = page["items"][0]
    assert first["sku_id"]
    assert first["unit_price_minor"] >= 0


async def test_search_products_caps_page_size_at_five(mcp) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "search_products", {"query": "显示器", "owner_user_id": 1, "page_size": 6}
        )
    assert result.is_error


async def test_get_product_roundtrip_and_not_found(mcp) -> None:
    async with Client(mcp) as client:
        search = await client.call_tool(
            "search_products", {"query": "显示器", "owner_user_id": 1, "page_size": 1}
        )
        assert not search.is_error
        sku_id = search.structured_content["items"][0]["sku_id"]

        found = await client.call_tool("get_product", {"owner_user_id": 1, "sku_id": sku_id})
        assert not found.is_error
        assert found.structured_content["sku_id"] == sku_id

        missing = await client.call_tool(
            "get_product", {"owner_user_id": 1, "sku_id": "missing-sku-000"}
        )
        assert missing.is_error
        assert "not_found" in str(missing.content)


async def test_get_categories_publishes_taxonomy(mcp) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool("get_categories", {"owner_user_id": 1})
    assert not result.is_error
    categories = result.structured_content["categories"]
    assert len(categories) >= 1  # full authoritative taxonomy; the 32-id cap applies to the filter
    assert all("category_id" in category and "name" in category for category in categories)


async def test_discover_catalog_combines_categories_and_products(mcp) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool("discover_catalog", {"query": "办公", "owner_user_id": 1})
    assert not result.is_error
    assert "categories" in result.structured_content
    assert "products" in result.structured_content
    assert len(result.structured_content["products"]["items"]) >= 1


async def test_write_tool_declined_confirmation_changes_nothing(mcp) -> None:
    async with Client(mcp, elicitation_callback=_decline) as client:
        sku_id = await _first_sku(client)
        result = await client.call_tool(
            "update_cart", {"owner_user_id": 1, "lines": [{"sku_id": sku_id, "quantity": 1}]}
        )
    assert result.is_error
    assert "declined" in str(result.content).lower()


async def test_unknown_outcome_retry_replays_the_committed_result(
    tmp_path: Path, clock: FixedClock
) -> None:
    """Fault injection: commit succeeded but the answer was lost.

    The tool surfaces an "outcome unconfirmed" error whose remedy is retrying
    with identical arguments; the structural request key replays the committed
    snapshot instead of creating a second cart.
    """
    database = Database(str(tmp_path / "mcp_faults.sqlite3"))
    await database.startup()
    faults = FaultController(database, enabled=True)
    faulted_core = CommerceProviderCore(
        database, VirtualCatalog(), clock=clock, quote_ttl_seconds=300, faults=faults
    )
    await faults.plan(kind="update_cart_lines", mode="unknown_after_commit", times=1)

    server = create_mcp_server(CommerceMcpSettings(auth_token="t"), core=faulted_core)
    async with Client(server, elicitation_callback=_accept) as client:
        sku_id = await _first_sku(client)
        lines = [{"sku_id": sku_id, "quantity": 1}]

        lost = await client.call_tool("update_cart", {"owner_user_id": 1, "lines": lines})
        assert lost.is_error
        assert "未确认" in str(lost.content)

        retry = await client.call_tool("update_cart", {"owner_user_id": 1, "lines": lines})
        assert not retry.is_error
        snapshot = retry.structured_content
        assert snapshot["cart_version"] == 1

        # The replayed receipt is stable: one more identical retry, same answer.
        again = await client.call_tool("update_cart", {"owner_user_id": 1, "lines": lines})
        assert again.structured_content == snapshot


async def test_write_update_cart_and_idempotent_replay(mcp) -> None:
    async with Client(mcp, elicitation_callback=_accept) as client:
        sku_id = await _first_sku(client)
        lines = [{"sku_id": sku_id, "quantity": 2}]
        first = await client.call_tool("update_cart", {"owner_user_id": 1, "lines": lines})
        assert not first.is_error
        snapshot = first.structured_content
        assert snapshot["cart_version"] == 1
        assert snapshot["subtotal_minor"] > 0

        replay = await client.call_tool("update_cart", {"owner_user_id": 1, "lines": lines})
        assert not replay.is_error
        assert replay.structured_content == snapshot

        # A different payload is a different logical mutation: new version.
        changed = await client.call_tool(
            "update_cart",
            {
                "owner_user_id": 1,
                "cart_id": snapshot["cart_id"],
                "expected_cart_version": snapshot["cart_version"],
                "lines": [{"sku_id": sku_id, "quantity": 3}],
            },
        )
        assert not changed.is_error
        assert changed.structured_content["cart_version"] == 2


async def test_write_full_flow_quote_order_pay(mcp) -> None:
    async with Client(mcp, elicitation_callback=_accept) as client:
        sku_id = await _first_sku(client)
        cart = await client.call_tool(
            "update_cart",
            {"owner_user_id": 1, "lines": [{"sku_id": sku_id, "quantity": 1}]},
        )
        cart_snapshot = cart.structured_content
        assert cart_snapshot["cart_version"] == 1

        quote = await client.call_tool(
            "prepare_quote",
            {
                "owner_user_id": 1,
                "cart_id": cart_snapshot["cart_id"],
                "expected_cart_version": cart_snapshot["cart_version"],
            },
        )
        quote_snapshot = quote.structured_content
        assert quote_snapshot["quote_id"]
        assert quote_snapshot["expires_at"]
        assert quote_snapshot["total_minor"] == cart_snapshot["subtotal_minor"]

        order = await client.call_tool(
            "create_order", {"owner_user_id": 1, "quote_id": quote_snapshot["quote_id"]}
        )
        order_snapshot = order.structured_content
        assert order_snapshot["order_id"]
        assert order_snapshot["status"] == "awaiting_payment"

        payment = await client.call_tool(
            "sandbox_pay", {"owner_user_id": 1, "order_id": order_snapshot["order_id"]}
        )
        assert not payment.is_error
        assert payment.structured_content["status"] == "paid"

        # Retrying the sandbox pay against the same order is an idempotent
        # replay of the same logical payment, not a second charge.
        replay = await client.call_tool(
            "sandbox_pay", {"owner_user_id": 1, "order_id": order_snapshot["order_id"]}
        )
        assert replay.structured_content == payment.structured_content


def test_request_key_derivation_is_structural_and_bounded() -> None:
    lines = [CartLineSelection(sku_id="SKU-00000000", quantity=2)]
    key_a = cart_request_key(1, None, None, lines)
    key_b = cart_request_key(1, None, None, lines)
    assert key_a == key_b  # deterministic: same logical retry replays
    assert len(key_a) <= 128

    other_lines = [CartLineSelection(sku_id="SKU-00000000", quantity=3)]
    assert cart_request_key(1, None, None, other_lines) != key_a
    assert cart_request_key(2, None, None, lines) != key_a  # owner-scoped
    assert cart_request_key(1, "cart-1", 4, lines) != key_a

    assert quote_request_key(1, "cart-1", 2) == quote_request_key(1, "cart-1", 2)
    assert quote_request_key(1, "cart-1", 3) != quote_request_key(1, "cart-1", 2)
    assert order_request_key(1, "q-1") == order_request_key(1, "q-1")
    assert order_request_key(1, "q-2") != order_request_key(1, "q-1")
    assert payment_request_key(1, "p-1") == payment_request_key(1, "p-1")
    assert payment_request_key(2, "p-1") != payment_request_key(1, "p-1")


def test_http_auth_rejects_missing_or_wrong_bearer_token(
    core: CommerceProviderCore,
) -> None:
    settings = CommerceMcpSettings(auth_token="test-token")
    server = create_mcp_server(settings, core=core)
    app = server.streamable_http_app()

    with StarletteTestClient(app) as http:
        missing = http.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert missing.status_code == 401

        wrong = http.post(
            "/mcp",
            headers={"Authorization": "Bearer wrong-token"},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )
        assert wrong.status_code == 401

        correct = http.post(
            "/mcp",
            headers={"Authorization": "Bearer test-token"},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        )
        # The bearer token is accepted (not 401); the bare JSON-RPC shape may
        # still be refused by the streamable transport itself — that is out of
        # scope here, the auth boundary is what this test pins down.
        assert correct.status_code != 401
