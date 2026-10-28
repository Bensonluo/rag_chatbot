"""Wire-contract tests for the provider's internal HTTP API.

The paths, bodies, and status codes mirror the contract documented in
``app/services/commerce/external_port.py`` so ``ExternalCommercePort`` can
talk to this service verbatim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.commerce_provider.api import create_app
from app.commerce_provider.settings import CommerceProviderSettings

AUTH_HEADERS = {"Authorization": "Bearer test-token"}


def _search_one(client: TestClient) -> dict[str, Any]:
    response = client.post(
        "/commerce/search",
        headers=AUTH_HEADERS,
        json={"owner_user_id": 1, "query": "拓普 显示器", "sort": "price_asc", "page_size": 5},
    )
    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    assert body["items"]
    item: dict[str, Any] = body["items"][0]
    return item


def test_health_and_ready_need_no_token(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "database": "ok"}


def test_bearer_token_is_enforced_when_configured(client: TestClient) -> None:
    missing = client.post(
        "/commerce/search",
        json={"owner_user_id": 1, "query": "显示器", "sort": "relevance", "page_size": 3},
    )
    assert missing.status_code == 401
    assert missing.json()["error"]["code"] == "unauthorized"

    wrong = client.post(
        "/commerce/search",
        headers={"Authorization": "Bearer wrong-token"},
        json={"owner_user_id": 1, "query": "显示器", "sort": "relevance", "page_size": 3},
    )
    assert wrong.status_code == 401


def test_empty_token_disables_authentication(tmp_path: Path) -> None:
    app = create_app(
        CommerceProviderSettings(database_path=str(tmp_path / "open.sqlite3"), auth_token="")
    )
    with TestClient(app) as open_client:
        response = open_client.post(
            "/commerce/search",
            json={"owner_user_id": 1, "query": "显示器", "sort": "relevance", "page_size": 3},
        )
        assert response.status_code == 200


def test_search_and_get_product(client: TestClient) -> None:
    item = _search_one(client)
    assert set(item) == {
        "product_id",
        "sku_id",
        "name",
        "description",
        "attributes",
        "unit_price_minor",
        "currency",
        "available_quantity",
        "max_order_quantity",
    }
    fetched = client.get(
        f"/commerce/products/{item['sku_id']}", headers=AUTH_HEADERS, params={"owner_user_id": 1}
    )
    assert fetched.status_code == 200
    assert fetched.json() == item

    unknown = client.get(
        "/commerce/products/SKU-99999999X", headers=AUTH_HEADERS, params={"owner_user_id": 1}
    )
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "not_found"


def test_pagination_roundtrip_over_http(client: TestClient) -> None:
    first = client.post(
        "/commerce/search",
        headers=AUTH_HEADERS,
        json={"owner_user_id": 1, "query": "耳机", "sort": "relevance", "page_size": 4},
    ).json()
    assert first["next_page_token"]
    second = client.post(
        "/commerce/search",
        headers=AUTH_HEADERS,
        json={
            "owner_user_id": 1,
            "query": "耳机",
            "sort": "relevance",
            "page_size": 4,
            "page_token": first["next_page_token"],
        },
    )
    assert second.status_code == 200
    assert {item["sku_id"] for item in first["items"]}.isdisjoint(
        {item["sku_id"] for item in second.json()["items"]}
    )

    mismatch = client.post(
        "/commerce/search",
        headers=AUTH_HEADERS,
        json={
            "owner_user_id": 1,
            "query": "显示器",
            "sort": "relevance",
            "page_size": 4,
            "page_token": first["next_page_token"],
        },
    )
    assert mismatch.status_code == 422


def test_full_purchase_flow_over_http(client: TestClient) -> None:
    item = _search_one(client)
    quantity = min(2, item["max_order_quantity"])
    lines = [{"sku_id": item["sku_id"], "quantity": quantity}]

    cart = client.post(
        "/commerce/cart/update",
        headers=AUTH_HEADERS,
        json={"owner_user_id": 1, "lines": lines, "request_key": "cart-1"},
    )
    assert cart.status_code == 200
    cart_body = cart.json()
    assert cart_body["cart_version"] == 1
    assert cart_body["subtotal_minor"] == quantity * item["unit_price_minor"]

    fetched_cart = client.get(
        f"/commerce/carts/{cart_body['cart_id']}", headers=AUTH_HEADERS, params={"owner_user_id": 1}
    )
    assert fetched_cart.status_code == 200
    assert fetched_cart.json() == cart_body

    quote = client.post(
        "/commerce/checkout",
        headers=AUTH_HEADERS,
        json={
            "owner_user_id": 1,
            "cart_id": cart_body["cart_id"],
            "expected_cart_version": 1,
            "request_key": "quote-1",
        },
    )
    assert quote.status_code == 200
    quote_body = quote.json()
    assert quote_body["total_minor"] == cart_body["subtotal_minor"]

    order = client.post(
        "/commerce/orders",
        headers=AUTH_HEADERS,
        json={"owner_user_id": 1, "quote_id": quote_body["quote_id"], "request_key": "order-1"},
    )
    assert order.status_code == 200
    order_body = order.json()
    assert order_body["status"] == "awaiting_payment"

    payment = client.post(
        "/commerce/payments",
        headers=AUTH_HEADERS,
        json={"owner_user_id": 1, "order_id": order_body["order_id"], "request_key": "pay-1"},
    )
    assert payment.status_code == 200
    payment_body = payment.json()
    assert payment_body["status"] == "awaiting_payment"

    paid = client.post(
        f"/commerce/payments/{payment_body['payment_id']}/sandbox-pay",
        headers=AUTH_HEADERS,
        json={"owner_user_id": 1, "request_key": "pay-2"},
    )
    assert paid.status_code == 200
    assert paid.json()["status"] == "paid"

    assert (
        client.get(
            f"/commerce/orders/{order_body['order_id']}",
            headers=AUTH_HEADERS,
            params={"owner_user_id": 1},
        ).json()["status"]
        == "paid"
    )
    assert (
        client.get(
            f"/commerce/payments/{payment_body['payment_id']}",
            headers=AUTH_HEADERS,
            params={"owner_user_id": 1},
        ).json()["status"]
        == "paid"
    )

    # Stock visibly decreased for the purchased SKU.
    assert (
        client.get(
            f"/commerce/products/{item['sku_id']}",
            headers=AUTH_HEADERS,
            params={"owner_user_id": 1},
        ).json()["available_quantity"]
        == item["available_quantity"] - quantity
    )


def test_idempotency_and_conflict_over_http(client: TestClient) -> None:
    item = _search_one(client)
    payload = {
        "owner_user_id": 1,
        "lines": [{"sku_id": item["sku_id"], "quantity": 1}],
        "request_key": "cart-1",
    }
    first = client.post("/commerce/cart/update", headers=AUTH_HEADERS, json=payload)
    assert first.status_code == 200
    replay = client.post("/commerce/cart/update", headers=AUTH_HEADERS, json=payload)
    assert replay.status_code == 200
    assert replay.json() == first.json()

    conflict_payload = {
        "owner_user_id": 1,
        "lines": [{"sku_id": item["sku_id"], "quantity": 2}],
        "request_key": "cart-1",
    }
    conflict = client.post("/commerce/cart/update", headers=AUTH_HEADERS, json=conflict_payload)
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "conflict"


def test_operation_lookup_returns_receipt_or_null(client: TestClient) -> None:
    item = _search_one(client)
    client.post(
        "/commerce/cart/update",
        headers=AUTH_HEADERS,
        json={
            "owner_user_id": 1,
            "lines": [{"sku_id": item["sku_id"], "quantity": 1}],
            "request_key": "cart-1",
        },
    )
    found = client.get(
        "/commerce/operations/cart-1", headers=AUTH_HEADERS, params={"owner_user_id": 1}
    )
    assert found.status_code == 200
    receipt = found.json()
    assert receipt["kind"] == "update_cart_lines"
    assert receipt["status"] == "completed"
    assert receipt["result"]["cart_id"]

    missing = client.get(
        "/commerce/operations/nope", headers=AUTH_HEADERS, params={"owner_user_id": 1}
    )
    assert missing.status_code == 200
    assert missing.json() is None

    other_owner = client.get(
        "/commerce/operations/cart-1", headers=AUTH_HEADERS, params={"owner_user_id": 2}
    )
    assert other_owner.status_code == 200
    assert other_owner.json() is None


def test_owner_isolation_over_http(client: TestClient) -> None:
    item = _search_one(client)
    cart = client.post(
        "/commerce/cart/update",
        headers=AUTH_HEADERS,
        json={
            "owner_user_id": 1,
            "lines": [{"sku_id": item["sku_id"], "quantity": 1}],
            "request_key": "cart-1",
        },
    ).json()
    denied = client.get(
        f"/commerce/carts/{cart['cart_id']}", headers=AUTH_HEADERS, params={"owner_user_id": 2}
    )
    assert denied.status_code == 404


def test_validation_rejections_are_4xx_with_error_body(client: TestClient) -> None:
    stale = client.post(
        "/commerce/cart/update",
        headers=AUTH_HEADERS,
        json={
            "owner_user_id": 1,
            "cart_id": "c_missing",
            "expected_cart_version": 1,
            "lines": [],
            "request_key": "cart-1",
        },
    )
    assert stale.status_code == 404
    assert stale.json()["error"]["code"] == "not_found"


def test_fault_routes_hidden_when_disabled(tmp_path: Path) -> None:
    app = create_app(
        CommerceProviderSettings(
            database_path=str(tmp_path / "clean.sqlite3"),
            auth_token="test-token",
            faults_enabled=False,
        )
    )
    with TestClient(app) as clean_client:
        hidden = clean_client.post(
            "/testing/faults",
            headers=AUTH_HEADERS,
            json={"kind": "create_order", "mode": "reject"},
        )
        assert hidden.status_code == 404


def test_fault_reject_over_http(client: TestClient) -> None:
    item = _search_one(client)
    cart = client.post(
        "/commerce/cart/update",
        headers=AUTH_HEADERS,
        json={
            "owner_user_id": 1,
            "lines": [{"sku_id": item["sku_id"], "quantity": 1}],
            "request_key": "cart-1",
        },
    ).json()
    quote = client.post(
        "/commerce/checkout",
        headers=AUTH_HEADERS,
        json={
            "owner_user_id": 1,
            "cart_id": cart["cart_id"],
            "expected_cart_version": 1,
            "request_key": "quote-1",
        },
    ).json()

    planned = client.post(
        "/testing/faults", headers=AUTH_HEADERS, json={"kind": "create_order", "mode": "reject"}
    )
    assert planned.status_code == 200
    rejected = client.post(
        "/commerce/orders",
        headers=AUTH_HEADERS,
        json={"owner_user_id": 1, "quote_id": quote["quote_id"], "request_key": "order-1"},
    )
    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "rejected"
    assert (
        client.get(
            "/commerce/operations/order-1", headers=AUTH_HEADERS, params={"owner_user_id": 1}
        ).json()
        is None
    )

    retried = client.post(
        "/commerce/orders",
        headers=AUTH_HEADERS,
        json={"owner_user_id": 1, "quote_id": quote["quote_id"], "request_key": "order-1"},
    )
    assert retried.status_code == 200


def test_fault_unknown_after_commit_over_http(client: TestClient) -> None:
    item = _search_one(client)
    cart = client.post(
        "/commerce/cart/update",
        headers=AUTH_HEADERS,
        json={
            "owner_user_id": 1,
            "lines": [{"sku_id": item["sku_id"], "quantity": 1}],
            "request_key": "cart-1",
        },
    ).json()
    quote = client.post(
        "/commerce/checkout",
        headers=AUTH_HEADERS,
        json={
            "owner_user_id": 1,
            "cart_id": cart["cart_id"],
            "expected_cart_version": 1,
            "request_key": "quote-1",
        },
    ).json()
    order = client.post(
        "/commerce/orders",
        headers=AUTH_HEADERS,
        json={"owner_user_id": 1, "quote_id": quote["quote_id"], "request_key": "order-1"},
    ).json()

    client.post(
        "/testing/faults",
        headers=AUTH_HEADERS,
        json={"kind": "create_payment", "mode": "unknown_after_commit"},
    )
    unknown = client.post(
        "/commerce/payments",
        headers=AUTH_HEADERS,
        json={"owner_user_id": 1, "order_id": order["order_id"], "request_key": "pay-1"},
    )
    assert unknown.status_code == 504
    assert unknown.json()["error"]["code"] == "unknown"

    receipt = client.get(
        "/commerce/operations/pay-1", headers=AUTH_HEADERS, params={"owner_user_id": 1}
    ).json()
    assert receipt["status"] == "completed"
    payment_id = receipt["result"]["payment_id"]

    replay = client.post(
        "/commerce/payments",
        headers=AUTH_HEADERS,
        json={"owner_user_id": 1, "order_id": order["order_id"], "request_key": "pay-1"},
    )
    assert replay.status_code == 200
    assert replay.json()["payment_id"] == payment_id


def test_fault_processing_payment_and_settlement_over_http(client: TestClient) -> None:
    item = _search_one(client)
    cart = client.post(
        "/commerce/cart/update",
        headers=AUTH_HEADERS,
        json={
            "owner_user_id": 1,
            "lines": [{"sku_id": item["sku_id"], "quantity": 1}],
            "request_key": "cart-1",
        },
    ).json()
    quote = client.post(
        "/commerce/checkout",
        headers=AUTH_HEADERS,
        json={
            "owner_user_id": 1,
            "cart_id": cart["cart_id"],
            "expected_cart_version": 1,
            "request_key": "quote-1",
        },
    ).json()
    order = client.post(
        "/commerce/orders",
        headers=AUTH_HEADERS,
        json={"owner_user_id": 1, "quote_id": quote["quote_id"], "request_key": "order-1"},
    ).json()
    payment = client.post(
        "/commerce/payments",
        headers=AUTH_HEADERS,
        json={"owner_user_id": 1, "order_id": order["order_id"], "request_key": "pay-1"},
    ).json()

    client.post(
        "/testing/faults",
        headers=AUTH_HEADERS,
        json={"kind": "mock_pay", "mode": "processing_payment", "target": payment["payment_id"]},
    )
    processing = client.post(
        f"/commerce/payments/{payment['payment_id']}/sandbox-pay",
        headers=AUTH_HEADERS,
        json={"owner_user_id": 1, "request_key": "pay-2"},
    )
    assert processing.status_code == 200
    assert processing.json()["status"] == "processing"

    settled = client.post(
        "/testing/payments/settle",
        headers=AUTH_HEADERS,
        json={
            "owner_user_id": 1,
            "payment_id": payment["payment_id"],
            "paid": True,
        },
    )
    assert settled.status_code == 200
    assert settled.json()["status"] == "paid"
    assert (
        client.get(
            f"/commerce/orders/{order['order_id']}",
            headers=AUTH_HEADERS,
            params={"owner_user_id": 1},
        ).json()["status"]
        == "paid"
    )


def test_fault_listing_and_clearing(client: TestClient) -> None:
    client.post(
        "/testing/faults", headers=AUTH_HEADERS, json={"kind": "mock_pay", "mode": "reject"}
    )
    listed = client.get("/testing/faults", headers=AUTH_HEADERS)
    assert listed.status_code == 200
    assert listed.json()["rules"][0]["kind"] == "mock_pay"
    cleared = client.delete("/testing/faults", headers=AUTH_HEADERS)
    assert cleared.status_code == 200
    assert cleared.json()["cleared"] == 1
    assert client.get("/testing/faults", headers=AUTH_HEADERS).json()["rules"] == []
