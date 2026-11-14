"""Talk to the Agent Store MCP server from any MCP host.

The commerce provider ships a Model Context Protocol surface (Streamable
HTTP). Read tools answer directly; write tools (update_cart → prepare_quote
→ create_order → sandbox_pay) ask for confirmation mid-call through MCP
elicitation — the caller cannot forge or skip the gate — and derive their
``request_key`` structurally, so a declined or retried request can never
double-charge.

This example speaks the wire protocol with the official ``mcp`` SDK the way
an external host would — no in-process shortcuts — and walks the complete
purchase journey, including a deliberate duplicate ``create_order`` that
replays the same order instead of creating a second one.

Usage:
    # terminal 1 — start the store (empty token = auth disabled, local only)
    python -m app.commerce_provider.mcp_server --host 127.0.0.1 --port 8072 \
        --database /tmp/commerce-demo.sqlite3 --token ""

    # terminal 2 — run this client (answers confirmation prompts by hand,
    # or pass --yes to auto-approve every gate for scripted demos)
    python examples/mcp_shopping_client.py [--url http://127.0.0.1:8072/mcp] \
        [--token TOKEN] [--yes]
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

import httpx2
from mcp import Client
from mcp.client.session import ClientRequestContext, ElicitationFnT
from mcp.client.streamable_http import streamable_http_client
from mcp.types import (
    ElicitRequestFormParams,
    ElicitRequestURLParams,
    ElicitResult,
    ErrorData,
)


def _answer(auto_yes: bool) -> ElicitationFnT:
    """Elicitation callback: the server asks, the human (or --yes) answers."""

    async def confirm(
        context: ClientRequestContext,  # noqa: ARG001 — protocol-mandated name
        params: ElicitRequestFormParams | ElicitRequestURLParams,
    ) -> ElicitResult | ErrorData:
        print(f"\n  ⟳ server asks: {params.message}")
        if auto_yes:
            print("  ✓ auto-approved (--yes)")
            return ElicitResult(action="accept", content={"confirmed": True})
        try:
            answer = input("  approve? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer in ("y", "yes"):
            return ElicitResult(action="accept", content={"confirmed": True})
        print("  ✗ declined — the tool aborts without touching the core")
        return ElicitResult(action="decline")

    return confirm


class Stop(Exception):
    """Demo stop condition — declined gate, failed call, failed check."""


def _text(content: object) -> str:
    """First text block of a tool result's content, for readable stops."""
    if isinstance(content, list) and content and hasattr(content[0], "text"):
        return str(content[0].text)
    return str(content)


async def _call(client: Client, name: str, args: dict[str, object]) -> Any:
    """Call a tool; a refused call (e.g. a declined confirmation) stops the demo."""
    result = await client.call_tool(name, args)
    if result.is_error:
        raise Stop(f"{name} refused: {_text(result.content)}")
    return result.structured_content


async def run(url: str, token: str | None, auto_yes: bool) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else None
    # The transport is the SDK's Streamable HTTP client; headers ride a
    # pre-configured httpx2.AsyncClient (the SDK's httpx fork). Client()
    # negotiates the protocol version and tool schemas by itself, and the
    # elicitation callback answers every mid-call confirmation request.
    transport = streamable_http_client(
        url, http_client=httpx2.AsyncClient(headers=headers) if headers else None
    )
    # Stops (declined gates, failed calls) are raised inside the session as
    # Stop and converted to SystemExit after the session closes cleanly —
    # exceptions raised across the SDK's task group get re-wrapped otherwise.
    stop: Stop | None = None
    async with Client(transport, elicitation_callback=_answer(auto_yes)) as client:
        try:
            listed = (await client.list_tools()).tools
            print(f"connected: {len(listed)} tools")
            for tool in sorted(listed, key=lambda t: t.name):
                hints = tool.annotations
                flags = (
                    " ".join(
                        f"{flag}={getattr(hints, flag)}"
                        for flag in ("read_only_hint", "destructive_hint", "idempotent_hint")
                        if getattr(hints, flag, None) is not None
                    )
                    if hints is not None
                    else ""
                )
                print(f"  {tool.name:<18} {flags or '(no annotations)'}")
            print()

            query = os.environ.get("DEMO_QUERY", "无线耳机")
            page = await _call(
                client, "search_products", {"query": query, "owner_user_id": 1, "page_size": 3}
            )
            items = page["items"]
            print(
                f"search {query!r}: {len(items)} result(s), total_estimate={page['total_estimate']}"
            )
            for item in items:
                price = item["unit_price_minor"] / 100
                print(f"  {item['sku_id']}  {item['name']}  ¥{price:,.2f}")
            if not items:
                raise Stop("no results — try DEMO_QUERY=显示器 or another term")
            first = items[0]
            product = await _call(
                client, "get_product", {"sku_id": first["sku_id"], "owner_user_id": 1}
            )
            print(
                f"\nget_product {first['sku_id']}: {product['name']} "
                f"(¥{product['unit_price_minor'] / 100:,.2f}, "
                f"stock={product['available_quantity']}, "
                f"max_order_quantity={product['max_order_quantity']})"
            )

            # --- full purchase journey: every write asks for confirmation -----
            print("\n=== purchase journey — each write needs a live confirmation ===")
            cart = await _call(
                client,
                "update_cart",
                {
                    "owner_user_id": 1,
                    "lines": [{"sku_id": first["sku_id"], "quantity": 2}],
                },
            )
            print(
                f"cart {cart['cart_id']} v{cart['cart_version']} "
                f"subtotal ¥{cart['subtotal_minor'] / 100:,.2f}"
            )

            quote = await _call(
                client,
                "prepare_quote",
                {
                    "owner_user_id": 1,
                    "cart_id": cart["cart_id"],
                    "expected_cart_version": cart["cart_version"],
                },
            )
            print(
                f"quote {quote['quote_id']} total ¥{quote['total_minor'] / 100:,.2f} "
                f"(expires {quote['expires_at']})"
            )

            order = await _call(
                client, "create_order", {"owner_user_id": 1, "quote_id": quote["quote_id"]}
            )
            print(f"order {order['order_id']} status={order['status']}")

            # Deliberate duplicate: identical logical operation → structural
            # request key → the idempotency ledger replays the original order.
            replay = await _call(
                client, "create_order", {"owner_user_id": 1, "quote_id": quote["quote_id"]}
            )
            replayed = replay["order_id"] == order["order_id"]
            print(
                f"retry create_order → {replay['order_id']} "
                + (
                    "(same order — replayed, nothing doubled)"
                    if replayed
                    else "(DIFFERENT ORDER — idempotency broken!)"
                )
            )

            payment = await _call(
                client, "sandbox_pay", {"owner_user_id": 1, "order_id": order["order_id"]}
            )
            print(f"payment {payment['payment_id']} status={payment['status']}")

            if not replayed or payment["status"] != "paid":
                raise Stop("journey verification failed")
            print("\njourney verified: quote → order → replay-safe → paid (sandbox)")
        except Stop as exc:
            stop = exc
    if stop is not None:
        raise SystemExit(str(stop))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8072/mcp")
    parser.add_argument("--token", default=os.environ.get("COMMERCE_PROVIDER_AUTH_TOKEN", ""))
    parser.add_argument("--yes", action="store_true", help="auto-approve every confirmation gate")
    args = parser.parse_args()
    asyncio.run(run(args.url, args.token or None, args.yes))


if __name__ == "__main__":
    main()
