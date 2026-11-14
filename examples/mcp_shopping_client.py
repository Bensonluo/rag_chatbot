"""Talk to the Agent Store MCP server from any MCP host.

The commerce provider ships a Model Context Protocol surface (Streamable
HTTP). Read tools answer directly; write tools (update_cart → prepare_quote
→ create_order → sandbox_pay) carry an elicitation-style confirmation and
derive ``request_key`` structurally, so a declined or retried request can
never double-charge.

This example speaks the wire protocol with the official ``mcp`` SDK the way
an external host would — no in-process shortcuts.

Usage:
    # terminal 1 — start the store (empty token = auth disabled, local only)
    python -m app.commerce_provider.mcp_server --host 127.0.0.1 --port 8072 \
        --database /tmp/commerce-demo.sqlite3 --token ""

    # terminal 2 — run this client
    python examples/mcp_shopping_client.py [--url http://127.0.0.1:8072/mcp] [--token TOKEN]
"""

from __future__ import annotations

import argparse
import asyncio
import os

import httpx
from mcp import Client
from mcp.client.streamable_http import streamable_http_client


async def run(url: str, token: str | None) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else None
    # The transport is the SDK's Streamable HTTP client; headers ride a
    # pre-configured httpx.AsyncClient. Client() negotiates the protocol
    # version and tool schemas by itself.
    transport = streamable_http_client(
        url, http_client=httpx.AsyncClient(headers=headers) if headers else None
    )
    async with Client(transport) as client:
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
        page = await client.call_tool(
            "search_products", {"query": query, "owner_user_id": 1, "page_size": 3}
        )
        if page.is_error:
            raise SystemExit(f"search failed: {page.content}")
        items = page.structured_content["items"]
        print(
            f"search {query!r}: {len(items)} result(s), "
            f"total_estimate={page.structured_content['total_estimate']}"
        )
        for item in items:
            price = item["unit_price_minor"] / 100
            print(f"  {item['sku_id']}  {item['name']}  ¥{price:,.2f}")
        if not items:
            raise SystemExit("no results — try DEMO_QUERY=显示器 or another term")
        first = items[0]
        detail = await client.call_tool(
            "get_product", {"sku_id": first["sku_id"], "owner_user_id": 1}
        )
        product = detail.structured_content
        print(
            f"\nget_product {first['sku_id']}: {product['name']} "
            f"(¥{product['unit_price_minor'] / 100:,.2f}, "
            f"stock={product['available_quantity']}, "
            f"max_order_quantity={product['max_order_quantity']})"
        )
        print(
            "\nWrite tools (update_cart → prepare_quote → create_order → "
            "sandbox_pay) require an elicitation handler; hosts supply "
            "the confirmation UI. See docs/design/commerce-mcp-server.md."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8072/mcp")
    parser.add_argument("--token", default=os.environ.get("COMMERCE_PROVIDER_AUTH_TOKEN", ""))
    args = parser.parse_args()
    asyncio.run(run(args.url, args.token or None))


if __name__ == "__main__":
    main()
