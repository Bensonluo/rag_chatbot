"""get_recent_orders: caller-scoped order listing for the agent loop.

The tool takes no model-supplied parameters — the caller's identity is
injected server-side by ToolRegistry.execute, so cross-customer
listing is impossible by construction. Pinned here: per-user isolation,
newest-first ordering, the anonymous no-op, schema export, and that no
dialogue intent can ever route into this registry-only tool.
"""

from typing import Any

import pytest

from app.models.enums.intent import Intent
from app.services.dialogue.tools import (
    ToolDefinition,
    ToolRegistry,
    create_default_tool_registry,
    mock_get_recent_orders,
)


@pytest.fixture
def registry():
    return create_default_tool_registry()


class TestHandler:
    def test_lists_caller_orders_newest_first(self):
        result = mock_get_recent_orders({"user_id": 1})

        assert [o["order_id"] for o in result["orders"]] == ["ORD1001", "ORD1002"]
        assert result["count"] == 2
        first = result["orders"][0]
        assert first["status"] == "已发货"
        assert first["total_amount"] == 598.00
        assert "created_at" in first and "items" in first

    def test_per_user_isolation(self):
        result = mock_get_recent_orders({"user_id": 2})
        assert [o["order_id"] for o in result["orders"]] == ["ORD2001"]

    def test_user_with_no_orders(self):
        result = mock_get_recent_orders({"user_id": 99})
        assert result == {"orders": [], "count": 0}

    def test_anonymous_returns_empty_with_message(self):
        result = mock_get_recent_orders({})
        assert result["orders"] == []
        assert result["count"] == 0
        assert "未登录" in result["message"]

    async def test_registry_executes_with_injected_identity(self, registry):
        result = await registry.execute("recent_orders", {}, user_id=1)
        assert result.success
        assert result.data["count"] == 2

    async def test_forged_user_id_in_args_cannot_override_identity(self, registry):
        # The agent loop passes LLM-generated args verbatim — a prompt
        # injection that smuggles user_id into them must not widen the
        # query to another customer's orders.
        result = await registry.execute("recent_orders", {"user_id": 2}, user_id=1)
        assert result.success
        assert [o["order_id"] for o in result.data["orders"]] == ["ORD1001", "ORD1002"]

    async def test_forged_user_id_cannot_read_foreign_orders(self, registry):
        # Same escalation via an order-scoped tool: model supplies both
        # the target order and a matching forged identity — the server's
        # user_id must win the ownership check.
        result = await registry.execute(
            "query_order", {"order_id": "ORD2001", "user_id": 2}, user_id=1
        )
        assert result.success is False
        assert "无权" in result.message

    async def test_anonymous_args_identity_is_stripped(self):
        seen: list[dict[str, Any]] = []

        def spying_handler(args: dict[str, Any]) -> dict[str, Any]:
            seen.append(args)
            return {"ok": True}

        bare = ToolRegistry()
        bare.register(
            ToolDefinition(
                name="spy",
                intent="query_order",
                description="spy",
                required_slots=[],
                handler=spying_handler,
            )
        )
        result = await bare.execute("query_order", {"order_id": "ORD1001", "user_id": 1})
        assert result.success is True
        # Demo-mode anonymous pass-through, but the handler must not see
        # the caller-supplied identity (it would otherwise act authenticated).
        assert "user_id" not in seen[0]


class TestSchemaExport:
    def test_exported_with_no_model_supplied_parameters(self, registry):
        schemas = {s["function"]["name"]: s["function"] for s in registry.to_function_schemas()}
        assert "get_recent_orders" in schemas
        parameters = schemas["get_recent_orders"]["parameters"]
        # Identity is injected server-side: the model cannot name a user.
        assert parameters["properties"] == {}
        assert parameters["required"] == []


class TestIntentRoutingIsolation:
    def test_no_dialogue_intent_routes_into_the_tool(self, registry):
        # "recent_orders" is a registry key only — every real Intent enum
        # value must resolve to a different tool (or none), never the
        # order-listing tool, so the slot pipeline can't reach it.
        tool = registry.get_tool_for_intent("recent_orders")
        for intent in Intent:
            if registry.get_tool_for_intent(intent.value) is tool:
                pytest.fail(f"intent {intent.value} routes into get_recent_orders")
