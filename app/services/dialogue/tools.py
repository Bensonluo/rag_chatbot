"""
Tool definitions and registry for Function Calling.

Provides mock tool handlers for customer service operations
(refund, order query, shipping tracking, etc.) that simulate
real API responses. Mock orders record their owner so tools
enforce per-user authorization (no cross-customer access), and
irreversible actions (refund / return) are flagged to require
explicit user confirmation before execution.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.models.enums.intent import TASK_INTENTS  # noqa: F401 (re-exported)

# Refunds above this amount skip auto-processing and escalate to a human.
REFUND_AUTO_THRESHOLD = 1000.00


@dataclass
class ToolDefinition:
    """Definition of a callable tool."""

    name: str
    intent: str
    description: str
    required_slots: list[str]
    handler: Callable[[dict], dict]
    # Irreversible actions ask the user for explicit confirmation before
    # the handler runs (Sierra/Fin-style confirmation gate).
    requires_confirmation: bool = False
    # JSON Schema for the function-calling API (LLM agent mode). When
    # None the tool is only reachable via the slot-filling pipeline.
    parameters_schema: dict[str, Any] | None = None


@dataclass
class ToolResult:
    """Result from tool execution."""

    success: bool
    data: dict
    message: str = ""


class ToolRegistry:
    """Registry of available tools for Function Calling."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.intent] = tool

    def get_tool_for_intent(self, intent: str) -> ToolDefinition | None:
        return self._tools.get(intent)

    def get_tool_by_name(self, name: str) -> ToolDefinition | None:
        """Look up a tool by its function name (LLM tool_call target)."""
        for tool in self._tools.values():
            if tool.name == name:
                return tool
        return None

    def to_function_schemas(self) -> list[dict[str, Any]]:
        """Export agent-reachable tools as OpenAI function schemas.

        Only tools carrying a ``parameters_schema`` are exported — a
        tool without one stays reachable via the slot-filling pipeline
        only, keeping agent scope explicit.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema,
                },
            }
            for tool in self._tools.values()
            if tool.parameters_schema is not None
        ]

    async def execute(
        self,
        intent: str,
        args: dict,
        user_id: int | None = None,
    ) -> ToolResult:
        """Run the handler for ``intent`` with per-user authorization.

        The caller's ``user_id`` (from the authenticated session, not the
        request body) is injected into the handler args so order-scoped
        tools can verify ownership.
        """
        tool = self._tools.get(intent)
        if not tool:
            return ToolResult(success=False, data={}, message=f"No tool for intent: {intent}")
        call_args = dict(args)
        if user_id is not None:
            call_args.setdefault("user_id", user_id)
        try:
            result = tool.handler(call_args)
            return ToolResult(success=True, data=result, message="OK")
        except Exception as e:
            return ToolResult(success=False, data={}, message=str(e))


# ── Demo order store ───────────────────────────────────────────────────────
#
# Keys are order IDs; each order records its owner so tools can enforce
# per-user authorization. Anonymous callers (no user_id) pass through in
# demo mode; once real auth lands, every request carries a user_id.

MOCK_ORDERS: dict[str, dict] = {
    "ORD1001": {
        "user_id": 1,
        "status": "已发货",
        "items": ["商品A x1", "商品B x2"],
        "total_amount": 598.00,
        "created_at": "2024-12-15 10:30:00",
        "estimated_delivery": "2024-12-20",
    },
    "ORD1002": {
        "user_id": 1,
        "status": "已完成",
        "items": ["商品C x1"],
        "total_amount": 199.00,
        "created_at": "2024-12-10 09:00:00",
        "estimated_delivery": "已送达",
    },
    "ORD2001": {
        "user_id": 2,
        "status": "已发货",
        "items": ["商品D x1"],
        "total_amount": 899.00,
        "created_at": "2024-12-16 14:20:00",
        "estimated_delivery": "2024-12-21",
    },
    "ORD3001": {
        "user_id": 3,
        "status": "已签收",
        "items": ["商品E x1 (高价值)"],
        "total_amount": 1299.00,
        "created_at": "2024-12-12 11:00:00",
        "estimated_delivery": "已送达",
    },
}


def _owned_order(args: dict) -> dict:
    """Fetch an order while enforcing ownership.

    Raises:
        LookupError: order does not exist.
        PermissionError: order belongs to another user (reported as
            "not found or no access" so existence is not leaked).
    """
    order_id = str(args.get("order_id", ""))
    order = MOCK_ORDERS.get(order_id)
    if order is None:
        raise LookupError(f"订单 {order_id} 不存在，请确认订单号是否正确")
    caller_id = args.get("user_id")
    if caller_id and order["user_id"] and caller_id != order["user_id"]:
        raise PermissionError("订单不存在或无权访问")
    return order


# ── Mock handlers ──────────────────────────────────────────────────────────


def mock_refund(args: dict) -> dict:
    order = _owned_order(args)
    amount = order["total_amount"]
    if amount > REFUND_AUTO_THRESHOLD:
        return {
            "status": "escalated",
            "order_id": order.get("order_id", args.get("order_id", "")),
            "amount": amount,
            "message": "退款金额超过自动处理阈值（¥1000），已为您转人工客服审核处理。",
        }
    return {
        "status": "success",
        "refund_id": f"RF{random.randint(100000, 999999)}",
        "order_id": args.get("order_id", ""),
        "reason": args.get("reason", ""),
        "amount": amount,
        "estimated_days": "3-5个工作日",
    }


def mock_return(args: dict) -> dict:
    order = _owned_order(args)
    return {
        "status": "success",
        "return_id": f"RT{random.randint(100000, 999999)}",
        "order_id": args.get("order_id", ""),
        "reason": args.get("reason", ""),
        "amount": order["total_amount"],
        "return_method": "快递上门取件",
        "estimated_days": "5-7个工作日",
    }


def mock_query_order(args: dict) -> dict:
    order = _owned_order(args)
    return {
        "order_id": args.get("order_id", ""),
        "status": order["status"],
        "items": order["items"],
        "total_amount": order["total_amount"],
        "created_at": order["created_at"],
        "estimated_delivery": order["estimated_delivery"],
    }


def mock_track_shipping(args: dict) -> dict:
    order = _owned_order(args)
    return {
        "order_id": args.get("order_id", ""),
        "carrier": "顺丰快递",
        "tracking_number": f"SF{random.randint(1000000000, 9999999999)}",
        "status": "运输中",
        "current_location": "北京分拨中心",
        "estimated_delivery": order["estimated_delivery"],
        "updates": [
            {"time": "12-18 08:00", "desc": "已从上海发出"},
            {"time": "12-18 14:30", "desc": "到达北京分拨中心"},
        ],
    }


def mock_complaint(args: dict) -> dict:
    return {
        "status": "success",
        "complaint_id": f"CP{random.randint(100000, 999999)}",
        "category": args.get("category", ""),
        "description": args.get("description", ""),
        "order_id": args.get("order_id", ""),
        "assigned_to": "客服专员-小李",
        "estimated_response": "24小时内",
    }


def create_default_tool_registry() -> ToolRegistry:
    """Create and populate registry with all mock tools."""
    registry = ToolRegistry()
    for tool in DEFAULT_TOOLS:
        registry.register(tool)
    return registry


DEFAULT_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="process_refund",
        intent="refund",
        description="处理退款申请。仅限用户本人的订单；执行前必须先向用户确认。",
        required_slots=["order_id", "reason"],
        handler=mock_refund,
        requires_confirmation=True,
        parameters_schema={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单号，例如 ORD1001",
                },
                "reason": {
                    "type": "string",
                    "description": "退款原因，用用户原话概括",
                },
            },
            "required": ["order_id", "reason"],
        },
    ),
    ToolDefinition(
        name="process_return",
        intent="return",
        description="处理退货申请。仅限用户本人的订单；执行前必须先向用户确认。",
        required_slots=["order_id", "reason"],
        handler=mock_return,
        requires_confirmation=True,
        parameters_schema={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单号，例如 ORD1001",
                },
                "reason": {
                    "type": "string",
                    "description": "退货原因，用用户原话概括",
                },
            },
            "required": ["order_id", "reason"],
        },
    ),
    ToolDefinition(
        name="query_order_status",
        intent="query_order",
        description="查询订单状态（状态、商品、金额、预计送达）。",
        required_slots=["order_id"],
        handler=mock_query_order,
        parameters_schema={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单号，例如 ORD1001",
                },
            },
            "required": ["order_id"],
        },
    ),
    ToolDefinition(
        name="track_shipping",
        intent="track_shipping",
        description="查询订单的物流配送信息。",
        required_slots=["order_id"],
        handler=mock_track_shipping,
        parameters_schema={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "订单号，例如 ORD1001",
                },
            },
            "required": ["order_id"],
        },
    ),
    ToolDefinition(
        name="submit_complaint",
        intent="complaint",
        description="提交投诉工单。",
        required_slots=["category", "description"],
        handler=mock_complaint,
        parameters_schema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "投诉分类，如：物流/商品质量/服务态度",
                },
                "description": {
                    "type": "string",
                    "description": "投诉内容描述",
                },
                "order_id": {
                    "type": "string",
                    "description": "相关订单号（如有）",
                },
            },
            "required": ["category", "description"],
        },
    ),
]
