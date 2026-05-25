"""
Tool definitions and registry for Function Calling.

Provides mock tool handlers for customer service operations
(refund, order query, shipping tracking, etc.) that simulate
real API responses.
"""
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.models.enums.intent import TASK_INTENTS


@dataclass
class ToolDefinition:
    """Definition of a callable tool."""
    name: str
    intent: str
    description: str
    required_slots: list[str]
    handler: Callable[[dict], dict]


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

    def get_tool_for_intent(self, intent: str) -> Optional[ToolDefinition]:
        return self._tools.get(intent)

    async def execute(self, intent: str, args: dict) -> ToolResult:
        tool = self._tools.get(intent)
        if not tool:
            return ToolResult(success=False, data={}, message=f"No tool for intent: {intent}")
        try:
            result = tool.handler(args)
            return ToolResult(success=True, data=result, message="OK")
        except Exception as e:
            return ToolResult(success=False, data={}, message=str(e))


# ── Mock handlers ──────────────────────────────────────────────────────────

def mock_refund(args: dict) -> dict:
    return {
        "status": "success",
        "refund_id": f"RF{random.randint(100000, 999999)}",
        "order_id": args.get("order_id", ""),
        "reason": args.get("reason", ""),
        "amount": args.get("amount", 299.00),
        "estimated_days": "3-5个工作日",
    }


def mock_return(args: dict) -> dict:
    return {
        "status": "success",
        "return_id": f"RT{random.randint(100000, 999999)}",
        "order_id": args.get("order_id", ""),
        "reason": args.get("reason", ""),
        "return_method": "快递上门取件",
        "estimated_days": "5-7个工作日",
    }


def mock_query_order(args: dict) -> dict:
    return {
        "order_id": args.get("order_id", ""),
        "status": "已发货",
        "items": ["商品A x1", "商品B x2"],
        "total_amount": 598.00,
        "created_at": "2024-12-15 10:30:00",
        "estimated_delivery": "2024-12-20",
    }


def mock_track_shipping(args: dict) -> dict:
    return {
        "order_id": args.get("order_id", ""),
        "carrier": "顺丰快递",
        "tracking_number": f"SF{random.randint(1000000000, 9999999999)}",
        "status": "运输中",
        "current_location": "北京分拨中心",
        "estimated_delivery": "明天下午",
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
        description="处理退款申请",
        required_slots=["order_id", "reason"],
        handler=mock_refund,
    ),
    ToolDefinition(
        name="process_return",
        intent="return",
        description="处理退货申请",
        required_slots=["order_id", "reason"],
        handler=mock_return,
    ),
    ToolDefinition(
        name="query_order_status",
        intent="query_order",
        description="查询订单状态",
        required_slots=["order_id"],
        handler=mock_query_order,
    ),
    ToolDefinition(
        name="track_shipping",
        intent="track_shipping",
        description="查询物流信息",
        required_slots=["order_id"],
        handler=mock_track_shipping,
    ),
    ToolDefinition(
        name="submit_complaint",
        intent="complaint",
        description="提交投诉",
        required_slots=["category", "description"],
        handler=mock_complaint,
    ),
]
