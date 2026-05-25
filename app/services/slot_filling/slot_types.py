"""
Slot schemas per business intent.

Defines required/optional slots for each task-oriented intent,
with extraction prompts for missing slot values.
"""
from typing import Dict, Any

# Per-intent slot definitions for task-oriented flows
INTENT_SLOT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "refund": {
        "required": ["order_id", "reason"],
        "optional": ["amount"],
        "slots": {
            "order_id": {
                "type": "string",
                "prompt": "请提供您的订单号",
                "patterns": [r"订单号[：:]?\s*([A-Za-z0-9]+)", r"(?:order|订单)\s*([A-Za-z0-9]{3,})"],
            },
            "reason": {
                "type": "string",
                "prompt": "请问退款原因是什么？",
                "patterns": [
                    r"原因[是为：:\s]+(.{2,20})",
                    r"因为\s*(.{2,20})",
                    r"由于\s*(.{2,20})",
                    r"(质量(?:问题|缺陷|有瑕疵)?)",
                ],
            },
            "amount": {
                "type": "number",
                "prompt": "请问退款金额是多少？",
            },
        },
    },
    "return": {
        "required": ["order_id", "reason"],
        "optional": [],
        "slots": {
            "order_id": {
                "type": "string",
                "prompt": "请提供您的订单号",
                "patterns": [r"订单号[：:]?\s*([A-Za-z0-9]+)", r"(?:order|订单)\s*([A-Za-z0-9]{3,})"],
            },
            "reason": {
                "type": "string",
                "prompt": "请问退货原因是什么？",
                "patterns": [
                    r"原因[是为：:\s]+(.{2,20})",
                    r"因为\s*(.{2,20})",
                    r"由于\s*(.{2,20})",
                    r"(质量(?:问题|缺陷|有瑕疵)?)",
                ],
            },
        },
    },
    "query_order": {
        "required": ["order_id"],
        "optional": [],
        "slots": {
            "order_id": {
                "type": "string",
                "prompt": "请提供您的订单号",
                "patterns": [r"订单号[：:]?\s*([A-Za-z0-9]+)", r"(?:order|订单)\s*([A-Za-z0-9]{3,})"],
            },
        },
    },
    "track_shipping": {
        "required": ["order_id"],
        "optional": [],
        "slots": {
            "order_id": {
                "type": "string",
                "prompt": "请提供您的订单号",
                "patterns": [r"订单号[：:]?\s*([A-Za-z0-9]+)", r"(?:order|订单)\s*([A-Za-z0-9]{3,})"],
            },
        },
    },
    "complaint": {
        "required": ["category", "description"],
        "optional": ["order_id"],
        "slots": {
            "category": {
                "type": "string",
                "prompt": "请问您要投诉哪个方面？（如：商品质量、服务态度、物流配送等）",
            },
            "description": {
                "type": "string",
                "prompt": "请详细描述您的问题",
            },
            "order_id": {
                "type": "string",
                "prompt": "请提供相关订单号（如有）",
                "patterns": [r"订单号[：:]?\s*(\d+)", r"(?:order|订单)\s*(\d{5,})"],
            },
        },
    },
}


def get_missing_slots(intent: str, filled: Dict[str, Any]) -> list[str]:
    """Return list of required slot names not yet filled."""
    schema = INTENT_SLOT_SCHEMAS.get(intent, {})
    required = schema.get("required", [])
    return [s for s in required if s not in filled]


def get_next_prompt(intent: str, filled: Dict[str, Any]) -> str | None:
    """Get the prompt for the next missing required slot."""
    schema = INTENT_SLOT_SCHEMAS.get(intent, {})
    missing = get_missing_slots(intent, filled)
    if not missing:
        return None
    slots = schema.get("slots", {})
    return slots.get(missing[0], {}).get("prompt", f"请提供{missing[0]}")


def extract_slots_from_message(intent: str, message: str, existing: Dict[str, Any]) -> Dict[str, Any]:
    """Extract slot values from user message using regex patterns."""
    import re

    schema = INTENT_SLOT_SCHEMAS.get(intent, {})
    slots = schema.get("slots", {})
    extracted = dict(existing)

    for slot_name, slot_def in slots.items():
        if slot_name in extracted:
            continue
        patterns = slot_def.get("patterns", [])
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                try:
                    extracted[slot_name] = match.group(1)
                except IndexError:
                    extracted[slot_name] = match.group(0)
                break

    return extracted


# Legacy slot definitions for backward compatibility with existing slot fillers.
# The dialogue module uses INTENT_SLOT_SCHEMAS above instead.
SLOT_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "product": {
        "entity_type": "Product",
        "keywords": {
            "iPhone": "iPhone", "iPad": "iPad", "MacBook": "MacBook",
            "MacBook Pro": "MacBook Pro", "MacBook Air": "MacBook Air",
            "iMac": "iMac", "Apple Watch": "Apple Watch", "AirPods": "AirPods",
        },
        "patterns": [r"(?P<product_name>[A-Z][a-zA-Z]+(?:\s(?:Pro|Air|Max|Mini|Plus|Ultra|SE|Lite))?)(?:\s\d+)?"],
    },
    "issue": {
        "entity_type": "Issue",
        "keywords": {
            "蓝屏": "蓝屏", "黑屏": "黑屏", "死机": "死机", "卡顿": "卡顿",
            "闪退": "闪退", "崩溃": "崩溃", "无法开机": "无法开机",
            "无法连接": "无法连接", "发热": "发热", "耗电快": "耗电异常",
            "信号差": "信号问题", "登录失败": "登录失败", "数据丢失": "数据丢失",
        },
        "patterns": [r"(?P<issue_desc>无法.{1,6}|不能.{1,6}|总是.{1,6}|经常.{1,6})"],
    },
    "platform": {
        "entity_type": "Platform",
        "keywords": {
            "iOS": "iOS", "Android": "Android", "Windows": "Windows",
            "macOS": "macOS", "Linux": "Linux", "HarmonyOS": "HarmonyOS",
            "鸿蒙": "HarmonyOS",
        },
        "patterns": [r"(?P<platform>iOS|Android|Windows|macOS|Linux|HarmonyOS)\s*(?P<version>\d+(?:\.\d+)*)?"],
    },
    "feature": {
        "entity_type": "Feature",
        "keywords": {
            "WiFi": "WiFi", "蓝牙": "蓝牙", "NFC": "NFC", "GPS": "GPS",
            "5G": "5G", "快充": "快充", "无线充电": "无线充电",
        },
        "patterns": [],
    },
    "category": {
        "entity_type": "Category",
        "keywords": {
            "硬件": "硬件", "软件": "软件", "网络": "网络", "账号": "账号",
            "退款": "退款", "物流": "物流", "售后": "售后", "退货": "退货",
        },
        "patterns": [],
    },
    "time_period": {
        "entity_type": "TimePeriod",
        "keywords": {},
        "patterns": [
            r"(?P<year>20\d{2})\s*年",
            r"(?P<month>\d{1,2})\s*月",
            r"(?P<relative>今天|昨天|前天|上周|本周|上个月|这个月|最近)",
        ],
    },
}
