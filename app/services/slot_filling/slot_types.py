"""
Slot schemas per business intent.

Defines required/optional slots for each task-oriented intent,
with extraction prompts for missing slot values.
"""

import re
from typing import Any

# Per-intent slot definitions for task-oriented flows
INTENT_SLOT_SCHEMAS: dict[str, dict[str, Any]] = {
    "refund": {
        "required": ["order_id", "reason"],
        "optional": ["amount"],
        "slots": {
            "order_id": {
                "type": "string",
                "prompt": "请提供您的订单号",
                "prompt_en": "Please provide your order number",
                "patterns": [
                    r"订单号[：:]?\s*([A-Za-z0-9]+)",
                    r"(?:order|订单)\s*(?:no\.?|number|#)?\s*([A-Za-z0-9]*\d[A-Za-z0-9]*)",
                    # Bare order code (ORD1001): digit-requiring with a
                    # letter floor, ASCII-lookaround boundaries so CJK
                    # adjacency ("ORD1001订单") still extracts and a
                    # letterless digit run never qualifies.
                    r"(?<![A-Za-z0-9])([A-Za-z]+\d{3,}[A-Za-z0-9]*)(?![A-Za-z0-9])",
                ],
            },
            "reason": {
                "type": "string",
                "prompt": "请问退款原因是什么？",
                "prompt_en": "May I ask the reason for the refund?",
                "patterns": [
                    r"原因[是为：:\s]+(.{2,20})",
                    r"因为\s*(.{2,20})",
                    r"由于\s*(.{2,20})",
                    r"(质量(?:问题|缺陷|有瑕疵)?)",
                    # English reason phrasings — without these, an English
                    # "because it arrived broken" could only be filled by
                    # the LLM extractor (same outage loop as complaint
                    # category, 2026-09-30).
                    r"(?:because|since)\s+(.{2,40})",
                    r"reason(?:\s+is)?[：:]?\s*(.{2,40})",
                    r"(?:damaged|broken|defective|faulty|wrong\s+item|never\s+arrived|didn'?t\s+arrive)",
                ],
            },
            "amount": {
                "type": "number",
                "prompt": "请问退款金额是多少？",
                "prompt_en": "What is the refund amount?",
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
                "prompt_en": "Please provide your order number",
                "patterns": [
                    r"订单号[：:]?\s*([A-Za-z0-9]+)",
                    r"(?:order|订单)\s*(?:no\.?|number|#)?\s*([A-Za-z0-9]*\d[A-Za-z0-9]*)",
                    # Bare order code (ORD1001): digit-requiring with a
                    # letter floor, ASCII-lookaround boundaries so CJK
                    # adjacency ("ORD1001订单") still extracts and a
                    # letterless digit run never qualifies.
                    r"(?<![A-Za-z0-9])([A-Za-z]+\d{3,}[A-Za-z0-9]*)(?![A-Za-z0-9])",
                ],
            },
            "reason": {
                "type": "string",
                "prompt": "请问退货原因是什么？",
                "prompt_en": "May I ask the reason for the return?",
                "patterns": [
                    r"原因[是为：:\s]+(.{2,20})",
                    r"因为\s*(.{2,20})",
                    r"由于\s*(.{2,20})",
                    r"(质量(?:问题|缺陷|有瑕疵)?)",
                    # English reason phrasings — without these, an English
                    # "because it arrived broken" could only be filled by
                    # the LLM extractor (same outage loop as complaint
                    # category, 2026-09-30).
                    r"(?:because|since)\s+(.{2,40})",
                    r"reason(?:\s+is)?[：:]?\s*(.{2,40})",
                    r"(?:damaged|broken|defective|faulty|wrong\s+item|never\s+arrived|didn'?t\s+arrive)",
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
                "prompt_en": "Please provide your order number",
                "patterns": [
                    r"订单号[：:]?\s*([A-Za-z0-9]+)",
                    r"(?:order|订单)\s*(?:no\.?|number|#)?\s*([A-Za-z0-9]*\d[A-Za-z0-9]*)",
                    # Bare order code (ORD1001): digit-requiring with a
                    # letter floor, ASCII-lookaround boundaries so CJK
                    # adjacency ("ORD1001订单") still extracts and a
                    # letterless digit run never qualifies.
                    r"(?<![A-Za-z0-9])([A-Za-z]+\d{3,}[A-Za-z0-9]*)(?![A-Za-z0-9])",
                ],
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
                "prompt_en": "Please provide your order number",
                "patterns": [
                    r"订单号[：:]?\s*([A-Za-z0-9]+)",
                    r"(?:order|订单)\s*(?:no\.?|number|#)?\s*([A-Za-z0-9]*\d[A-Za-z0-9]*)",
                    # Bare order code (ORD1001): digit-requiring with a
                    # letter floor, ASCII-lookaround boundaries so CJK
                    # adjacency ("ORD1001订单") still extracts and a
                    # letterless digit run never qualifies.
                    r"(?<![A-Za-z0-9])([A-Za-z]+\d{3,}[A-Za-z0-9]*)(?![A-Za-z0-9])",
                ],
            },
        },
    },
    "complaint": {
        "required": ["category", "description"],
        "optional": ["order_id"],
        "slots": {
            # Keyword taxonomy, not free-form capture: a pattern-less
            # category meant the slot could ONLY be filled by the LLM
            # extractor — during a provider outage (GLM 429 storm,
            # 2026-09-30) the complaint flow re-prompted forever. The
            # patterns have no capture group, so the slot value is the
            # canonical keyword itself (group(0)), not the sentence.
            "category": {
                "type": "string",
                "prompt": "请问您要投诉哪个方面？（如：商品质量、服务态度、物流配送等）",
                "prompt_en": "What would you like to complain about? (e.g. product quality, service attitude, delivery)",
                "patterns": [
                    r"(商品质量|质量问题|假冒|假货|虚假宣传|破损|损坏|坏了|有瑕疵)",
                    r"(服务态度|态度差|客服态度|服务差|客服不专业)",
                    r"(物流|快递|配送|发货|送货|到货)",
                    r"(售后|维修|退换|三包)",
                    r"(价格|收费|乱扣费|虚假发货)",
                    r"(product\s+quality|quality\s+issue|damaged|broken|defective|faulty|counterfeit|fake)",
                    r"(service\s+attitude|rude|unhelpful|unprofessional)",
                    r"(delivery|shipping|logistics|courier|packaging|late\s+arrival)",
                    r"(after.?sale|customer\s+service|overcharg\w+|pricing|wrong\s+charge)",
                ],
            },
            "description": {
                "type": "string",
                "prompt": "请详细描述您的问题",
                "prompt_en": "Please describe the problem in detail",
            },
            # Digit-requiring: the shared "(?:order)\s*([A-Za-z0-9]{3,})"
            # pattern captures the word after "order" ("...my ORD1001
            # order arrived..." → "arrived") — an English sentence's
            # verb must not become an order id.
            "order_id": {
                "type": "string",
                "prompt": "请提供相关订单号（如有）",
                "prompt_en": "Please provide the relevant order number, if any",
                "patterns": [
                    r"订单号[：:]?\s*([A-Za-z0-9]+)",
                    r"(?:order|订单)\s*(?:no\.?|number|#)?\s*([A-Za-z0-9]*\d[A-Za-z0-9]*)",
                    # ASCII-only boundaries: \b counts CJK as word chars,
                    # so "ORD1001订单" has no \b between them.
                    r"(?<![A-Za-z0-9])([A-Za-z]+\d{3,}[A-Za-z0-9]*)(?![A-Za-z0-9])",
                ],
            },
        },
    },
}


def get_missing_slots(intent: str, filled: dict[str, Any]) -> list[str]:
    """Return list of required slot names not yet filled."""
    schema = INTENT_SLOT_SCHEMAS.get(intent, {})
    required = schema.get("required", [])
    return [s for s in required if s not in filled]


def get_next_prompt(intent: str, filled: dict[str, Any], lang: str = "zh") -> str | None:
    """Get the prompt for the next missing required slot.

    ``lang`` selects the asking language ("en" picks ``prompt_en``;
    everything else falls back to the zh template — a zh ask beats no
    ask for a slot definition shipped without an EN prompt).
    """
    schema = INTENT_SLOT_SCHEMAS.get(intent, {})
    missing = get_missing_slots(intent, filled)
    if not missing:
        return None
    slots = schema.get("slots", {})
    slot_def = slots.get(missing[0], {})
    prompt = slot_def.get("prompt", f"请提供{missing[0]}")
    if lang == "en":
        prompt = slot_def.get("prompt_en") or prompt
    return str(prompt)


def extract_slots_from_message(
    intent: str, message: str, existing: dict[str, Any]
) -> dict[str, Any]:
    """Extract slot values from user message using regex patterns."""
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


# Digit-bearing tokens (ORD1001, 12345) plus the connector words an
# order-reference answer is made of. Used to tell "my order is ORD1001"
# (an order_id answer, not a complaint) from "the screen of ORD1001
# arrived cracked" (a narrative that happens to cite an order).
_ORDER_TOKEN = re.compile(r"[A-Za-z]*\d[\dA-Za-z]*|\d+")
_ORDER_WORDS = re.compile(
    r"order|no\.?|number|#|订单号|订单|单号|号|is|my|the|的|是", re.IGNORECASE
)


def is_order_reference_only(message: str) -> bool:
    """True when the message carries nothing but an order reference.

    The complaint description terminator (collect_slots_node) consults
    this: a bare order-number answer must fill ``order_id``, never
    masquerade as the free-form complaint description. Everything else
    — a narrative citing an order, a plain description — reads as a
    description once a category is known.
    """
    stripped = _ORDER_TOKEN.sub(" ", message)
    stripped = _ORDER_WORDS.sub(" ", stripped)
    words = re.findall(r"[A-Za-z]{2,}", stripped)
    cjk = re.findall(r"[一-鿿]", stripped)
    return len(words) + len(cjk) < 2


# Legacy slot definitions for backward compatibility with existing slot fillers.
# The dialogue module uses INTENT_SLOT_SCHEMAS above instead.
SLOT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "product": {
        "entity_type": "Product",
        "keywords": {
            "iPhone": "iPhone",
            "iPad": "iPad",
            "MacBook": "MacBook",
            "MacBook Pro": "MacBook Pro",
            "MacBook Air": "MacBook Air",
            "iMac": "iMac",
            "Apple Watch": "Apple Watch",
            "AirPods": "AirPods",
        },
        "patterns": [
            r"(?P<product_name>[A-Z][a-zA-Z]+(?:\s(?:Pro|Air|Max|Mini|Plus|Ultra|SE|Lite))?)(?:\s\d+)?"
        ],
    },
    "issue": {
        "entity_type": "Issue",
        "keywords": {
            "蓝屏": "蓝屏",
            "黑屏": "黑屏",
            "死机": "死机",
            "卡顿": "卡顿",
            "闪退": "闪退",
            "崩溃": "崩溃",
            "无法开机": "无法开机",
            "无法连接": "无法连接",
            "发热": "发热",
            "耗电快": "耗电异常",
            "信号差": "信号问题",
            "登录失败": "登录失败",
            "数据丢失": "数据丢失",
        },
        "patterns": [r"(?P<issue_desc>无法.{1,6}|不能.{1,6}|总是.{1,6}|经常.{1,6})"],
    },
    "platform": {
        "entity_type": "Platform",
        "keywords": {
            "iOS": "iOS",
            "Android": "Android",
            "Windows": "Windows",
            "macOS": "macOS",
            "Linux": "Linux",
            "HarmonyOS": "HarmonyOS",
            "鸿蒙": "HarmonyOS",
        },
        "patterns": [
            r"(?P<platform>iOS|Android|Windows|macOS|Linux|HarmonyOS)\s*(?P<version>\d+(?:\.\d+)*)?"
        ],
    },
    "feature": {
        "entity_type": "Feature",
        "keywords": {
            "WiFi": "WiFi",
            "蓝牙": "蓝牙",
            "NFC": "NFC",
            "GPS": "GPS",
            "5G": "5G",
            "快充": "快充",
            "无线充电": "无线充电",
        },
        "patterns": [],
    },
    "category": {
        "entity_type": "Category",
        "keywords": {
            "硬件": "硬件",
            "软件": "软件",
            "网络": "网络",
            "账号": "账号",
            "退款": "退款",
            "物流": "物流",
            "售后": "售后",
            "退货": "退货",
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
