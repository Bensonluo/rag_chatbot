"""LLM-assisted slot extraction for the task slot pipeline.

Regex extraction (``extract_slots_from_message``) stays the first pass:
cheap, deterministic, zero latency. When it finds nothing new while a
slot collection is actually in progress, this module asks the LLM to
pull slot values out of the free-form message, guided by
``INTENT_SLOT_SCHEMAS``. Natural answers ("上周买的手机想退，屏幕刮花
了") land in the right slots instead of depending on the whole-message
fallback that assigns any short reply to the first missing slot.

Every failure path degrades to an empty dict — slot extraction must
never break a turn.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from app.services.llm.base import LLMMessage
from app.services.slot_filling.slot_types import INTENT_SLOT_SCHEMAS

if TYPE_CHECKING:
    from app.services.llm.base import LLMServiceBase

logger = logging.getLogger(__name__)

# Cap extracted values so a hostile or rambling message cannot stuff
# arbitrarily long strings into slot state (regex paths cap at ~20
# chars for the same reason).
_MAX_VALUE_LENGTH = 200
_MAX_LLM_TOKENS = 256

_PROMPT_TEMPLATE = """你是电商智能客服的槽位提取器。当前任务意图是「{intent}」。

可提取的槽位（输出 JSON 的键只能是下列槽位名）：
{slot_descriptions}

已收集的槽位（不要重复返回）：{existing}

用户消息：{message}

只输出一个 JSON 对象，例如 {{"order_id": "2024090112345"}}。
提取不到的槽位不要出现在输出里，不要输出任何其他文字。"""


def _build_prompt(intent: str, message: str, filled: dict[str, Any]) -> str | None:
    """Render the extraction prompt; None when the intent has no schema."""
    schema = INTENT_SLOT_SCHEMAS.get(intent)
    if not schema:
        return None
    slots = schema.get("slots", {})
    descriptions = "\n".join(
        f"- {name} ({definition.get('type', 'string')}): {definition.get('prompt', '')}"
        for name, definition in slots.items()
    )
    existing = json.dumps(filled, ensure_ascii=False) if filled else "无"
    return _PROMPT_TEMPLATE.format(
        intent=intent,
        slot_descriptions=descriptions,
        existing=existing,
        message=message,
    )


def _strip_code_fences(text: str) -> str:
    """Drop ```json fences some models wrap around JSON output."""
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _coerce(raw: Any, slot_type: str) -> Any:
    """Coerce a raw JSON value to the schema-declared slot type."""
    if slot_type == "number":
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    value = str(raw).strip()
    return value[:_MAX_VALUE_LENGTH]


def parse_slot_json(content: str, intent: str) -> dict[str, Any]:
    """Parse and validate an LLM slot response against the intent schema.

    Unknown keys, wrong types, and empty values are dropped — the
    schema is the contract, the model only proposes values.
    """
    slot_defs = INTENT_SLOT_SCHEMAS.get(intent, {}).get("slots", {})
    try:
        data = json.loads(_strip_code_fences(content.strip()))
    except json.JSONDecodeError:
        logger.debug("Slot JSON parse failed for intent %s", intent)
        return {}
    if not isinstance(data, dict):
        return {}

    result: dict[str, Any] = {}
    for name, raw in data.items():
        definition = slot_defs.get(name)
        if definition is None or raw is None:
            continue
        value = _coerce(raw, str(definition.get("type", "string")))
        if value is None or value == "":
            continue
        result[name] = value
    return result


async def llm_extract_slots(
    intent: str,
    message: str,
    filled: dict[str, Any],
    llm_service: LLMServiceBase,
) -> dict[str, Any]:
    """Extract NEW slot values via the LLM; {} when nothing usable.

    Never raises — callers treat an empty dict as "nothing extracted"
    and continue with their existing fallbacks. Only slots missing
    from ``filled`` are returned so callers can merge blindly.
    """
    prompt = _build_prompt(intent, message, filled)
    if prompt is None:
        return {}
    try:
        response = await llm_service.generate(
            messages=[
                LLMMessage(role="system", content="Output only valid JSON."),
                LLMMessage(role="user", content=prompt),
            ],
            max_tokens=_MAX_LLM_TOKENS,
            temperature=0.0,
        )
        extracted = parse_slot_json(response.content, intent)
    except Exception as exc:
        logger.warning("LLM slot extraction failed for intent %s: %s", intent, exc)
        return {}
    return {name: value for name, value in extracted.items() if name not in filled}
