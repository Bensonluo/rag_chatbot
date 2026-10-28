"""Model-free copy for durable action outcomes and saved adapter receipts.

Ledger completion records an adapter's return value, not payment settlement.
The caller owns execution/state transitions and the outer output/stream guard.
This module deliberately never quotes provider diagnostics or arbitrary JSON.
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import UUID

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")
_UUID = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_DURATION = re.compile(
    r"[0-9]{1,3}(?: *[-–] *[0-9]{1,3})? *"
    r"(?:(?:business|working) +)?(?:days?|hours?|weeks?|工作日|个工作日|天|小时|周)",
    re.IGNORECASE,
)
_CANCEL_CONTEXTS = {"cancel", "deny", "canceldeny"}

_COPY: dict[str, tuple[str, str]] = {
    "recorded": (
        "已找到此前保存的处理结果，本轮没有重复执行。",
        "The previously saved result was found. No new action was executed in this turn.",
    ),
    "unknown": (
        "本次办理结果尚未确认。为避免重复，不会再次执行同一请求。请联系人工核实。",
        "The result of this action is still unconfirmed. To avoid duplicates, this same "
        "request will not be executed again. Please contact support to verify it.",
    ),
    "unavailable": (
        "暂时无法核对办理记录，本轮未发起新的操作。请稍后查询或联系人工核实。",
        "The processing record could not be checked, so no new action was started in "
        "this turn. Please check again later or contact support.",
    ),
    "conflict": (
        "本次确认与原请求记录不一致，本轮未发起新的操作。请联系人工核实原请求。",
        "This confirmation does not match the original request record. No new action "
        "was started in this turn. Please contact support to verify the original request.",
    ),
    "unverifiable": (
        "此前待确认记录无法核验，本轮未发起新的操作。请联系人工核实后再办理。",
        "The earlier pending-action record could not be verified. No new action was "
        "started in this turn. Please contact support to verify it before proceeding.",
    ),
    "expired": (
        "此前待确认内容已过期，本轮未发起新的操作。如仍需办理，请重新提供办理信息并确认。",
        "The pending confirmation has expired. No new action was started in this turn. "
        "If you still need the action, provide the details again and confirm them.",
    ),
    "cancel_pending": (
        "已取消本次待确认内容，本轮未发起新的操作。",
        "The pending confirmation was cancelled. No new action was started in this turn.",
    ),
    "stop_only": (
        "本次只停止后续办理，不代表此前请求已撤销。",
        "Further processing is stopped; this does not mean the earlier request was reversed.",
    ),
    "recorded_cancel": (
        "本次只停止后续办理，不会撤销此前已提交的请求。",
        "Further processing is stopped; this does not reverse a previously submitted request.",
    ),
    "unsuccessful": (
        "此前请求返回未成功，请联系人工核实。",
        "The saved result reports that the earlier request was unsuccessful. "
        "Please contact support to verify it.",
    ),
    "review": (
        "此前工具记录显示该退款仍需人工审核，未发起退款。本轮没有重复执行。",
        "The saved tool result requires human review and reports that no refund was "
        "issued. No new action was executed in this turn.",
    ),
}


def _reference(value: str | None) -> str | None:
    if not isinstance(value, str) or not _UUID.fullmatch(value):
        return None
    return str(UUID(value))


def _identifier(value: Any) -> str | None:
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else None


def _amount(value: Any) -> str | None:
    if type(value) not in (int, float, Decimal):
        return None
    try:
        number = Decimal(str(value))
        if not number.is_finite() or not Decimal(0) <= number <= Decimal("1000000000000"):
            return None
        # Do not round an unsupported precision into a fabricated saved amount.
        if number != number.quantize(Decimal("0.01")):
            return None
        return format(number, ".2f")
    except (InvalidOperation, ValueError):
        return None


def _receipt_fields(data: dict[str, Any], intent: str, is_en: bool) -> list[str]:
    if intent not in {"refund", "return"}:
        return []
    labels = {
        "order_id": ("订单编号", "Order reference"),
        "refund_id": ("退款记录编号", "Refund record reference"),
        "return_id": ("退货记录编号", "Return record reference"),
        "amount": ("记录金额", "Amount"),
        "estimated_days": ("记录的预计处理时间", "Recorded estimated processing time"),
    }
    fields = ("order_id", "refund_id" if intent == "refund" else "return_id")
    values = [(key, _identifier(data.get(key))) for key in fields]
    values.append(("amount", _amount(data.get("amount"))))
    duration = data.get("estimated_days")
    if isinstance(duration, str) and len(duration) <= 80 and _DURATION.fullmatch(duration):
        values.append(("estimated_days", duration))
    return [f"{labels[key][int(is_en)]}: {value}" for key, value in values if value is not None]


def render_action_outcome(
    *,
    status: str,
    replayed: bool,
    tool_intent: str,
    success: bool,
    data: dict[str, Any],
    message: str,
    language: str,
    operation_id: str | None,
    context: str = "confirm",
) -> str | None:
    """Render a safe outcome, or defer a new result/refund-review to existing nodes.

    ``unknown`` may have reached the provider; ``unavailable`` must mean this
    turn never entered its handler. Neither says earlier attempts did nothing.
    ``confirmation_required`` also covers legacy confirmations without an ID.
    Cancel/deny only cancel pending content, never reverse submitted business.
    """
    del message  # Provider error strings are intentionally never user-facing copy.
    is_en = language != "zh"
    cancelling = context in _CANCEL_CONTEXTS
    reference = _reference(operation_id)
    parts: list[str] = []

    def copy(key: str) -> str:
        return _COPY[key][int(is_en)]

    if status == "completed":
        if reference is None:
            parts.append(copy("unverifiable"))
        else:
            if cancelling:
                parts.append(copy("recorded_cancel"))
            if success is not True:
                if replayed is not False or context != "confirm":
                    parts.append(copy("recorded"))
                parts.append(copy("unsuccessful"))
            elif tool_intent == "refund" and data.get("status") == "escalated":
                if not cancelling:
                    return None  # The existing hook must still record/reuse the review ticket.
                parts.append(copy("review"))
            elif replayed is False and context == "confirm":
                return None
            else:
                parts.append(copy("recorded"))
                parts.extend(_receipt_fields(data, tool_intent, is_en))
    elif status == "not_started":
        parts.append(copy("cancel_pending" if cancelling else "expired"))
    elif status in {"confirmation_required", "legacy_unverifiable"}:
        parts.append(copy("unverifiable"))
    else:
        if cancelling:
            parts.append(copy("stop_only"))
        parts.append(copy(status if status in {"unavailable", "conflict"} else "unknown"))
    if reference is not None:
        label = "Verification reference" if is_en else "核对参考编号"
        parts.append(f"{label}: {reference}")
    return "\n".join(parts)
