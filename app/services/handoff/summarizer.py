"""Best-effort LLM conversation summary for handoff tickets.

The handoff chat path must never depend on the LLM being available —
a model outage cannot block a user from reaching a human — so every
failure mode here (error, timeout, empty input) returns None and the
caller falls back to the structured context alone.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol

from app.services.llm.base import LLMMessage

if TYPE_CHECKING:
    from app.models.database.message import Message

_SUMMARY_PROMPT = (
    "你是电商客服转人工摘要助手。根据以下对话记录，为人工坐席写一份不超过5句话的接手摘要，"
    "必须包含：用户核心诉求、关键事实（订单号/商品/金额等）、机器人已尝试的处理、用户当前情绪。"
    "直接输出摘要正文，不要任何前缀。\n\n对话记录：\n{transcript}"
)

# Long rants must not blow the prompt budget; each turn is capped.
_MAX_CHARS_PER_MESSAGE = 200


class SummaryLLM(Protocol):
    """Narrow LLM seam: anything with an async generate()."""

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> Any: ...


def _role_label(role: Any) -> str:
    """Render either a MessageRole enum or a bare string as 'user'/'assistant'."""
    return str(getattr(role, "value", role))


def _build_transcript(messages: list[Message]) -> str:
    return "\n".join(
        f"{_role_label(getattr(m, 'role', 'user'))}: "
        f"{str(getattr(m, 'content', ''))[:_MAX_CHARS_PER_MESSAGE]}"
        for m in messages
    )


async def summarize_for_handoff(
    messages: list[Message],
    llm_service: SummaryLLM,
    timeout_seconds: float = 3.0,
) -> str | None:
    """
    Summarize the dialogue so the human agent lands mid-conversation.

    Args:
        messages: Recent dialogue turns (any role order)
        llm_service: LLM with an async generate() (SummaryLLM protocol)
        timeout_seconds: Budget before the summary is abandoned

    Returns:
        str | None: Summary text, or None on empty input / error / timeout
    """
    if not messages:
        return None
    prompt = _SUMMARY_PROMPT.format(transcript=_build_transcript(messages))
    try:
        response = await asyncio.wait_for(
            llm_service.generate([LLMMessage(role="user", content=prompt)]),
            timeout=timeout_seconds,
        )
    except Exception:
        return None
    text = str(getattr(response, "content", "") or "").strip()
    return text or None
