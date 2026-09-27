"""Condense a follow-up question into a standalone retrieval query.

History joins the RAG pipeline at generation time — the retrieval legs
see the raw message, so 「那运费呢？」 searches a fragment whose
subject only exists in the conversation (review 2026-09-26, #6). This
module rewrites the retrieval query against recent turns before
search. Fail-open by design, like every retrieval leg: any failure
returns None and the caller searches the original message — the
rewrite is recall leverage, never a new outage path.
"""

import logging

from app.services.llm.base import LLMMessage, LLMServiceBase

logger = logging.getLogger(__name__)

_MAX_HISTORY_MESSAGES = 6
_MAX_QUERY_CHARS = 300
_REWRITE_SYSTEM_PROMPT = (
    "你是对话式检索的查询改写器。结合对话历史，把用户的最新消息改写成"
    "一个独立、完整、可用于知识库检索的问题。直接输出改写后的问题，"
    "不要解释、不要加引号。如果最新消息本身已经独立完整，原样输出它。"
)


async def condense_for_retrieval(
    llm_service: LLMServiceBase,
    message: str,
    history: list[LLMMessage],
) -> str | None:
    """Return a standalone retrieval query, or None to keep the original.

    None on every failure path (LLM error, empty, whitespace-only, or
    runaway output) — the rewrite must not break the search it feeds.
    """
    if not message.strip() or not history:
        return None
    recent = history[-_MAX_HISTORY_MESSAGES:]
    messages = [
        LLMMessage(role="system", content=_REWRITE_SYSTEM_PROMPT),
        *recent,
        LLMMessage(role="user", content=f"最新消息：{message}"),
    ]
    try:
        response = await llm_service.generate(messages=messages, max_tokens=128, temperature=0.0)
    except Exception:  # noqa: BLE001 - rewrite is recall leverage, not availability
        logger.warning("Query rewrite failed; searching the original message")
        return None
    rewritten = (response.content or "").strip()
    if len(rewritten) < 2 or len(rewritten) > _MAX_QUERY_CHARS:
        return None
    return rewritten
