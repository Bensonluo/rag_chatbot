"""Language-following copy for the dialogue graph's fixed branches.

The personas instruct the model to answer in the user's language, but
several branches never touch the LLM: the deterministic evidence-gap
copy, guardrail refusals, the generation-failure fallback, and the
handoff acknowledgement (deliberately model-free so an LLM outage can
never block reaching a human). Those branches pick their canned copy
here, keyed by the user's current message language.

The product serves a global audience — no language is assumed. For
LLM-generated text the persona follows whatever language the user
writes in. For the canned copy here, detection is intentionally a
cheap heuristic, not a language model: a CJK ideograph makes the turn
Chinese (a Chinese sentence naming an English product stays Chinese),
and everything else — Latin letters, other alphabets (Japanese kana,
Korean, Cyrillic...), digits-only, emoji-only, empty — selects
English, the international fallback, since zh/en are the only copy
pairs that exist today.

Known limitation: slot-collection prompts gained EN templates via
``prompt_en`` in slot_types.py; the meta-intent replies (resume hints,
cancel/confirm/deny acknowledgements) remain Chinese-only — they are
money-moving copy and stay single-language until reviewed.
"""

import re

LANG_ZH = "zh"
LANG_EN = "en"

_CJK_RE = re.compile(r"[一-鿿]")


def detect_language(text: str) -> str:
    """Return the turn language for canned-copy selection."""
    if _CJK_RE.search(text):
        return LANG_ZH
    return LANG_EN


# Deterministic evidence-gap copy (review 2026-09-26, #6): a knowledge
# question with an empty KB slice must not fall through to free LLM
# generation. Two sentences, because the correct next move differs:
# rephrase/handoff when the KB lacks the answer, retry/handoff when
# retrieval itself failed.
NO_EVIDENCE: dict[str, str] = {
    LANG_ZH: (
        "抱歉，暂未在知识库中找到与您问题相关的信息。"
        "您可以换个说法再试，或输入「转人工」联系人工客服。"
    ),
    LANG_EN: (
        "Sorry, I couldn't find anything about that in the knowledge base. "
        'You could rephrase and try again, or type "human agent" to reach one.'
    ),
}

RETRIEVAL_DEGRADED: dict[str, str] = {
    LANG_ZH: "检索服务暂时不可用，请稍后再试，或输入「转人工」联系人工客服。",
    LANG_EN: (
        "Search is temporarily unavailable. Please try again shortly, "
        'or type "human agent" to reach one.'
    ),
}

GUARDRAIL_INPUT_BLOCKED: dict[str, str] = {
    LANG_ZH: "抱歉，您的消息未通过安全检查，请重新描述您的问题。",
    LANG_EN: "Sorry, your message didn't pass our safety check. Please rephrase your question.",
}

GUARDRAIL_OUTPUT_BLOCKED: dict[str, str] = {
    LANG_ZH: "抱歉，该回复未能通过安全检查，请重新提问。",
    LANG_EN: "Sorry, that reply didn't pass our safety check. Please ask differently.",
}

GENERATION_FAILED: dict[str, str] = {
    LANG_ZH: "抱歉，生成回复时出现错误，请稍后重试。",
    LANG_EN: "Sorry, something went wrong while generating a reply. Please try again shortly.",
}

# Handoff acknowledgement parts (fixed template, never LLM-generated).
HANDOFF_EMOTION_PREFIX: dict[str, str] = {
    LANG_ZH: "非常抱歉给您带来了不好的体验，",
    LANG_EN: "We're very sorry about the experience, ",
}

HANDOFF_NO_TICKET: dict[str, str] = {
    LANG_ZH: "正在为您转接人工客服，请稍候。",
    LANG_EN: "Connecting you to a human agent, please hold on.",
}

HANDOFF_ACK: dict[str, str] = {
    LANG_ZH: "已为您转接人工客服（工单号 #{ticket_id}）",
    LANG_EN: "You're being transferred to a human agent (ticket #{ticket_id})",
}

HANDOFF_QUEUE: dict[str, str] = {
    LANG_ZH: "当前排队人数：{queue_position} 人",
    LANG_EN: "people ahead of you in the queue: {queue_position}",
}

HANDOFF_ALREADY_QUEUED: dict[str, str] = {
    LANG_ZH: "您已在排队中，请耐心等待",
    LANG_EN: "you're already in the queue, please hold on",
}

HANDOFF_CONTEXT_NOTE: dict[str, str] = {
    LANG_ZH: "人工客服可查看本次会话的完整上下文，请稍候",
    LANG_EN: "the human agent can see this conversation's full context, one moment",
}

# Irreversible-action confirmation gate (fixed template, never
# LLM-generated): the ask that protects a money-moving action must be
# deterministic AND intelligible to the user it addresses — an English
# speaker who cannot read the gate copy cannot meaningfully confirm.
CONFIRMATION_ACTION_NAMES: dict[str, dict[str, str]] = {
    "refund": {LANG_ZH: "退款", LANG_EN: "Refund"},
    "return": {LANG_ZH: "退货", LANG_EN: "Return"},
    "query_order": {LANG_ZH: "查订单", LANG_EN: "Order lookup"},
    "track_shipping": {LANG_ZH: "查物流", LANG_EN: "Shipping tracking"},
    "complaint": {LANG_ZH: "投诉", LANG_EN: "Complaint"},
    "faq": {LANG_ZH: "常见问题", LANG_EN: "FAQ"},
    "policy": {LANG_ZH: "政策查询", LANG_EN: "Policy lookup"},
    "handoff": {LANG_ZH: "转人工", LANG_EN: "Human agent"},
}

CONFIRMATION_ASK: dict[str, str] = {
    LANG_ZH: (
        "⚠️ 即将为您执行「{action}」：{detail}。\n"
        "该操作不可自动撤销。请回复「确认」执行，或回复「取消」放弃。"
    ),
    LANG_EN: (
        '⚠️ About to execute "{action}": {detail}.\n'
        'This action cannot be undone automatically. Reply "confirm" to '
        'proceed, or "cancel" to abandon.'
    ),
}

CONFIRMATION_DETAIL_JOIN: dict[str, str] = {
    LANG_ZH: "，",
    LANG_EN: ", ",
}

CONFIRMATION_DETAIL_EMPTY: dict[str, str] = {
    LANG_ZH: "（无附加信息）",
    LANG_EN: "(no additional details)",
}
