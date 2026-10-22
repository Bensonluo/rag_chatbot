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


# Reply-language pin for LLM generation (global-audience doctrine).
# The personas' one-line rule「用与用户当前消息相同的语言回答」loses to
# zh-dominant context — a Chinese persona, Chinese tool JSON, Chinese
# prompt scaffolding (observed live 2026-09-30: an English conversation
# got full-Chinese replies on both a tool turn and a direct turn).
# Generators append this directive keyed off the RAW user turn so the
# reply language is pinned regardless of what the context drags toward.
GENERATION_LANG_DIRECTIVE: dict[str, str] = {
    LANG_EN: (
        "[Language requirement: Reply in English only — every sentence "
        "addressed to the user must be in English, including suggested commands. "
        'If suggesting human help, ask the user to type "human agent" to request '
        "assistance. This submits a request; do not promise a live connection, "
        "pickup or response time.]"
    ),
    LANG_ZH: "[语言要求：请只用中文回答用户。]",
}


# Deterministic evidence-gap copy (review 2026-09-26, #6): a knowledge
# question with an empty KB slice must not fall through to free LLM
# generation. Two sentences, because the correct next move differs:
# rephrase/handoff when the KB lacks the answer, retry/handoff when
# retrieval itself failed.
NO_EVIDENCE: dict[str, str] = {
    LANG_ZH: (
        "抱歉，暂未在知识库中找到与您问题相关的信息。"
        "您可以换个说法再试，或输入「转人工」请求人工协助。"
    ),
    LANG_EN: (
        "Sorry, I couldn't find anything about that in the knowledge base. "
        'You could rephrase and try again, or type "human agent" to request human assistance.'
    ),
}

RETRIEVAL_DEGRADED: dict[str, str] = {
    LANG_ZH: "检索服务暂时不可用，请稍后再试，或输入「转人工」请求人工协助。",
    LANG_EN: (
        "Search is temporarily unavailable. Please try again shortly, "
        'or type "human agent" to request human assistance.'
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
    LANG_ZH: "暂时未能确认转接人工客服成功。请稍后输入「转人工」重试。",
    LANG_EN: (
        "Sorry, I couldn't confirm your transfer to a human agent. "
        'Please try again shortly by typing "human agent".'
    ),
}

HANDOFF_ACK: dict[str, str] = {
    LANG_ZH: "您的人工客服请求已记录（工单号 #{ticket_id}）",
    LANG_EN: "Your request for a human agent is recorded (ticket #{ticket_id})",
}

HANDOFF_QUEUE: dict[str, str] = {
    LANG_ZH: "当前排队位置：第 {queue_position} 位",
    LANG_EN: "your position in the queue: {queue_position}",
}

HANDOFF_ALREADY_QUEUED: dict[str, str] = {
    LANG_ZH: "已有您的人工客服请求记录，请稍候",
    LANG_EN: "your request is already recorded",
}

HANDOFF_CONTEXT_NOTE: dict[str, str] = {
    LANG_ZH: "工单已关联本次会话，请稍候",
    LANG_EN: "the ticket is linked to this conversation",
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

CONFIRMATION_EXPIRED: dict[str, str] = {
    LANG_ZH: "您此前提交的操作已超时失效，为保障安全未执行。如仍需办理，请重新发起。",
    LANG_EN: (
        "The action you started earlier has expired. For your safety it "
        "was not executed — please start again if you still need it."
    ),
}
