"""
Negative-emotion detection for human-handoff escalation.

Deterministic keyword matching — deliberately not an LLM call. At
800K-1M daily requests, emotion screening runs on every message, so
it must be cheap, latency-free, and reproducible in tests; an
occasional false positive only costs a human-agent handoff (the safe
direction), while a false negative leaves an angry user talking to a
bot (the expensive direction: churn + complaints).

Escalation fires on either signal class:
- **Anger**: explicit strong-negative words (气死, 骗子, scam, …).
- **Frustration**: repeated-failure phrasing (还是没收到, 一直没解决,
  第三次了, …). The completion marker 了 / failure verbs keep benign
  uses ("第三次下单有优惠吗") out of the escalation set.
"""

from __future__ import annotations

from dataclasses import dataclass

# Explicit anger — escalate immediately.
ANGER_KEYWORDS: tuple[str, ...] = (
    "气死",
    "太生气",
    "火大",
    "太过分",
    "骗子",
    "骗人",
    "垃圾服务",
    "垃圾客服",
    "垃圾平台",
    "恶心",
    "欺负人",
    "欺骗消费者",
    "投诉你们",
    "忍无可忍",
    "失望透顶",
    "太失望了",
    "scam",
    "terrible service",
    "unacceptable",
)

# Repeated-failure frustration — escalate (the bot already had its chance).
FRUSTRATION_KEYWORDS: tuple[str, ...] = (
    "还是没",
    "还是不行",
    "还是没有",
    "又不行",
    "又失败了",
    "一直没",
    "一直没有",
    "一直不",
    "第三次了",
    "好几次了",
    "都说了",
    "没人管",
    "不解决就投诉",
    "再也不买了",
    "still not working",
    "still no",
    "again and again",
)


@dataclass(frozen=True)
class EmotionSignal:
    """Result of screening one user message."""

    is_negative: bool
    matched: tuple[str, ...]
    signal_class: str  # "anger" | "frustration" | ""

    @property
    def should_escalate(self) -> bool:
        """Whether this message should escalate to a human agent."""
        return self.is_negative


def assess_emotion(message: str) -> EmotionSignal:
    """
    Screen a message for negative emotion.

    Args:
        message: Raw (post-guardrail) user message

    Returns:
        EmotionSignal with the matched keywords and their class.
    """
    if not message:
        return EmotionSignal(is_negative=False, matched=(), signal_class="")

    text = message.lower()
    matched_anger = tuple(kw for kw in ANGER_KEYWORDS if kw in text or kw in message)
    if matched_anger:
        return EmotionSignal(True, matched_anger, "anger")

    matched_frustration = tuple(kw for kw in FRUSTRATION_KEYWORDS if kw in text or kw in message)
    if matched_frustration:
        return EmotionSignal(True, matched_frustration, "frustration")

    return EmotionSignal(is_negative=False, matched=(), signal_class="")
