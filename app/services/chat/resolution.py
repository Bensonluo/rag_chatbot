"""User-confirmed resolution signal (review 2026-09-26, #11).

Containment (no escalation, no downvote) is the ABSENCE of failure
signals. Resolution confirmation is the PRESENCE of a positive one:
the user's own message says the problem is solved. Deliberately
precision-first — a false "resolved" is worse than a missed one,
because the reopen detector flips the session open on any follow-up
and a wrongly-resolved session would both inflate the KPI and hide
the reopen. Plain thanks is NOT resolution: "谢谢" closes a
conversation, not a problem.

When to upgrade: these patterns are the deterministic floor. An
LLM-based classifier can replace ``is_resolution_confirmation``
behind the same signature once there is labeled traffic to tune on
(the slot-filling hybrid has the same shape).
"""

from __future__ import annotations

import re

# zh: 解决了 (not preceded by 没/未 — "还没解决" means still open), 搞定了,
# 没有问题了. en: solved/fixed/sorted/worked as whole words, "all set".
RESOLVE_CONFIRMATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<![没未])解决了"),
    re.compile(r"搞定了"),
    re.compile(r"没有问题了"),
    re.compile(r"\b(?:solved|fixed|sorted|worked)\b", re.IGNORECASE),
    re.compile(r"\ball\s+set\b", re.IGNORECASE),
)


def is_resolution_confirmation(text: str) -> bool:
    """True when the user's message confirms the problem is solved.

    Precision over recall: borderline phrasings ("试试看", "先这样")
    return False — an unresolved-but-missed session only loses the
    resolved tag, while a wrongly-resolved one silently breaks reopen
    detection.
    """
    if not text:
        return False
    return any(pattern.search(text) for pattern in RESOLVE_CONFIRMATION_PATTERNS)
