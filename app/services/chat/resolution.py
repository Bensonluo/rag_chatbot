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
import unicodedata

# Match the complete utterance, not a positive word inside a denial,
# question, quotation, hypothetical, or a request for further help.
# A negative-keyword blacklist cannot cover those constructions reliably.
# Permit only a short affirmative statement with optional courtesy phrases;
# unfamiliar wording stays unconfirmed and eligible for human QA.
_SEPARATOR = r"[\s,.!;，。！；]*"
_COURTESY = (
    r"(?:谢谢(?:你|您)?(?:的帮助)?|多谢(?:你|您)?|"
    r"(?:非常)?感谢(?:(?:你|您)(?:的帮助)?|帮助)?|"
    r"好的|好了|好|是的|嗯|yes|ok(?:ay)?|thanks(?:\s+a\s+lot)?|"
    r"thank\s+you(?:\s+(?:very\s+much|for\s+your\s+help))?)"
)
_ZH_CONFIRMATION = (
    r"(?:(?:我的|这个)?问题)?(?:现在|这下)?(?:已经|终于)?"
    r"(?:(?:都)?解决了|搞定了|没有问题了)"
)
_EN_CONFIRMATION = (
    r"(?:"
    r"(?:solved|fixed|sorted|resolved|worked)|"
    r"(?:it|that|this)\s+(?:worked|solved\s+it|fixed\s+it)|"
    r"(?:(?:my|the)\s+)?(?:problem|issue)\s+(?:(?:is|has\s+been)\s+)?"
    r"(?:solved|fixed|sorted|resolved)|"
    r"(?:it|that|this)(?:\s+(?:is|has\s+been)|'s)\s+(?:solved|fixed|sorted|resolved)|"
    r"(?:all\s+set|(?:i\s+am|i'm|we\s+are|we're)\s+all\s+set)|"
    r"(?:everything\s+is|it\s+is|it's)\s+working\s+now"
    r")(?:\s+now)?"
)
_CONFIRMATION = re.compile(
    rf"(?:{_COURTESY}{_SEPARATOR}){{0,2}}"
    rf"(?:{_ZH_CONFIRMATION}|{_EN_CONFIRMATION})"
    rf"{_SEPARATOR}(?:{_COURTESY}{_SEPARATOR}){{0,2}}",
    re.IGNORECASE,
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
    normalized = unicodedata.normalize("NFKC", text).replace("’", "'").strip()
    return _CONFIRMATION.fullmatch(normalized) is not None
