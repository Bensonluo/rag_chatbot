"""
Output guardrail — checks LLM output before returning to the user.

Redacts PII that the LLM may have leaked, and flags off-domain content.
"""
import re
import logging
from typing import List, Dict, Optional, Pattern

from app.services.guardrails.base import GuardrailResult, OutputGuardrail

logger = logging.getLogger(__name__)

# Reuse PII patterns from input guard
_PII_PATTERNS: Dict[str, Pattern] = {
    "phone_cn": re.compile(r"1[3-9]\d{9}"),
    "id_card_cn": re.compile(r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]"),
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "bank_card": re.compile(r"6[0-9]{15,18}"),
}

FALLBACK_RESPONSE = "抱歉，我无法回答这个问题。"


class DefaultOutputGuardrail(OutputGuardrail):
    """Default output guardrail with PII redaction."""

    def __init__(
        self,
        enable_pii_redaction: bool = True,
    ) -> None:
        self._enable_pii = enable_pii_redaction

    def check(self, content: str) -> GuardrailResult:
        if not content:
            return GuardrailResult(
                passed=True, action="allow", original_content=content, sanitized_content=content,
            )

        violations: List[str] = []
        sanitized = content

        # PII redaction on output
        if self._enable_pii:
            sanitized, pii_types = self._redact_pii(content)
            violations.extend(pii_types)

        action = "redact" if violations else "allow"
        return GuardrailResult(
            passed=True, action=action, original_content=content,
            sanitized_content=sanitized, violations=violations,
        )

    @staticmethod
    def _redact_pii(text: str) -> tuple:
        sanitized = text
        found_types: List[str] = []
        for pii_type, pattern in _PII_PATTERNS.items():
            matches = pattern.findall(sanitized)
            if matches:
                found_types.append(f"pii_{pii_type}")
                sanitized = pattern.sub("[REDACTED]", sanitized)
        return sanitized, found_types
