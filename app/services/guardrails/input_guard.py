"""
Input guardrail — checks user input before it reaches the LLM.

Detects prompt injection, redacts PII, blocks toxic content, and enforces
length limits. All checks are local (no external API dependency).
"""
import re
import logging
from typing import List, Dict, Optional, Pattern

from app.services.guardrails.base import GuardrailResult, InputGuardrail

logger = logging.getLogger(__name__)

# Prompt injection patterns
_INJECTION_PATTERNS: List[Pattern] = [
    re.compile(r"ignore\s+(previous|prior|above|all)\s+(instructions?|prompts?|rules?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|your\s+instructions?)", re.IGNORECASE),
    re.compile(r"system\s*:", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"###\s*instruction", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"dan\s+mode", re.IGNORECASE),
    re.compile(r"developer\s+mode", re.IGNORECASE),
    re.compile(r"override\s+(previous|all|safety)\s*(instructions?|rules?|guidelines?)?", re.IGNORECASE),
]

# PII patterns
_PII_PATTERNS: Dict[str, Pattern] = {
    "phone_cn": re.compile(r"1[3-9]\d{9}"),
    "id_card_cn": re.compile(r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]"),
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "bank_card": re.compile(r"6[0-9]{15,18}"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# Toxicity keywords (Chinese + English)
_TOXICITY_KEYWORDS: List[str] = [
    "炸弹", "制造炸弹", "毒品", "制毒", "自杀", "杀人",
    "恐怖袭击", "袭击计划",
]

BLOCK_MESSAGE = "抱歉，您的输入包含不允许的内容，请重新表述您的问题。"


class DefaultInputGuardrail(InputGuardrail):
    """Default input guardrail with injection detection, PII redaction, and toxicity blocking."""

    def __init__(
        self,
        enable_pii_redaction: bool = True,
        enable_injection_detection: bool = True,
        enable_toxicity_check: bool = True,
        max_length: int = 5000,
        custom_blocklist: Optional[List[str]] = None,
    ) -> None:
        self._enable_pii = enable_pii_redaction
        self._enable_injection = enable_injection_detection
        self._enable_toxicity = enable_toxicity_check
        self._max_length = max_length
        self._blocklist = _TOXICITY_KEYWORDS + (custom_blocklist or [])

    def check(self, content: str) -> GuardrailResult:
        if not content or not content.strip():
            return GuardrailResult(
                passed=False, action="block", original_content=content,
                sanitized_content="", violations=["empty_content"],
            )

        violations: List[str] = []
        sanitized = content

        # 1. Length check
        if len(content) > self._max_length:
            violations.append("content_too_long")
            return GuardrailResult(
                passed=False, action="block", original_content=content,
                sanitized_content="", violations=violations,
                metadata={"length": len(content), "max_length": self._max_length},
            )

        # 2. Prompt injection detection
        if self._enable_injection:
            for pattern in _INJECTION_PATTERNS:
                if pattern.search(content):
                    violations.append("prompt_injection_detected")
                    logger.warning("Prompt injection detected: %s", pattern.pattern)
                    return GuardrailResult(
                        passed=False, action="block", original_content=content,
                        sanitized_content=BLOCK_MESSAGE, violations=violations,
                    )

        # 3. Toxicity check
        if self._enable_toxicity:
            content_lower = content.lower()
            for keyword in self._blocklist:
                if keyword in content_lower:
                    violations.append("toxic_content")
                    return GuardrailResult(
                        passed=False, action="block", original_content=content,
                        sanitized_content=BLOCK_MESSAGE, violations=violations,
                    )

        # 4. PII redaction
        if self._enable_pii:
            sanitized, pii_types = self._redact_pii(content)
            if pii_types:
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
