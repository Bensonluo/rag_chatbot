"""Sentence-buffered PII redaction for token streams (review #4, 2026-09-26).

The finalize-time output guardrail rewrites the persisted text, but on
the token-streamed path the chunks have already reached the consumer —
the user sees the raw email while the stored turn holds ``[REDACTED]``.
Same doctrine as the claim stream gate (app/services/facts/stream_gate.py):
buffer the stream, redact each complete sentence, release only the
clean text. What the consumer sees and what persists stay identical.

The buffer is the sound unit because the PII patterns (phone, ID card,
email, bank card) are whitespace-free runs — a sentence boundary can
never split one mid-token, so a whole PII value is always inside a
single gated window.
"""

import logging

from app.services.guardrails.base import GuardrailService

logger = logging.getLogger(__name__)

_SENTENCE_TERMINATORS = "。！？!?；;\n"


def _split_complete(buffer: str) -> tuple[str, str]:
    """Split the buffer into (text through the last terminator, remainder).

    Mirrors the claim gate's splitter so the two buffers compose: a
    sentence the claim gate releases (terminator included) passes this
    splitter in one piece and is released immediately, not re-buffered.
    """
    cut = 0
    for i, char in enumerate(buffer):
        if char in _SENTENCE_TERMINATORS:
            cut = i + 1
    if cut == 0:
        return "", buffer
    return buffer[:cut], buffer[cut:]


class PIIStreamRedactor:
    """Hold streamed tokens until a sentence completes, redact PII, release.

    ``feed`` returns the text that is safe to emit now (possibly empty
    while a sentence is still forming); ``flush`` redacts and returns
    the remainder at end of stream. Never allowed to break chat: a
    guardrail outage releases the text unchanged (the finalize-time
    full-text pass still runs as backstop), and a blocking verdict
    drops the sentence (the finalize pass then judges the whole
    accumulated text).
    """

    def __init__(self, guardrail_service: GuardrailService) -> None:
        self._guard = guardrail_service
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        """Buffer one chunk; return redacted complete sentences."""
        self._buffer += chunk
        complete, self._buffer = _split_complete(self._buffer)
        if not complete:
            return ""
        return self._redact(complete)

    def flush(self) -> str:
        """Redact and return any held remainder; idempotent at stream end."""
        remainder, self._buffer = self._buffer, ""
        if not remainder:
            return ""
        return self._redact(remainder)

    def _redact(self, text: str) -> str:
        try:
            result = self._guard.check_output(text)
        except Exception:  # noqa: BLE001 - never break chat on a guard outage
            logger.exception("Stream PII redaction failed; releasing text unchanged")
            return text
        if result.was_blocked:
            # Drop the sentence; the finalize-time full-text pass still
            # sees everything emitted and can refuse the whole answer.
            logger.warning("Stream sentence blocked by output guardrail; dropped")
            return ""
        if result.sanitized_content and result.sanitized_content != text:
            return result.sanitized_content
        return text
