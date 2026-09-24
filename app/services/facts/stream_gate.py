"""Sentence-buffered claim gating for token streams (A2 streaming close-out).

The post-hoc claim gate rewrites the persisted text, but on the
token-streamed path violating numbers may already have reached the
consumer — the user screenshots the wrong promise (中消协 liability
direction) while the history stores the corrected one. Industry
consensus for streamed output moderation is chunked, buffered checking
(NeMo Guardrails chunked streaming; "The Guardrail Tax" 2026): buffer
the stream, check each window, release only what passes. This checker
is clause/sentence-scoped, so the sound window is the SENTENCE — hold
tokens until a terminator, gate the complete sentence, release the
corrected text. What the user sees and what persists stay identical.

The gate is never allowed to break chat: a checker failure releases
the text ungated (the post-hoc full-text pass still runs as backstop).
"""

import logging

from app.config.settings import settings
from app.services.facts.claim_check import apply_violations, check_policy_claims
from app.services.facts.fact_store import FactStore, PolicyFact
from app.services.facts.metrics import CLAIM_VIOLATIONS
from app.services.observability.trace_events import emit_trace

logger = logging.getLogger(__name__)

_SENTENCE_TERMINATORS = "。！？!?；;\n"


class StreamClaimGate:
    """Hold streamed tokens until a sentence completes, gate it, release it.

    ``feed`` returns the text that is safe to emit now (possibly empty
    while a sentence is still forming); ``flush`` gates and releases the
    remainder at end of stream. Emitted text is the corrected text —
    concatenating everything emitted reproduces the gated response
    byte-for-byte, so the persisted turn equals what the user saw.
    """

    def __init__(self, facts: list[PolicyFact]) -> None:
        self._facts = facts
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        """Buffer one token chunk; return gated complete sentences."""
        self._buffer += chunk
        complete, self._buffer = _split_complete(self._buffer)
        if not complete:
            return ""
        return self._gate(complete)

    def flush(self) -> str:
        """Gate and return any held remainder; idempotent at stream end."""
        remainder, self._buffer = self._buffer, ""
        if not remainder:
            return ""
        return self._gate(remainder)

    def _gate(self, text: str) -> str:
        try:
            result = check_policy_claims(text, self._facts, check_actions=True)
        except Exception:
            # Never-fail-chat: a gate outage must not eat the response.
            # The post-hoc full-text pass still runs as backstop.
            logger.exception("Stream claim gate failed; releasing text ungated")
            return text
        for violation in result.violations:
            CLAIM_VIOLATIONS.labels(reason=violation.reason).inc()
            # Same demo-panel triple as the post-hoc gate in nodes.py —
            # streamed or not, the viewer sees the catch.
            emit_trace(
                "cs.claim_gate",
                action="rewrite",
                reason=violation.reason,
                clause=violation.clause,
                grounded=violation.grounded_statement,
            )
        if result.violations:
            return apply_violations(text, result)
        return text


def _split_complete(buffer: str) -> tuple[str, str]:
    """Split the buffer into (text through the last terminator, remainder)."""
    cut = 0
    for i, char in enumerate(buffer):
        if char in _SENTENCE_TERMINATORS:
            cut = i + 1
    if cut == 0:
        return "", buffer
    return buffer[:cut], buffer[cut:]


def make_stream_gate(message: str) -> StreamClaimGate | None:
    """Pre-decide sentence gating for a turn (deterministic, pre-generation).

    Returns None when the turn is not checkable (setting off, empty
    message, or empty fact subgraph) so pure token streaming keeps its
    latency profile for chitchat; returns an armed gate exactly when the
    post-hoc gate would have checked this response.
    """
    if not message or not settings.FACT_CLAIM_CHECK_ENABLED:
        return None
    facts = FactStore.load_default().subgraph_for(message)
    if not facts:
        return None
    return StreamClaimGate(facts)
