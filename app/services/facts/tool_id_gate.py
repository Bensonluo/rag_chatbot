"""Sentence-buffered identifier honesty gate for tool-result turns.

Live stress test (2026-10-22): the complaint tool turn replied with an
invented ticket id (「CP122245」— real tickets are DB integers like #5)
and an invented staff member (「客服专员-小李」). The persona already
forbids fabrication, but a deterministic backstop must exist: a reply
quoting an identifier the tool result does not contain is a promise the
system never made, money-visible to the user and quotable in a dispute.

Mechanics mirror StreamClaimGate (stream_gate.py): identifiers are
asserted at sentence granularity, so the stream is buffered per
sentence, the complete sentence is checked, and only gated text is
released. Concatenating everything emitted reproduces the gated
response byte-for-byte — the persisted turn equals what the user saw.

The gate is never allowed to break chat: any checker failure releases
the text ungated.
"""

import json
import logging
import re
from typing import Any

from app.services.dialogue.i18n import LANG_EN, LANG_ZH
from app.services.facts.metrics import TOOL_ID_VIOLATIONS
from app.services.observability.trace_events import emit_trace

logger = logging.getLogger(__name__)

_SENTENCE_TERMINATORS = "。！？!?；;\n"

# Identifier-shaped tokens: ticket/order/refund-style prefixes with a
# digit run (CP122245, TK12, ORD1001), or a #prefixed number (#5).
# Deliberately narrow — bare numbers (amounts, years) and non-prefix
# letter runs (RMB100) never match, so prices and dates pass untouched.
ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:CP|TK|ORD|RF|WO|TICKET)[-_]?\d{2,}(?![A-Za-z0-9])"
    r"|(?<![A-Za-z0-9])#\d{1,10}(?!\d)"
)

# Deterministic replacement sentence quoted from the tool result, so a
# corrected turn still tells the user what reference actually exists.
_REPLACEMENT: dict[str, str] = {
    LANG_ZH: "本次操作的系统编号：{ids}。",
    LANG_EN: "System reference for this action: {ids}.",
}
_NO_REFERENCE: dict[str, str] = {
    LANG_ZH: "本次操作已受理，暂无可展示的编号。",
    LANG_EN: "The request has been accepted; no reference number is available to show.",
}


def tool_result_references(tool_result: dict[str, Any]) -> list[str]:
    """Collect displayable id values from the tool result.

    Top-level fields whose name ends in ``_id`` (ticket_id, refund_id,
    order_id...) plus any id-shaped token already inside a string value
    — quoting these keeps the corrected sentence informative.
    """
    references: list[str] = []
    for key, value in tool_result.items():
        if not isinstance(key, str):
            continue
        if key.endswith("_id") and value is not None and str(value) != "":
            references.append(str(value))
        elif isinstance(value, str):
            references.extend(ID_TOKEN_RE.findall(value))
    return references


def _foreign_identifiers(text: str, tool_json: str) -> list[str]:
    """Id-shaped tokens in ``text`` that appear nowhere in the tool JSON."""
    foreign: list[str] = []
    for token in ID_TOKEN_RE.findall(text):
        if token in tool_json:
            continue
        # "#5" is honest when the tool result carries the number 5.
        if token.startswith("#") and token[1:] in tool_json:
            continue
        foreign.append(token)
    return foreign


def enforce_tool_id_honesty(text: str, tool_result: dict[str, Any], lang: str) -> str:
    """Replace every sentence asserting a foreign identifier.

    A sentence quoting a fabricated id is wholly untrustworthy — the
    live failure bundled the fake ticket with a fake staff name and a
    fake response ETA in one sentence — so the whole sentence is
    replaced by the deterministic reference line, not token-patched.
    Never raises; on checker failure the text returns unchanged.
    """
    if not text:
        return text
    try:
        tool_json = json.dumps(tool_result, ensure_ascii=False, default=str)
        if not _foreign_identifiers(text, tool_json):
            return text
        ids = tool_result_references(tool_result)
        replacement = (
            _REPLACEMENT.get(lang, _REPLACEMENT[LANG_EN]).format(ids=", ".join(ids))
            if ids
            else _NO_REFERENCE.get(lang, _NO_REFERENCE[LANG_EN])
        )
        complete, remainder = _split_complete(text)
        gated = _gate_sentences(complete, tool_json, replacement)
        tail = _gate_sentences(remainder, tool_json, replacement)
        return gated + tail
    except Exception:  # noqa: BLE001 - never-fail-chat
        logger.exception("Tool-id honesty gate failed; releasing text ungated")
        return text


def _gate_sentences(text: str, tool_json: str, replacement: str) -> str:
    """Gate one sentence-run, replacing violating sentences in place."""
    if not text:
        return ""
    output: list[str] = []
    for sentence in _iter_sentences(text):
        if _foreign_identifiers(sentence, tool_json):
            TOOL_ID_VIOLATIONS.labels(kind="fabricated_identifier").inc()
            emit_trace(
                "cs.tool_id_gate",
                action="replace_sentence",
                sentence=sentence[:80],
            )
            output.append(replacement)
        else:
            output.append(sentence)
    return "".join(output)


def _split_complete(buffer: str) -> tuple[str, str]:
    """Split the buffer into (text through the last terminator, remainder)."""
    cut = 0
    for i, char in enumerate(buffer):
        if char in _SENTENCE_TERMINATORS:
            cut = i + 1
    if cut == 0:
        return "", buffer
    return buffer[:cut], buffer[cut:]


def _iter_sentences(text: str) -> list[str]:
    """Split a sentence-run into its sentences (terminators kept)."""
    sentences: list[str] = []
    start = 0
    for i, char in enumerate(text):
        if char in _SENTENCE_TERMINATORS:
            sentences.append(text[start : i + 1])
            start = i + 1
    if start < len(text):
        sentences.append(text[start:])
    return sentences


class ToolIdStreamGate:
    """Hold streamed tokens until a sentence completes, gate it, release it.

    Same interface and byte-identical-emission contract as
    ``StreamClaimGate``: ``feed`` returns safe-to-emit text (possibly
    empty mid-sentence); ``flush`` gates the held remainder.
    """

    def __init__(self, tool_result: dict[str, Any], lang: str) -> None:
        self._tool_json = json.dumps(tool_result, ensure_ascii=False, default=str)
        self._ids = tool_result_references(tool_result)
        self._lang = lang
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
        replacement = (
            _REPLACEMENT.get(self._lang, _REPLACEMENT[LANG_EN]).format(ids=", ".join(self._ids))
            if self._ids
            else _NO_REFERENCE.get(self._lang, _NO_REFERENCE[LANG_EN])
        )
        return _gate_sentences(text, self._tool_json, replacement)
