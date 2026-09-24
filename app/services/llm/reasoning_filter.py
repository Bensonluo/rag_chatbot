"""Stateful filter that strips ``<think>…</think>`` reasoning spans from LLM streams.

GLM (and other reasoning models) may inline ``<think>`` blocks in the streamed
content. The streaming path cannot regex the whole text per chunk because the
tags can be split across chunk boundaries at arbitrary offsets, so this module
provides a small state machine that holds back only the bytes that could still
grow into a tag.
"""

from __future__ import annotations

import re

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_COMPLETE_SPAN_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _partial_tag_length(buffer: str, tag: str) -> int:
    """Return the length of the longest buffer suffix that is a proper prefix of tag."""
    max_check = min(len(buffer), len(tag) - 1)
    for length in range(max_check, 0, -1):
        if buffer.endswith(tag[:length]):
            return length
    return 0


class ReasoningFilter:
    """Drop reasoning spans from a chunked token stream.

    ``feed`` returns only the bytes that are safe to emit now; tag fragments
    split across chunks are held internally. Call ``flush`` at end of stream
    to release held non-tag text (an unterminated span is dropped).
    """

    def __init__(self) -> None:
        self._thinking = False
        self._pending = ""

    def feed(self, chunk: str) -> str:
        """Consume one stream chunk and return the emittable text."""
        if not chunk:
            return ""
        buffer = self._pending + chunk
        self._pending = ""
        out: list[str] = []
        while buffer:
            tag = _THINK_CLOSE if self._thinking else _THINK_OPEN
            index = buffer.find(tag)
            if index != -1:
                if not self._thinking:
                    out.append(buffer[:index])
                self._thinking = not self._thinking
                buffer = buffer[index + len(tag) :]
                continue
            hold = _partial_tag_length(buffer, tag)
            if hold:
                if not self._thinking:
                    out.append(buffer[:-hold])
                self._pending = buffer[-hold:]
            elif not self._thinking:
                out.append(buffer)
            buffer = ""
        return "".join(out)

    def flush(self) -> str:
        """Release any held text; an unterminated reasoning span is discarded."""
        released = "" if self._thinking else self._pending
        self._pending = ""
        return released


def strip_reasoning(text: str) -> str:
    """Remove all complete reasoning spans from a finished (non-streamed) response."""
    if _THINK_OPEN not in text:
        return text
    return _COMPLETE_SPAN_RE.sub("", text).strip()
