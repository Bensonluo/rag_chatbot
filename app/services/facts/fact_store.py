"""Curated policy-fact table and entity-anchored subgraph retrieval.

Industry consensus from the Air Canada / Klarna postmortems: hard policy
numbers (SLA days, return windows, thresholds) must never live only in
LLM-visible free text — they live in typed, curated data, and generated
claims are verified against that data before leaving the graph. The LLM
proposes; this table is the authority.

Design (mirrors the FAQ fast path's shipped-configuration pattern):

- ``policy_facts.json`` is versioned configuration extracted verbatim
  from the FAQ table and tool constants. Every numeric fact that has a
  ``faq_id`` is pinned to the FAQ answer by a no-drift test — the two
  sources cannot silently diverge.
- ``subgraph_for(message)`` pulls an entity-anchored subgraph (GRAG
  style): only facts whose anchor keywords appear in the user's message
  enter the check, so the claim verifier never evaluates facts the turn
  never invited.
- Facts carry subject metadata (支付宝 vs 银行卡 vs 信用卡, 普通订单 vs
  大促) so entailment is channel-aware rather than one-number-fits-all.
- Load failures degrade to an empty table: grounding is a verification
  layer, never a dependency of availability.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_FACTS_FILE = Path(__file__).parent / "policy_facts.json"

_VALUE_RE = re.compile(r"^(\d+)(?:-(\d+))?$")


@dataclass(frozen=True)
class PolicyFact:
    """One checkable policy fact with anchoring and subject metadata."""

    id: str
    topic: str
    subject: str
    # Substring anchors: any hit in the user message pulls this fact.
    anchors: tuple[str, ...]
    # A sentence/clause must contain one of these for the fact to be
    # eligible to check a numeric claim (claim↔topic binding).
    topic_keywords: tuple[str, ...]
    # When present in the clause, restricts checking to this subject.
    subject_keywords: tuple[str, ...]
    # The subject assumed when the clause names none (exactly one per
    # topic — enforced by tests).
    default_subject: bool
    # duration_range | duration_point | window | cap | statement
    kind: str
    # Numeric value as "lo-hi" or a point "n"; "" for statements.
    value: str
    # 工作日 | 小时 | 分钟 | 天 | 元 | "" — mismatched units are never
    # cross-checked (conservative).
    unit: str
    # Grounded replacement text quoted when a claim is corrected.
    statement: str
    # Source FAQ entry for the no-drift pin; None for tool-only facts.
    faq_id: str | None

    def value_bounds(self) -> tuple[int, int] | None:
        """Numeric [lo, hi] bounds; None when the fact has no number."""
        if not self.value:
            return None
        match = _VALUE_RE.match(self.value)
        if match is None:
            return None
        lo = int(match.group(1))
        hi = int(match.group(2) or match.group(1))
        return lo, hi


class FactStore:
    """Read-only in-process view of the curated fact table."""

    def __init__(self, facts_file: str | Path = DEFAULT_FACTS_FILE) -> None:
        self._facts = _load_facts(facts_file)

    @property
    def facts(self) -> list[PolicyFact]:
        return list(self._facts)

    def subgraph_for(self, message: str) -> list[PolicyFact]:
        """Facts anchored to the message: any anchor substring hit pulls
        the fact into the verification subgraph."""
        if not message:
            return []
        return [fact for fact in self._facts if any(a in message for a in fact.anchors)]

    _default: FactStore | None = None

    @classmethod
    def load_default(cls) -> FactStore:
        """Process-wide singleton over the shipped fact table."""
        if cls._default is None:
            cls._default = cls()
        return cls._default


def _load_facts(facts_file: str | Path) -> list[PolicyFact]:
    """Load facts from JSON, degrading to empty on any failure.

    A missing or malformed table disables grounding (nothing is
    checkable) but never breaks the chat path.
    """
    path = Path(facts_file)
    if not path.exists():
        logger.info("Policy facts file not found (%s); claim gate inactive", path)
        return []
    try:
        raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Policy facts file unreadable (%s); claim gate inactive", exc)
        return []

    facts: list[PolicyFact] = []
    for entry in raw:
        try:
            facts.append(
                PolicyFact(
                    id=str(entry["id"]),
                    topic=str(entry["topic"]),
                    subject=str(entry.get("subject", "")),
                    anchors=tuple(str(a) for a in entry.get("anchors", [])),
                    topic_keywords=tuple(str(k) for k in entry.get("topic_keywords", [])),
                    subject_keywords=tuple(str(k) for k in entry.get("subject_keywords", [])),
                    default_subject=bool(entry.get("default_subject", False)),
                    kind=str(entry.get("kind", "statement")),
                    value=str(entry.get("value", "")),
                    unit=str(entry.get("unit", "")),
                    statement=str(entry.get("statement", "")),
                    faq_id=entry.get("faq_id"),
                )
            )
        except (KeyError, TypeError) as exc:
            logger.warning("Skipping malformed policy fact entry: %s", exc)
    return facts
