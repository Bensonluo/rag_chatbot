"""Offline golden-set evaluation for intent routing (industry quality gate).

Every mainstream customer-service bot platform ships a labeled evaluation
set and gates releases on it (Zendesk resolutions benchmark, Intercom Fin
resolution rate, 阿里小蜜 意图准确率考核). This module is that gate for
the rule-based routing layer: a curated golden set of e-commerce queries
with expected intents, scored by strict intent accuracy and by routing-
group accuracy (which pipeline bucket the intent lands in — the property
that actually determines user-visible behavior).

The runner is provider-agnostic: any intent detector implementing
``detect_with_confidence`` can be scored, but the bundled gate runs the
deterministic rule-based detector so CI holds without LLM keys.
"""

import inspect
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.models.enums.intent import (
    DIRECT_INTENTS,
    GRAPH_INTENTS,
    HANDOFF_INTENTS,
    META_INTENTS,
    RAG_INTENTS,
    TASK_INTENTS,
)
from app.services.intent.base import IntentDetector

logger = logging.getLogger(__name__)

GOLDEN_SET_FILE = Path(__file__).parent / "golden_set.json"

# Routing buckets mirror route_intent_node in the dialogue graph, with
# task intents split out of the graph's agent/task distinction (that
# split is a wiring choice, not a detection property).
_ROUTE_GROUPS = (
    ("task", frozenset(TASK_INTENTS)),
    ("rag", frozenset(RAG_INTENTS)),
    ("direct", frozenset(DIRECT_INTENTS)),
    ("meta", frozenset(META_INTENTS)),
    ("handoff", frozenset(HANDOFF_INTENTS)),
    ("graph", frozenset(GRAPH_INTENTS)),
)


def route_bucket(intent: str) -> str:
    """Map an intent to its dialogue-pipeline routing bucket."""
    for bucket, intents in _ROUTE_GROUPS:
        if intent in intents:
            return bucket
    return "direct"  # unknown and unmapped intents fall through to direct


@dataclass
class GoldenCase:
    """One labeled query; strict intent plus routing-bucket expectation."""

    id: str
    query: str
    expect_intent: str


@dataclass
class CaseResult:
    case: GoldenCase
    predicted_intent: str
    passed: bool
    route_passed: bool


@dataclass
class EvalReport:
    """Aggregate scores over a golden set run."""

    total: int = 0
    passed: int = 0
    route_passed: int = 0
    failures: list[CaseResult] = field(default_factory=list)
    by_intent: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def route_accuracy(self) -> float:
        return self.route_passed / self.total if self.total else 0.0


def load_golden_cases(path: str | Path = GOLDEN_SET_FILE) -> list[GoldenCase]:
    """Load golden cases, degrading to empty on any read/shape failure."""
    file_path = Path(path)
    try:
        raw: list[dict[str, Any]] = json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Golden set unreadable (%s): %s", file_path, exc)
        return []
    cases: list[GoldenCase] = []
    for index, item in enumerate(raw):
        try:
            cases.append(
                GoldenCase(
                    id=str(item["id"]),
                    query=str(item["query"]),
                    expect_intent=str(item["expect_intent"]),
                )
            )
        except (KeyError, TypeError) as exc:
            logger.warning("Skipping malformed golden case #%d: %s", index, exc)
    return cases


async def run_intent_eval(
    detector: IntentDetector,
    cases: list[GoldenCase] | None = None,
) -> EvalReport:
    """Score a detector over the golden set (or a provided subset)."""
    if cases is None:
        cases = load_golden_cases()

    report = EvalReport(total=len(cases))
    for case in cases:
        result = detector.detect_with_confidence(case.query)
        # All detector implementations are async now; the shim keeps
        # plain (sync) test doubles working without AsyncMock.
        if inspect.isawaitable(result):
            result = await result
        predicted = result.intent.value
        passed = predicted == case.expect_intent
        route_passed = route_bucket(predicted) == route_bucket(case.expect_intent)
        report.passed += int(passed)
        report.route_passed += int(route_passed)

        stats = report.by_intent.setdefault(case.expect_intent, {"total": 0, "passed": 0})
        stats["total"] += 1
        stats["passed"] += int(passed)

        if not passed:
            report.failures.append(
                CaseResult(
                    case=case,
                    predicted_intent=predicted,
                    passed=passed,
                    route_passed=route_passed,
                )
            )
    return report
