"""Claim-gate golden eval (Phase E Tier 1 — deterministic, keyless).

The evaluation stack now has three tiers, matching the industry's
cost-tiered consensus (golden set 50-200 cases; free deterministic
checks in CI, LLM-judge tiers on demand):

1. Intent-routing golden (``runner.py``) — gates routing, keyless.
2. **This module** — gates the claim gate itself, keyless: every
   checkable policy topic carries a canonical response the gate must
   pass through untouched (over-blocking guard) and an adversarial
   response whose hallucinated number must be rewritten to the
   grounded statement (under-blocking guard — the Air Canada
   liability direction). Deterministic assertions have no judge noise,
   so both floors are 1.0.
3. Answer golden (``answer_eval.py``) — LLM-judge answer quality on
   demand.

A coverage contract closes the loop with the fact table: any fact with
a value but no golden case is an unguarded policy surface and fails
the report, so adding facts without probes is a reviewable event.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.facts.claim_check import apply_violations, check_policy_claims
from app.services.facts.fact_store import FactStore, PolicyFact

logger = logging.getLogger(__name__)

CLAIM_GOLDEN_SET_FILE = Path(__file__).parent / "claim_golden_set.json"


@dataclass(frozen=True)
class ClaimCase:
    """One policy topic probed from both directions.

    ``canonical`` is a correct reply the gate must leave untouched;
    ``adversarial`` carries a hallucinated number the gate must rewrite
    to a statement containing ``grounded_contains``.
    """

    id: str
    topic: str
    query: str
    canonical: str
    adversarial: str
    grounded_contains: str


@dataclass(frozen=True)
class ClaimCaseResult:
    """Per-direction outcome for one case — attribution shows which
    direction of the gate regressed, not just that it failed."""

    case: ClaimCase
    canonical_untouched: bool
    adversarial_caught: bool
    grounded_carried: bool

    @property
    def passed(self) -> bool:
        return self.canonical_untouched and self.adversarial_caught and self.grounded_carried


@dataclass(frozen=True)
class ClaimEvalReport:
    results: list[ClaimCaseResult]
    uncovered_topics: list[str]

    @property
    def failures(self) -> list[ClaimCaseResult]:
        return [r for r in self.results if not r.passed]

    @property
    def passed_all(self) -> bool:
        return not self.failures and not self.uncovered_topics


def load_claim_cases(cases_file: str | Path = CLAIM_GOLDEN_SET_FILE) -> list[ClaimCase]:
    """Load the versioned claim-gate contract; unreadable → empty (an
    empty set then fails the coverage contract loudly, never silently)."""
    path = Path(cases_file)
    try:
        raw: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Claim golden set unreadable (%s); claim eval inactive", exc)
        return []
    cases: list[ClaimCase] = []
    for entry in raw:
        try:
            cases.append(
                ClaimCase(
                    id=str(entry["id"]),
                    topic=str(entry["topic"]),
                    query=str(entry["query"]),
                    canonical=str(entry["canonical"]),
                    adversarial=str(entry["adversarial"]),
                    grounded_contains=str(entry.get("grounded_contains", "")),
                )
            )
        except (KeyError, TypeError) as exc:
            logger.warning("Skipping malformed claim golden case: %s", exc)
    return cases


def run_claim_eval(
    store: FactStore | None = None,
    cases: list[ClaimCase] | None = None,
) -> ClaimEvalReport:
    """Probe every case through the shipped gate; report per-direction
    results plus any checkable fact topic the set fails to cover."""
    store = store or FactStore.load_default()
    cases = cases if cases is not None else load_claim_cases()

    results = [_run_case(case, store) for case in cases]
    covered = {case.topic for case in cases}
    uncovered = sorted({fact.topic for fact in store.facts if _checkable(fact)} - covered)
    return ClaimEvalReport(results=results, uncovered_topics=uncovered)


def _run_case(case: ClaimCase, store: FactStore) -> ClaimCaseResult:
    facts = store.subgraph_for(case.query)

    canonical_result = check_policy_claims(case.canonical, facts)
    canonical_untouched = (
        canonical_result.passed
        and apply_violations(case.canonical, canonical_result) == case.canonical
    )

    adversarial_result = check_policy_claims(case.adversarial, facts)
    adversarial_caught = not adversarial_result.passed
    rewritten = apply_violations(case.adversarial, adversarial_result)
    grounded_carried = bool(case.grounded_contains) and case.grounded_contains in rewritten

    return ClaimCaseResult(
        case=case,
        canonical_untouched=canonical_untouched,
        adversarial_caught=adversarial_caught,
        grounded_carried=grounded_carried,
    )


def _checkable(fact: PolicyFact) -> bool:
    """A fact is checkable policy surface when it carries a value."""
    return bool(fact.value)
