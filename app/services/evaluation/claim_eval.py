"""Claim-gate golden eval (Phase E Tier 1 — deterministic, keyless).

Deterministic checks run without model keys; semantic model grading is separate:

1. Intent-routing golden (``runner.py``) — gates routing, keyless.
2. **This module** — gates the claim gate itself, keyless: every
   checkable policy fact carries a canonical response the gate must
   pass through untouched (over-blocking guard) and an adversarial
   response whose hallucinated number must be rewritten to the
   grounded statement (under-blocking guard). Deterministic assertions have no judge noise,
   so both floors are 1.0.
3. Answer golden (``answer_eval.py``) measures given-context point coverage.
   ``pipeline_eval.py`` checks isolated production-graph artifacts;
   ``groundedness_eval.py`` separately grades observed evidence and actions.

A coverage contract closes the loop with the fact table: any fact with
a value but no golden case is an unguarded policy surface and fails
the report, so adding facts without probes is a reviewable event.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.services.facts.claim_check import apply_violations, check_policy_claims
from app.services.facts.fact_store import FactStore, PolicyFact

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
    covered_fact_ids: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.canonical_untouched
            and self.adversarial_caught
            and self.grounded_carried
            and bool(self.covered_fact_ids)
        )


@dataclass(frozen=True)
class ClaimEvalReport:
    results: list[ClaimCaseResult]
    uncovered_topics: list[str]
    uncovered_fact_ids: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[ClaimCaseResult]:
        return [r for r in self.results if not r.passed]

    @property
    def passed_all(self) -> bool:
        return (
            bool(self.results)
            and not self.failures
            and not self.uncovered_topics
            and not self.uncovered_fact_ids
        )


_CASE_FIELDS = ("id", "topic", "query", "canonical", "adversarial", "grounded_contains")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate claim golden field: {key}")
        result[key] = value
    return result


def _validate_cases(cases: list[ClaimCase]) -> None:
    seen: set[str] = set()
    for index, case in enumerate(cases):
        for name in _CASE_FIELDS:
            value = getattr(case, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Claim golden case #{index}: {name} must be a nonempty string")
        identity = case.id.strip()
        if identity in seen:
            raise ValueError(f"Duplicate claim golden case id: {identity}")
        seen.add(identity)


def load_claim_cases(cases_file: str | Path = CLAIM_GOLDEN_SET_FILE) -> list[ClaimCase]:
    """Load the complete contract or fail; malformed probes must not disappear."""
    path = Path(cases_file)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Cannot load claim golden set: {path}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("Claim golden set must be a nonempty list")
    cases: list[ClaimCase] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict) or set(entry) != set(_CASE_FIELDS):
            raise ValueError(f"Claim golden case #{index} must contain exactly the required fields")
        cases.append(ClaimCase(**entry))
    _validate_cases(cases)
    return cases


def run_claim_eval(
    store: FactStore | None = None,
    cases: list[ClaimCase] | None = None,
) -> ClaimEvalReport:
    """Probe the shipped gate and report facts with no verified numeric probe."""
    store = store or FactStore.load_default()
    cases = cases if cases is not None else load_claim_cases()

    _validate_cases(cases)
    results = [_run_case(case, store) for case in cases]
    covered = {fact_id for result in results for fact_id in result.covered_fact_ids}
    uncovered = [fact for fact in store.facts if _checkable(fact) and fact.id not in covered]
    return ClaimEvalReport(
        results=results,
        uncovered_topics=sorted({fact.topic for fact in uncovered}),
        uncovered_fact_ids=sorted({fact.id for fact in uncovered}),
    )


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

    covered_fact_ids: list[str] = []
    if canonical_untouched and adversarial_caught and grounded_carried:
        # The production correction quotes all subjects in a topic, even when
        # only one was checked. Quoted statements and declared topics alone
        # cannot establish coverage. Re-run the same checker on each retrieved
        # fact and require a numeric violation of the SAME clause, as well as
        # a passing canonical response. Action violations never establish it.
        unit_violations = {
            unit: {
                violation.clause
                for violation in check_policy_claims(
                    case.adversarial,
                    [fact for fact in facts if fact.unit == unit],
                    check_actions=False,
                ).violations
                if violation.reason == "numeric_mismatch"
            }
            for unit in {fact.unit for fact in facts if _checkable(fact)}
        }
        for fact in facts:
            if fact.topic != case.topic or not _checkable(fact) or not fact.statement:
                continue
            canonical_probe = check_policy_claims(case.canonical, [fact], check_actions=False)
            if not canonical_probe.passed:
                continue
            adversarial_probe = check_policy_claims(case.adversarial, [fact], check_actions=False)
            if any(
                single.reason == full.reason == "numeric_mismatch"
                and single.clause == full.clause
                and single.clause in unit_violations[fact.unit]
                and _subject_selected(fact, single.clause, facts)
                and fact.statement in full.grounded_statement
                for single in adversarial_probe.violations
                for full in adversarial_result.violations
            ):
                covered_fact_ids.append(fact.id)

    return ClaimCaseResult(
        case=case,
        canonical_untouched=canonical_untouched,
        adversarial_caught=adversarial_caught,
        grounded_carried=grounded_carried,
        covered_fact_ids=sorted(set(covered_fact_ids)),
    )


def _subject_selected(fact: PolicyFact, clause: str, facts: list[PolicyFact]) -> bool:
    """Retain peer-subject restrictions lost when a fact is probed in isolation.

    Without this check, a bank claim could receive credit for the default
    Alipay fact because that fact becomes the fallback in its singleton probe.
    Be conservative: any named peer excludes an unnamed default.
    """
    named = [
        peer
        for peer in facts
        if peer.topic == fact.topic
        and peer.unit == fact.unit
        and _checkable(peer)
        and any(keyword in clause for keyword in peer.subject_keywords)
    ]
    return fact in named if named else fact.default_subject


def _checkable(fact: PolicyFact) -> bool:
    """A fact is checkable policy surface when it carries a value."""
    return bool(fact.value)
