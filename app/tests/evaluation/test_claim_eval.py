"""Claim-gate golden set (Phase E Tier 1 — deterministic, keyless CI).

The intent golden set gates routing; the answer golden set gates
generation with an LLM judge. This tier gates the claim gate itself:
every checkable policy topic must carry (a) a canonical response the
gate passes through untouched (over-blocking regression) and (b) an
adversarial response with a hallucinated number the gate must rewrite
to the grounded statement (under-blocking regression). Both directions
matter at 800K-1M requests/day: an over-eager gate mangles correct
replies at scale, a sleepy gate ships Air-Canada-style liabilities.
"""

import json
from pathlib import Path

from app.services.evaluation.claim_eval import (
    CLAIM_GOLDEN_SET_FILE,
    ClaimCase,
    load_claim_cases,
    run_claim_eval,
)
from app.services.facts.fact_store import FactStore

FACTS_FILE = Path(__file__).parents[2] / "services/facts/policy_facts.json"


class TestSetIntegrity:
    def test_loads_versioned_contract(self):
        cases = load_claim_cases()
        assert len(cases) >= 15  # one per checkable topic + subject variants

    def test_case_ids_unique(self):
        cases = load_claim_cases()
        ids = [c.id for c in cases]
        assert len(ids) == len(set(ids))

    def test_every_case_fully_specified(self):
        for case in load_claim_cases():
            assert case.query.strip(), case.id
            assert case.canonical.strip(), case.id
            assert case.adversarial.strip(), case.id
            assert case.grounded_contains.strip(), case.id

    def test_file_is_valid_json_with_expected_shape(self):
        raw = json.loads(CLAIM_GOLDEN_SET_FILE.read_text(encoding="utf-8"))
        assert isinstance(raw, list) and raw
        for entry in raw:
            assert {"id", "topic", "query", "canonical", "adversarial"} <= set(entry)


class TestCoverageContract:
    def test_every_checkable_fact_topic_is_covered(self):
        """A fact with a value but no golden case is an unguarded policy
        surface — adding facts without probes must fail the eval."""
        report = run_claim_eval(FactStore(FACTS_FILE))
        assert report.uncovered_topics == []

    def test_coverage_gap_is_detected(self, tmp_path: Path) -> None:
        facts = json.loads(FACTS_FILE.read_text(encoding="utf-8"))
        facts.append(
            {
                "id": "mystery_cap",
                "topic": "mystery_topic",
                "subject": "未知",
                "anchors": ["神秘"],
                "topic_keywords": ["神秘"],
                "subject_keywords": [],
                "default_subject": True,
                "kind": "cap",
                "value": "50",
                "unit": "元",
                "statement": "神秘业务最高 50 元。",
                "faq_id": None,
            }
        )
        extra = tmp_path / "extra_facts.json"
        extra.write_text(json.dumps(facts), encoding="utf-8")

        report = run_claim_eval(FactStore(extra))

        assert report.uncovered_topics == ["mystery_topic"]


class TestGateContract:
    def test_all_cases_pass_both_directions(self):
        report = run_claim_eval(FactStore(FACTS_FILE))
        assert report.passed_all, [
            (r.case.id, r.canonical_untouched, r.adversarial_caught, r.grounded_carried)
            for r in report.failures
        ]

    def test_canonical_responses_pass_untouched(self):
        """Over-blocking regression guard: apply_violations must be the
        identity on every canonical response."""
        report = run_claim_eval(FactStore(FACTS_FILE))
        assert all(r.canonical_untouched for r in report.results)

    def test_adversarial_always_rewritten_with_grounding(self):
        report = run_claim_eval(FactStore(FACTS_FILE))
        assert all(r.adversarial_caught for r in report.results)
        assert all(r.grounded_carried for r in report.results)

    def test_runner_can_fail(self, tmp_path: Path) -> None:
        """The gate must be able to fail: a case whose grounded marker the
        table cannot produce shows up as a failure, not a silent pass."""
        broken = ClaimCase(
            id="broken",
            topic="return_window",
            query="怎么退货",
            canonical="签收后 7 天内可无理由退货。",
            adversarial="我们支持 15 天无理由退货。",
            grounded_contains="999 个工作日",  # nothing grounds this
        )
        broken_file = tmp_path / "broken.json"
        broken_file.write_text(
            json.dumps(
                [
                    {
                        "id": broken.id,
                        "topic": broken.topic,
                        "query": broken.query,
                        "canonical": broken.canonical,
                        "adversarial": broken.adversarial,
                        "grounded_contains": broken.grounded_contains,
                    }
                ]
            ),
            encoding="utf-8",
        )

        report = run_claim_eval(FactStore(FACTS_FILE), cases=load_claim_cases(broken_file))

        assert not report.passed_all
        assert [r.case.id for r in report.failures] == ["broken"]
        # The failing result still caught the adversarial text — only the
        # grounding marker missed — proving per-direction attribution.
        failure = report.failures[0]
        assert failure.adversarial_caught
        assert not failure.grounded_carried
