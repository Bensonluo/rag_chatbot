"""CI quality gate: golden-set intent accuracy (industry eval practice).

Runs the deterministic rule-based detector over the bundled golden set
(no LLM, no network) and fails the build when routing quality regresses.
Headroom: the gate is set below the current 1.0 so adding hard cases to
the golden set does not break CI before rules catch up.
"""

import pytest

from app.services.evaluation import load_golden_cases, run_intent_eval
from app.services.intent.rule_based import RuleBasedIntentDetector

# Every intent family in the set must have at least this many cases —
# a single-case family cannot distinguish accuracy from luck.
MIN_CASES_PER_INTENT = 2
INTENT_ACCURACY_GATE = 0.9
ROUTE_ACCURACY_GATE = 0.95


class TestGoldenIntentGate:
    async def test_rule_detector_meets_accuracy_gate(self):
        report = await run_intent_eval(RuleBasedIntentDetector())

        assert report.total >= 40, "golden set unexpectedly small — was the file loaded?"
        assert report.accuracy >= INTENT_ACCURACY_GATE, [f.case.id for f in report.failures]
        assert report.route_accuracy >= ROUTE_ACCURACY_GATE, [
            f"{f.case.id}: {f.case.expect_intent} → {f.predicted_intent}"
            for f in report.failures
            if not f.route_passed
        ]

    def test_golden_set_covers_every_intention_family(self):
        cases = load_golden_cases()
        by_intent: dict[str, int] = {}
        for case in cases:
            by_intent[case.expect_intent] = by_intent.get(case.expect_intent, 0) + 1

        assert len(cases) >= 40
        thin = {intent: n for intent, n in by_intent.items() if n < MIN_CASES_PER_INTENT}
        assert not thin, f"under-covered intents: {thin}"

    def test_no_duplicate_case_ids(self):
        cases = load_golden_cases()
        ids = [case.id for case in cases]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize(
        "case_id",
        # The four misroutes the golden set caught when it was introduced —
        # pinned individually so a regression names the exact case.
        [
            "faq-01",  # 怎么退货 → how-to FAQ, not the return action
            "cancel-01",  # 取消订单 → cancel, not order query
            "cancel-04",  # 我要取消刚下的订单 → cancel
            "handoff-02",  # 退款没到账,给我转人工 → handoff outranks refund
        ],
    )
    async def test_pinned_misroute_regressions(self, case_id):
        detector = RuleBasedIntentDetector()
        case = next(c for c in load_golden_cases() if c.id == case_id)

        result = detector.detect_with_confidence(case.query)

        assert result.intent.value == case.expect_intent
