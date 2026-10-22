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

import pytest

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
        assert report.uncovered_fact_ids == []
        assert {fact_id for result in report.results for fact_id in result.covered_fact_ids} == {
            fact.id for fact in FactStore(FACTS_FILE).facts if fact.value
        }

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


class TestFactCoverage:
    @pytest.mark.parametrize("anchors", [["退款"], ["未被现有探针提及的新渠道"]])
    def test_new_subject_in_covered_topic_is_not_covered(
        self,
        tmp_path: Path,
        anchors: list[str],
    ) -> None:
        facts = json.loads(FACTS_FILE.read_text(encoding="utf-8"))
        facts.append(
            {
                "id": "refund_arrival_new_channel",
                "topic": "refund_arrival",
                "subject": "新渠道",
                "anchors": anchors,
                "topic_keywords": ["到账"],
                "subject_keywords": ["新渠道"],
                "default_subject": False,
                "kind": "duration_range",
                "value": "50-60",
                "unit": "工作日",
                "statement": "新渠道退款50-60个工作日到账。",
                "faq_id": None,
            }
        )
        path = tmp_path / "facts.json"
        path.write_text(json.dumps(facts), encoding="utf-8")
        report = run_claim_eval(FactStore(path))
        assert not report.passed_all
        assert report.uncovered_fact_ids == ["refund_arrival_new_channel"]
        assert report.uncovered_topics == ["refund_arrival"]

    def test_grounding_all_variants_does_not_credit_unchecked_subjects(self) -> None:
        case = next(c for c in load_claim_cases() if c.id == "cg-refund-arrival-default")
        report = run_claim_eval(FactStore(FACTS_FILE), [case])
        assert report.results[0].covered_fact_ids == ["refund_arrival_alipay_wechat"]
        assert "refund_arrival_bank" in report.uncovered_fact_ids
        assert "refund_arrival_credit" in report.uncovered_fact_ids

    def test_declared_topic_without_retrieved_facts_is_not_coverage(self) -> None:
        case = ClaimCase(
            id="unretrieved",
            topic="return_window",
            query="你好",
            canonical="签收后7天内可退货。",
            adversarial="签收后15天内可退货。",
            grounded_contains="7 天",
        )
        report = run_claim_eval(FactStore(FACTS_FILE), [case])
        assert "return_window" in report.uncovered_topics
        assert report.results[0].covered_fact_ids == []

    def test_unexecuted_action_cannot_count_as_numeric_fact_coverage(self) -> None:
        from app.services.facts.claim_check import ACTION_FALLBACK

        case = ClaimCase(
            id="action-only",
            topic="refund_arrival",
            query="退款",
            canonical="退款请查看订单。",
            adversarial="已经为您办理退款。",
            grounded_contains=ACTION_FALLBACK,
        )
        report = run_claim_eval(FactStore(FACTS_FILE), [case])
        assert not report.results[0].passed
        assert report.results[0].covered_fact_ids == []

    def test_matching_default_range_does_not_credit_a_named_peer_subject(self) -> None:
        # Three days happens to fit both channels. Removing the peer fact for
        # an isolated probe must not change which subject the full gate chose.
        case = ClaimCase(
            id="bank-shared-boundary",
            topic="refund_arrival",
            query="银行卡退款",
            canonical="银行卡退款3个工作日到账。",
            adversarial="银行卡退款30个工作日到账。",
            grounded_contains="3-7 个工作日",
        )
        result = run_claim_eval(FactStore(FACTS_FILE), [case]).results[0]
        assert result.covered_fact_ids == ["refund_arrival_bank"]

    def test_violation_in_another_fact_does_not_credit_the_declared_topic(self) -> None:
        case = ClaimCase(
            id="wrong-fact",
            topic="refund_arrival",
            query="退款和运费险",
            canonical="退款1-3个工作日到账。运费险最高赔付25元。",
            adversarial="退款1-3个工作日到账。运费险最高赔付80元。",
            grounded_contains="25 元",
        )
        result = run_claim_eval(FactStore(FACTS_FILE), [case]).results[0]
        assert result.adversarial_caught
        assert result.grounded_carried
        assert result.covered_fact_ids == []
        assert not result.passed

    def test_other_unit_in_same_clause_cannot_credit_a_valid_money_claim(self) -> None:
        case = ClaimCase(
            id="mixed-units",
            topic="coupon_usage",
            query="优惠券",
            canonical="优惠券最高减30元。",
            adversarial="满减券满300元可用且有效期60天。",
            grounded_contains="30 天",
        )
        result = run_claim_eval(FactStore(FACTS_FILE), [case]).results[0]
        assert result.covered_fact_ids == ["coupon_validity"]

    def test_empty_fact_and_case_sets_cannot_pass(self, tmp_path: Path) -> None:
        path = tmp_path / "empty-facts.json"
        path.write_text("[]", encoding="utf-8")
        assert not run_claim_eval(FactStore(path), []).passed_all


class TestStrictClaimLoading:
    @staticmethod
    def valid_case() -> dict[str, str]:
        return {
            "id": "case",
            "topic": "return_window",
            "query": "退货",
            "canonical": "7天退货",
            "adversarial": "15天退货",
            "grounded_contains": "7 天",
        }

    @pytest.mark.parametrize("payload", [[], {}, None, "invalid", [None], ["invalid"]])
    def test_rejects_invalid_root_or_entry(self, tmp_path: Path, payload: object) -> None:
        path = tmp_path / "cases.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            load_claim_cases(path)

    @pytest.mark.parametrize(
        "field", ["id", "topic", "query", "canonical", "adversarial", "grounded_contains"]
    )
    @pytest.mark.parametrize("value", [None, 123, [], "", "   "])
    def test_rejects_malformed_case_instead_of_skipping(
        self,
        tmp_path: Path,
        field: str,
        value: object,
    ) -> None:
        invalid: dict[str, object] = dict(self.valid_case())
        invalid["id"] = "invalid"
        invalid[field] = value
        path = tmp_path / "cases.json"
        path.write_text(json.dumps([self.valid_case(), invalid]), encoding="utf-8")
        with pytest.raises(ValueError):
            load_claim_cases(path)

    def test_rejects_missing_field(self, tmp_path: Path) -> None:
        invalid = self.valid_case()
        del invalid["query"]
        path = tmp_path / "cases.json"
        path.write_text(json.dumps([self.valid_case(), invalid]), encoding="utf-8")
        with pytest.raises(ValueError):
            load_claim_cases(path)

    def test_rejects_duplicate_case_ids(self, tmp_path: Path) -> None:
        path = tmp_path / "cases.json"
        path.write_text(json.dumps([self.valid_case(), self.valid_case()]), encoding="utf-8")
        with pytest.raises(ValueError):
            load_claim_cases(path)

    def test_missing_file_is_a_configuration_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            load_claim_cases(tmp_path / "missing.json")

    @pytest.mark.parametrize("content", ["[", '{"id":"first","id":"second"}'])
    def test_rejects_malformed_json_or_duplicate_object_keys(
        self,
        tmp_path: Path,
        content: str,
    ) -> None:
        path = tmp_path / "cases.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError):
            load_claim_cases(path)

    def test_rejects_unknown_fields(self, tmp_path: Path) -> None:
        entry = {**self.valid_case(), "fact_id": "pretended-coverage"}
        path = tmp_path / "cases.json"
        path.write_text(json.dumps([entry]), encoding="utf-8")
        with pytest.raises(ValueError):
            load_claim_cases(path)

    def test_explicit_cases_cannot_bypass_duplicate_validation(self) -> None:
        case = ClaimCase(**self.valid_case())
        with pytest.raises(ValueError):
            run_claim_eval(cases=[case, case])
