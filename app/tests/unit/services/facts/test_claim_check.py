"""Policy-claim checker: deterministic verification of generated text.

The LLM proposes; this gate verifies (never the reverse). Chinese
duration claims in a response are extracted per clause, anchored to the
message's fact subgraph by topic keywords, subject-restricted (支付宝 vs
银行卡 vs 信用卡), and checked by range entailment against the curated
table. Claims that no eligible fact can back are rewritten to the
grounded statement. Unexecuted-action assertions ("已为您办理退款")
without a tool result are the #1 trust killer in CS bots — they are
flagged unconditionally.
"""

from pathlib import Path

from app.services.facts.claim_check import apply_violations, check_policy_claims
from app.services.facts.fact_store import FactStore, PolicyFact

FACTS_FILE = Path(__file__).parents[5] / "app/services/facts/policy_facts.json"
STORE = FactStore(FACTS_FILE)


def _facts_for(message: str) -> list[PolicyFact]:
    return STORE.subgraph_for(message)


class TestDurationClaims:
    def test_correct_default_subject_claim_passes(self):
        result = check_policy_claims(
            "退款将原路退回，1-3 个工作日到账。", _facts_for("退款多久到账")
        )
        assert result.passed, result.violations

    def test_wrong_duration_is_violation(self):
        result = check_policy_claims("退款将在 10 个工作日内到账。", _facts_for("退款多久到账"))
        assert not result.passed
        violation = result.violations[0]
        assert violation.reason == "numeric_mismatch"
        assert "1-3 个工作日" in violation.grounded_statement

    def test_subject_restriction_catches_cross_channel_slip(self):
        """「银行卡退款 1-3 个工作日」is the alipay number on the bank
        channel — exactly the cross-subject hallucination to catch."""
        result = check_policy_claims(
            "银行卡退款 1-3 个工作日到账。", _facts_for("银行卡退款多久到账")
        )
        assert not result.passed
        assert result.violations[0].reason == "numeric_mismatch"
        assert "3-7 个工作日" in result.violations[0].grounded_statement

    def test_credit_card_range_passes(self):
        result = check_policy_claims(
            "信用卡退款 7-15 个工作日到账。", _facts_for("信用卡退款多久到账")
        )
        assert result.passed, result.violations

    def test_point_claim_within_range_passes(self):
        result = check_policy_claims("大约 2 个工作日到账。", _facts_for("退款多久到账"))
        assert result.passed, result.violations

    def test_unit_mismatch_is_not_checked(self):
        """工作日 facts cannot disprove a 天 claim — conservative skip."""
        result = check_policy_claims("退款 3 天内到账。", _facts_for("退款多久到账"))
        assert result.passed, result.violations


class TestShippingSubjectSplit:
    def test_canonical_faq_sentence_passes(self):
        """The FAQ's own sentence mixes two subjects across comma
        clauses — clause-level checking must let each number pair with
        its own subject."""
        answer = "普通订单付款后 48 小时内发货，大促期间可能延迟至 72 小时。"
        result = check_policy_claims(answer, _facts_for("多久发货"))
        assert result.passed, result.violations

    def test_peak_shipping_claim_needs_peak_subject(self):
        result = check_policy_claims("大促期间 72 小时内发货。", _facts_for("多久发货"))
        assert result.passed, result.violations

    def test_unqualified_72h_promise_is_violation(self):
        result = check_policy_claims("您的订单 72 小时内发货。", _facts_for("多久发货"))
        assert not result.passed
        assert result.violations[0].reason == "numeric_mismatch"
        assert "48 小时" in result.violations[0].grounded_statement


class TestWindowClaims:
    def test_inflated_return_window_is_violation(self):
        result = check_policy_claims("我们支持 15 天无理由退货。", _facts_for("怎么退货"))
        assert not result.passed
        assert "7 天" in result.violations[0].grounded_statement

    def test_correct_return_window_passes(self):
        result = check_policy_claims("签收后 7 天内可无理由退货。", _facts_for("怎么退货"))
        assert result.passed, result.violations

    def test_exchange_window_is_separate_topic(self):
        """15 天换货 and 7 天退货 coexist — the checker must not apply
        the return window to a 换货 claim."""
        result = check_policy_claims("签收后 15 天内可以换货。", _facts_for("换货政策"))
        assert result.passed, result.violations


class TestUnexecutedActions:
    def test_completed_action_without_tool_result_is_violation(self):
        result = check_policy_claims("已为您办理退款，请耐心等待。", _facts_for("我要退款"))
        assert not result.passed
        assert result.violations[0].reason == "unexecuted_action"

    def test_future_action_promise_is_not_flagged(self):
        result = check_policy_claims("确认后将为您办理退款。", _facts_for("我要退款"))
        assert result.passed, result.violations

    def test_return_and_cancel_actions_covered(self):
        for text in ("已经为您提交退货申请。", "已为您操作了取消订单。"):
            result = check_policy_claims(text, _facts_for("退货"))
            assert not result.passed, text


class TestConservativeBoundaries:
    def test_plain_answer_passes(self):
        result = check_policy_claims("您可以在订单页面查看物流信息。", _facts_for("退货"))
        assert result.passed

    def test_no_facts_numeric_claims_pass_action_still_flagged(self):
        """Numeric checks need facts (empty subgraph → conservative
        pass); the unexecuted-action assertion is fact-independent and
        still flagged — the wiring layer, not this function, decides
        when the subgraph is too empty to gate at all."""
        assert check_policy_claims("退款将在 10 个工作日内到账。", []).passed
        result = check_policy_claims("已为您办理退款。", [])
        assert not result.passed
        assert result.violations[0].reason == "unexecuted_action"

    def test_claims_without_eligible_topic_pass(self):
        """A duration claim about something the table does not cover is
        not checkable — conservative pass (cannot disprove)."""
        result = check_policy_claims("会员积分将在 5 个工作日内发放。", _facts_for("我要退款"))
        assert result.passed


class TestSmokeRegressions:
    """Pinned from pre-ship smoke runs on LLM-style outputs."""

    def test_retreat_wording_binds_refund_arrival(self):
        """「5 个工作日内退回」uses 退回, not 到账 — the arrival topic
        must still bind (regression: keyword gap let 5 工作日 pass)."""
        text = "您好！关于您的退款，款项将在 5 个工作日内退回您的账户，请您留意查收哦～"
        result = check_policy_claims(text, _facts_for("退款多久到账"))
        assert not result.passed
        assert "1-3 个工作日" in result.violations[0].grounded_statement

    def test_claim_in_later_clause_binds_via_sentence_fallback(self):
        """LLMs front-load context ("已经安排发货啦，24 小时内发出") —
        the number clause carries no keyword, so binding falls back to
        the sentence; subject pairing stays at clause level."""
        result = check_policy_claims(
            "亲，您的订单已经安排发货啦，24 小时内就会发出！", _facts_for("多久发货")
        )
        assert not result.passed
        assert "48 小时" in result.violations[0].grounded_statement

    def test_sentence_fallback_keeps_later_clause_peak_subject(self):
        """The fallback binds the topic but must not mispair subjects:
        大促 in the number's own clause still selects the peak fact."""
        text = "普通订单正常发货，大促期间可能延迟至 72 小时。"
        result = check_policy_claims(text, _facts_for("多久发货"))
        assert result.passed, result.violations


class TestMoneyClaims:
    """Money claims (运费险 caps, auto-refund thresholds, 运费 ranges) —
    the Air Canada failure mode: a hallucinated compensation amount is a
    legal liability the company must honor, not a wording quirk.

    Cap semantics are one-sided by design: only over-promising (hi above
    the cap) is a violation — under-claiming a cap is conservative and
    safe, so it passes."""

    def test_inflated_insurance_payout_is_violation(self):
        result = check_policy_claims("运费险最高赔付 80 元。", _facts_for("运费险能赔多少"))
        assert not result.passed
        assert result.violations[0].reason == "numeric_mismatch"
        assert "25 元" in result.violations[0].grounded_statement

    def test_insurance_payout_within_cap_passes(self):
        assert check_policy_claims("运费险最高赔付 25 元。", _facts_for("运费险")).passed
        assert check_policy_claims("运费险赔付 20 元。", _facts_for("运费险")).passed

    def test_insurance_decimal_claim_checked(self):
        assert check_policy_claims("运费险赔付 24.5 元。", _facts_for("运费险")).passed
        assert not check_policy_claims("运费险赔付 30.5 元。", _facts_for("运费险")).passed

    def test_auto_refund_cap_enforced(self):
        """The 1000 元 auto-refund cap sat uncheckable (no 元 regex, no
        topic keywords) until money claims became visible to the gate."""
        result = check_policy_claims("5000 元以内可自动退款。", _facts_for("退款"))
        assert not result.passed
        assert "1000 元" in result.violations[0].grounded_statement

    def test_auto_refund_within_cap_passes(self):
        assert check_policy_claims("1000 元以内的退款可自动处理。", _facts_for("退款")).passed

    def test_shipping_fee_range_enforced(self):
        result = check_policy_claims("未满额订单收取 15 元运费。", _facts_for("运费多少钱"))
        assert not result.passed
        assert "6-12 元" in result.violations[0].grounded_statement

    def test_in_range_shipping_fee_passes(self):
        assert check_policy_claims("未满额收取 8 元运费。", _facts_for("运费")).passed

    def test_free_shipping_threshold_enforced(self):
        assert check_policy_claims("订单满 99 元包邮。", _facts_for("运费多少钱")).passed
        result = check_policy_claims("订单满 59 元就包邮了。", _facts_for("运费多少钱"))
        assert not result.passed
        assert "99 元" in result.violations[0].grounded_statement

    def test_order_amount_is_not_a_policy_claim(self):
        """Order-specific amounts name no money topic — conservative
        pass (the table cannot disprove a concrete order amount)."""
        assert check_policy_claims("您的订单金额为 200 元。", _facts_for("运费险")).passed


class TestApplyViolations:
    def test_violating_clause_is_replaced_and_rest_kept(self):
        text = "您好。退款将在 10 个工作日内到账。请问还有其他问题吗？"
        result = check_policy_claims(text, _facts_for("退款多久到账"))
        rewritten = apply_violations(text, result)
        assert "10 个工作日" not in rewritten
        assert "1-3 个工作日" in rewritten
        assert rewritten.startswith("您好。")
        assert rewritten.endswith("请问还有其他问题吗？")

    def test_clean_text_untouched(self):
        text = "退款将原路退回，1-3 个工作日到账。"
        result = check_policy_claims(text, _facts_for("退款多久到账"))
        assert apply_violations(text, result) == text


class TestCouponClaims:
    """Coupon/满减 claims (A2 wrap-up): 中消协 2026 上半年投诉热点正是
    AI 客服对优惠活动作出明确承诺后经营者拒绝履约——优惠券数字是
    Air-Canada 同款法律责任方向，拿到确定性门内。京东式满减券语义：
    门槛（满 X 可用）+ 面额（最高减 Y）+ 有效期。"""

    def test_understated_threshold_violates_and_grounds(self):
        text = "满 200 元就可以使用这张满减券。"
        result = check_policy_claims(text, _facts_for("满减券怎么用"))
        assert not result.passed
        rewritten = apply_violations(text, result)
        assert "满 300 元" in rewritten

    def test_exact_threshold_passes(self):
        result = check_policy_claims("满 300 元即可使用该满减券。", _facts_for("满减券怎么用"))
        assert result.passed

    def test_inflated_threshold_also_corrected(self):
        """Saying 满 350 when the threshold is 300 misleads the customer
        into overspending — money_point semantics demand the exact number."""
        text = "需要满 350 元才能用这张满减券。"
        result = check_policy_claims(text, _facts_for("满减券怎么用"))
        assert not result.passed
        rewritten = apply_violations(text, result)
        assert "满 300 元" in rewritten

    def test_discount_over_cap_violates_and_grounds(self):
        text = "这张满减券下单立减 80 元。"
        result = check_policy_claims(text, _facts_for("满减券怎么用"))
        assert not result.passed
        rewritten = apply_violations(text, result)
        assert "50 元" in rewritten

    def test_discount_within_cap_passes(self):
        result = check_policy_claims("这张满减券最高减 50 元。", _facts_for("满减券怎么用"))
        assert result.passed

    def test_discount_under_cap_is_conservative_pass(self):
        result = check_policy_claims("这张满减券可以给您减免 30 元。", _facts_for("满减券怎么用"))
        assert result.passed

    def test_inflated_validity_violates_and_grounds(self):
        text = "优惠券有效期长达 90 天。"
        result = check_policy_claims(text, _facts_for("满减券怎么用"))
        assert not result.passed
        rewritten = apply_violations(text, result)
        assert "30 天" in rewritten

    def test_exact_validity_passes(self):
        result = check_policy_claims("优惠券领取后 30 天内有效。", _facts_for("满减券怎么用"))
        assert result.passed

    def test_non_coupon_amounts_untouched(self):
        """Order totals never bind to coupon topics."""
        result = check_policy_claims(
            "您的订单共 200 元，退款已原路退回。", _facts_for("订单多少钱")
        )
        assert result.passed
