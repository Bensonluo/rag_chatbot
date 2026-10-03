"""Tests for rule-based intent detector"""

import pytest

from app.models.enums.intent import Intent


class TestRuleBasedIntentDetector:
    """Test rule-based intent detector"""

    async def test_initialization(self):
        """Test detector initialization"""
        # Arrange & Act
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Assert
        assert detector is not None
        assert hasattr(detector, "rules")

    async def test_detect_refund_keyword(self):
        """Test detecting refund intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("我要退款")

        # Assert
        assert intent == Intent.REFUND

    async def test_detect_refund_english(self):
        """Test detecting refund intent from English keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("I want a refund")

        # Assert
        assert intent == Intent.REFUND

    async def test_detect_return_keyword(self):
        """Test detecting return intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("我要退货")

        # Assert
        assert intent == Intent.RETURN

    async def test_detect_return_exchange(self):
        """Test detecting return intent from exchange keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("我想换货")

        # Assert
        assert intent == Intent.RETURN

    async def test_detect_query_order_keyword(self):
        """Test detecting query_order intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("查一下订单")

        # Assert
        assert intent == Intent.QUERY_ORDER

    async def test_detect_query_order_status(self):
        """Test detecting query_order intent from status pattern"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("我的订单状态是什么")

        # Assert
        assert intent == Intent.QUERY_ORDER

    async def test_detect_track_shipping_keyword(self):
        """Test detecting track_shipping intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("快递到哪了")

        # Assert
        assert intent == Intent.TRACK_SHIPPING

    async def test_detect_track_shipping_logistics(self):
        """Test detecting track_shipping intent from logistics keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("物流查询")

        # Assert
        assert intent == Intent.TRACK_SHIPPING

    async def test_detect_complaint_keyword(self):
        """Test detecting complaint intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("我要投诉")

        # Assert
        assert intent == Intent.COMPLAINT

    async def test_detect_policy_keyword(self):
        """Test detecting policy intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("退货政策是什么")

        # Assert
        assert intent == Intent.POLICY

    async def test_detect_policy_english(self):
        """English policy questions must hit POLICY from rules alone —
        the live deployment must not depend on an LLM fallback (which
        the free GLM tier serves intermittently) to route them."""
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        intent = await detector.detect("What is your return policy")
        assert intent == Intent.POLICY

        intent = await detector.detect("how does your refund policy work")
        assert intent == Intent.POLICY

        result = await detector.detect_with_confidence("What is your return policy")
        assert result.intent == Intent.POLICY
        assert result.confidence >= 0.7  # rule short-circuit threshold

    async def test_detect_faq_keyword(self):
        """Test detecting FAQ intent from general question keywords"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act - "能不能" is a FAQ keyword that doesn't match task-oriented patterns
        intent = await detector.detect("能不能帮我看看")

        # Assert
        assert intent == Intent.FAQ

    async def test_detect_greeting_keyword(self):
        """Test detecting greeting intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("你好")

        # Assert
        assert intent == Intent.GREETING

    async def test_detect_greeting_english(self):
        """Test detecting greeting intent from English keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("hello")

        # Assert
        assert intent == Intent.GREETING

    async def test_reversed_howto_phrasing_is_faq_not_action(self):
        """Action-then-question phrasing ("退货怎么算") is a knowledge
        question, not a request to execute the action.

        Observed live (2026-09-30): "7天无理由退货怎么算" classified as
        RETURN (keyword 退货 + pattern 退[换货] = 2.7 → 0.9) and the
        bot demanded an order number instead of explaining the policy —
        the zh mirror of the EN "return policy" bug. The FAQ patterns
        only looked for the question word BEFORE the action noun."""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        result = await detector.detect_with_confidence("7天无理由退货怎么算")

        # Assert — FAQ (怎么 keyword 0.5 + reversed pattern 2.4 = 2.9 →
        # 0.97) must outrank RETURN (2.7 → 0.9) and cross the 0.7
        # short-circuit so no LLM call is needed.
        assert result.intent is Intent.FAQ
        assert result.confidence >= 0.7
        # The direct action phrasing must stay an action request.
        assert await detector.detect("我要退货") is Intent.RETURN

    async def test_reversed_timeline_phrasing_is_faq_not_action(self):
        """Timeline phrasing ("退款多久到账") is a knowledge question —
        the shipped FAQ table even carries this exact question with the
        timeline answer.

        Found by the routing golden-set eval (2026-10-03): REFUND's
        keyword (1.5) and regex (1.2) double-fire on the same 退款 token
        (2.7) and outrank the FAQ reversed pattern (2.4) because 多久,
        unlike 怎么, carried no FAQ keyword support — the bot demanded an
        order number for a when-will-my-money-arrive question, the exact
        20201e9 failure family. Timeline keywords join the question-word
        support so informational phrasing keeps winning arbitration."""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        result = await detector.detect_with_confidence("退款多久到账")

        # Assert — FAQ (reversed 2.4 + 多久 0.5 = 2.9 → 0.97) must outrank
        # REFUND (2.7 → 0.9) and cross the 0.7 short-circuit.
        assert result.intent is Intent.FAQ
        assert result.confidence >= 0.7
        # The action phrasings must stay action requests.
        assert await detector.detect("我要退款") is Intent.REFUND
        assert await detector.detect("退款进度") is Intent.REFUND

    async def test_english_word_interior_never_matches_keywords(self):
        """ASCII keywords match whole words, not word interiors.

        Observed live (2026-09-30): "the thing I bought last week
        arrived broken, what are my options" classified as GREETING
        because the keyword "hi" substring-matches inside "thing";
        the same flaw reads "no" out of "know"/"nothing" as DENY."""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        broken_item = await detector.detect_with_confidence(
            "the thing I bought last week arrived broken, what are my options"
        )
        know_nothing = await detector.detect_with_confidence(
            "I don't know my order number, can you help"
        )

        # Assert — no GREETING/DENY rule fires on word interiors; the
        # second query keeps its legitimate QUERY_ORDER signal.
        assert broken_item.intent is not Intent.GREETING
        assert "greeting" not in [
            r["intent"] for r in (broken_item.metadata or {}).get("matched_rules", [])
        ]
        assert know_nothing.intent is Intent.QUERY_ORDER
        assert "deny" not in [
            r["intent"] for r in (know_nothing.metadata or {}).get("matched_rules", [])
        ]

    async def test_english_standalone_short_keyword_still_matches(self):
        """Word-boundary matching must not break real "hi"/"no" hits or
        morphological variants (refunded, cancelled)."""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act / Assert
        assert await detector.detect("hi there!") is Intent.GREETING
        assert await detector.detect("no, that's wrong") is Intent.DENY
        assert await detector.detect("I want to be refunded for order A100") is Intent.REFUND
        assert await detector.detect("the subscription was cancelled without telling me") is (
            Intent.CANCEL
        )

    async def test_detect_chitchat_keyword(self):
        """Test detecting chitchat intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("哈哈")

        # Assert
        assert intent == Intent.CHITCHAT

    async def test_detect_confirm_keyword(self):
        """Test detecting confirm intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("是的")

        # Assert
        assert intent == Intent.CONFIRM

    async def test_detect_deny_keyword(self):
        """Test detecting deny intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("不是")

        # Assert
        assert intent == Intent.DENY

    async def test_detect_cancel_keyword(self):
        """Test detecting cancel intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("取消")

        # Assert
        assert intent == Intent.CANCEL

    async def test_detect_unknown(self):
        """Test detecting unknown intent"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("Xylophone zebra yellow")

        # Assert
        assert intent == Intent.UNKNOWN

    async def test_detect_with_confidence_high(self):
        """Test detecting intent with high confidence"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        result = await detector.detect_with_confidence("我要退款")

        # Assert
        assert result.intent == Intent.REFUND
        assert result.confidence > 0.5
        assert result.confidence <= 1.0

    async def test_detect_with_confidence_low(self):
        """Test detecting intent with low confidence"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        result = await detector.detect_with_confidence("Xylophone zebra yellow")

        # Assert
        assert result.intent == Intent.UNKNOWN
        assert result.confidence < 0.5

    async def test_detect_with_confidence_metadata(self):
        """Test that confidence result includes metadata"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        result = await detector.detect_with_confidence("我要退款")

        # Assert
        assert result.intent == Intent.REFUND
        assert result.confidence > 0
        assert result.metadata is not None
        # Metadata should contain info about matched rules
        assert "matched_rules" in result.metadata

    async def test_case_insensitive(self):
        """Test that detection is case-insensitive"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent1 = await detector.detect("REFUND")
        intent2 = await detector.detect("refund")

        # Assert
        assert intent1 == intent2
        assert intent1 == Intent.REFUND

    async def test_punctuation_handling(self):
        """Test that punctuation is handled correctly"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("你好！！！")

        # Assert
        assert intent == Intent.GREETING

    async def test_empty_query(self):
        """Test handling empty query"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("")

        # Assert
        assert intent == Intent.UNKNOWN

    async def test_multiple_keywords_refund_and_return(self):
        """Test query with both refund and return keywords"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act - Both "退款" and "退货" present
        intent = await detector.detect("我要退款退货")

        # Assert - Should match one intent based on rule priority
        assert intent in [Intent.REFUND, Intent.RETURN]

    async def test_detect_with_context_parameter(self):
        """Test that detect accepts optional context parameter"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = await detector.detect("我要退款", context={"previous_messages": []})

        # Assert
        assert intent == Intent.REFUND

    async def test_detect_track_shipping_pattern(self):
        """Test detecting track_shipping via regex pattern"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act - matches pattern r"(物流|快递|包裹).{0,4}(到|在|哪|状态|查询)"
        intent = await detector.detect("包裹到了吗")

        # Assert
        assert intent == Intent.TRACK_SHIPPING


# The exact message the zh live mirror sent while the intent LLM leg was
# down (GLM 429 storm, 2026-09-30): keyword hits tied COMPLAINT with
# QUERY_ORDER and TRACK_SHIPPING at 1.5 each, and dict-iteration order
# crowned query_order — the complaint never reached the slot pipeline.
LIVE_ZH_COMPLAINT = "我要投诉物流配送，ORD1001订单的包裹三天了还没送到"


class TestComplaintExplicitVerbArbitration:
    """Finding ④ (2026-09-30): an explicit intent verb (我要投诉) must
    outrank generic keyword ties. COMPLAINT was the only task intent
    without a pattern arm; peers get +1.3 from theirs and ties break by
    enum order, so a complaint that also mentions 订单/物流 routed to
    query_order/track_shipping instead."""

    async def test_live_message_routes_to_complaint(self):
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        result = await detector.detect_with_confidence(LIVE_ZH_COMPLAINT)

        assert result.intent == Intent.COMPLAINT
        # ≥ hybrid threshold 0.7: the rule layer settles it without the
        # LLM leg, so the verdict survives a provider outage.
        assert result.confidence >= 0.7

    async def test_plain_explicit_complaint_short_circuits(self):
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        result = await detector.detect_with_confidence("我要投诉")

        assert result.intent == Intent.COMPLAINT
        assert result.confidence >= 0.7

    async def test_english_explicit_complaint_arbitrates(self):
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        result = await detector.detect_with_confidence(
            "I want to file a complaint about the delivery of order ORD1001"
        )

        assert result.intent == Intent.COMPLAINT

    async def test_message_leading_complaint_verb_wins(self):
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        result = await detector.detect_with_confidence("投诉物流配送太慢")

        assert result.intent == Intent.COMPLAINT

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("我的订单到哪了", Intent.QUERY_ORDER),
            ("物流查询", Intent.TRACK_SHIPPING),
            ("我要退货", Intent.RETURN),
            ("再不退款我就投诉", Intent.REFUND),
            ("查一下订单状态", Intent.QUERY_ORDER),
        ],
    )
    async def test_non_complaint_verbs_unchanged(self, message, expected):
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        result = await detector.detect_with_confidence(message)

        assert result.intent == expected, message

    async def test_english_verb_form_complain_arbitrates(self):
        """EN verb form ("complain", not the noun "complaint") must win
        the same arbitration — zh got verb coverage free because 投诉 is
        both verb and noun."""
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        result = await detector.detect_with_confidence(
            "I want to complain about the delivery of order ORD1001"
        )

        assert result.intent == Intent.COMPLAINT
