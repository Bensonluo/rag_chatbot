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

    async def test_cancel_howto_phrasing_is_faq_not_meta_cancel(self):
        """How-to-cancel phrasing ("怎么取消订单") is a knowledge question,
        not a request to cancel whatever is currently staged.

        Found by the offline detector probe (2026-10-03, evidence
        generation follow-up to the routing eval): CANCEL's keyword
        (取消@1.5) and the 取消.{0,4}订单 pattern (1.3) double-fire to
        2.8 (→0.93) and detect_intent gives cancel priority over
        everything — so a user mid-refund asking HOW to cancel an order
        destroyed their staged refund (meta-cancel clears
        pending_confirmation) instead of getting the answer. The shipped
        FAQ table carries this exact question (怎么取消订单 + variants);
        the forward how-to pattern just never listed 取消 as an action
        noun. Same mechanics as the 怎么退货 fix."""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act / Assert — FAQ (怎么 0.5 + forward pattern 2.4 = 2.9 → 0.97)
        # must outrank CANCEL (2.8 → 0.93) and cross the 0.7
        # short-circuit so no LLM call is needed.
        for utterance in ("怎么取消订单", "如何取消订单", "怎么才能取消"):
            result = await detector.detect_with_confidence(utterance)
            assert result.intent is Intent.FAQ, utterance
            assert result.confidence >= 0.7, utterance
        # The imperative phrasings must stay cancel requests.
        assert await detector.detect("取消订单") is Intent.CANCEL
        assert await detector.detect("我要取消订单") is Intent.CANCEL
        assert await detector.detect("帮我取消") is Intent.CANCEL

    async def test_when_timeline_phrasing_is_faq_not_action(self):
        """什么时候-timeline phrasing ("退款什么时候到账") is a knowledge
        question — the shipped FAQ table carries the 退款什么时候到 variant.

        Found by the offline detector probe (2026-10-03): 多久/几天 joined
        the FAQ keyword support (routing-eval fix) but 什么时候 never did,
        so REFUND's keyword+regex double-fire (2.7 → 0.9) kept swallowing
        the when-will-my-money-arrive question — the exact 20201e9
        failure family with a different timeline word. 什么时候/何时 join
        the keyword support and both how-to pattern question sets."""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act / Assert — FAQ (什么时候 0.5 + pattern 2.4 = 2.9 → 0.97)
        # must outrank REFUND/RETURN (2.7 → 0.9) and cross the 0.7
        # short-circuit.
        for utterance in (
            "退款什么时候到账",
            "退货什么时候到账",
            "什么时候退款",
        ):
            result = await detector.detect_with_confidence(utterance)
            assert result.intent is Intent.FAQ, utterance
            assert result.confidence >= 0.7, utterance
        # Per-order tracking questions keep their task routing — the
        # answer is the order's live status, not a policy doc.
        assert await detector.detect("我的订单什么时候到") is Intent.QUERY_ORDER
        assert await detector.detect("我要退款") is Intent.REFUND

    async def test_english_cancel_howto_and_imperative(self):
        """EN cancel phrasings: how-to reaches FAQ, imperative breaks the
        query_order tie.

        Found by the offline detector probe (2026-10-03): "How do I
        cancel my order" (a shipped FAQ-table entry) landed query_order
        because the "order" keyword (1.5) ties CANCEL's "cancel" keyword
        (1.5) and Intent-enum order crowns query_order — the f723874 tie
        family; the imperative "I want to cancel my order" had the same
        hijack, so an explicit cancel request produced an order-status
        query instead of cancelling. The how-to mirror of the zh forward
        pattern + a COMPLAINT-style explicit-verb arm fix both."""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act / Assert — how-to: FAQ pattern (2.4 → 0.8) crosses the 0.7
        # short-circuit and serves the curated EN entry.
        result = await detector.detect_with_confidence("How do I cancel my order")
        assert result.intent is Intent.FAQ
        assert result.confidence >= 0.7
        # Imperative: explicit verb (1.5 + 1.3 = 2.8 → 0.93) outranks the
        # query_order keyword tie (1.5).
        assert await detector.detect("I want to cancel my order") is Intent.CANCEL
        assert await detector.detect("cancel my order please") is Intent.CANCEL

    async def test_english_handoff_phrasings_reach_handoff(self):
        """EN human-request phrasings reach HANDOFF at outage-immune strength.

        Found by the offline detector probe (2026-10-03): every probed EN
        handoff phrasing ("how do I talk to a human", "I want to speak to
        a human", "please connect me to an agent", "human please", ...)
        scored unknown@0.00 — the HANDOFF arm was zh keywords plus two EN
        noun phrases only, so a global audience's explicit escalation
        request reached neither the deterministic leg nor a ticket. The
        pattern arm mirrors the zh keywords' 3.0 weight: it always wins
        over co-occurring task intents (peak 2.8) and crosses the 0.7
        short-circuit, so the ticket is created without the LLM leg."""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act / Assert — every probed phrasing, rule short-circuit (3.0 → 1.0).
        for utterance in [
            "how do I talk to a human",
            "I want to speak to a human",
            "can I talk to a real person",
            "please connect me to an agent",
            "human please",
            "let me talk to customer service",
        ]:
            result = await detector.detect_with_confidence(utterance)
            assert result.intent is Intent.HANDOFF, utterance
            assert result.confidence >= 0.7, utterance
        # Guard: an explicit complaint stays a complaint — "talk to" here
        # names customer service but the utterance never asks for a human.
        assert await detector.detect("I want to file a complaint") is Intent.COMPLAINT

    async def test_english_tracking_phrasings_reach_track_shipping(self):
        """EN package-tracking phrasings reach TRACK_SHIPPING, not unknown.

        Found by the offline detector probe (2026-10-03): "where is my
        package" / "track my package" / "has my shipment arrived" scored
        unknown@0.00 — the TRACK_SHIPPING pattern arm was zh-only and the
        lone "shipping" keyword does not prefix-match "shipment". The EN
        pattern arm mirrors the zh pattern weight (1.3). Bare "order"
        phrasings stay QUERY_ORDER on purpose: the zh doctrine pins
        我的订单什么时候到 → query_order (per-order tracking is an order
        query), and "do you ship internationally" must stay unknown so
        the knowledge-question fallback owns it."""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act / Assert — package/parcel/shipment nouns track.
        for utterance in [
            "where is my package",
            "where is my parcel",
            "track my package",
            "has my shipment arrived",
        ]:
            assert await detector.detect(utterance) is Intent.TRACK_SHIPPING, utterance
        # Guards: order-noun phrasings keep the per-order query semantics
        # and the capability question stays unclassified.
        assert await detector.detect("where is my order") is Intent.QUERY_ORDER
        assert await detector.detect("do you ship internationally") is Intent.UNKNOWN
        assert await detector.detect("物流查询") is Intent.TRACK_SHIPPING

    async def test_english_faq_cost_and_refund_howto(self):
        """EN cost questions and refund how-tos reach FAQ, not the task flow.

        Found by the offline detector probe (2026-10-03): "how much is
        shipping" was stolen by the TRACK_SHIPPING "shipping" keyword
        (1.5 → 0.50) although delivery_scope_en curates the exact variant
        — the zh FAQ table has a 运费…多少 cost guard but EN had none;
        "How long do refunds take" scored refund@0.50 (keyword only)
        although refund_timeline_en answers it. The EN cost patterns
        mirror the zh 运费 guard and the how-to arm mirrors the zh
        怎么…退 family (2.4 > 1.5 keyword), so the curated answers are
        served with zero LLM calls."""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act / Assert — curated-entry questions cross the 0.7 short-circuit.
        for utterance in [
            "how much is shipping",
            "how much does shipping cost",
            "How long do refunds take",
            "how do I get a refund",
            "how do I return an item",
        ]:
            result = await detector.detect_with_confidence(utterance)
            assert result.intent is Intent.FAQ, utterance
            assert result.confidence >= 0.7, utterance
        # Guards: imperative refund requests stay money-moving tasks and
        # policy phrasings keep POLICY (keyword 1.0 + pattern 2.0 = 3.0
        # outranks the FAQ how-to arm).
        assert await detector.detect("I want a refund") is Intent.REFUND
        result = await detector.detect_with_confidence("how does the refund policy work")
        assert result.intent is Intent.POLICY

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
