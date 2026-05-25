"""Tests for rule-based intent detector"""
import pytest
from app.models.enums.intent import Intent


class TestRuleBasedIntentDetector:
    """Test rule-based intent detector"""

    def test_initialization(self):
        """Test detector initialization"""
        # Arrange & Act
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Assert
        assert detector is not None
        assert hasattr(detector, 'rules')

    def test_detect_refund_keyword(self):
        """Test detecting refund intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("我要退款")

        # Assert
        assert intent == Intent.REFUND

    def test_detect_refund_english(self):
        """Test detecting refund intent from English keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("I want a refund")

        # Assert
        assert intent == Intent.REFUND

    def test_detect_return_keyword(self):
        """Test detecting return intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("我要退货")

        # Assert
        assert intent == Intent.RETURN

    def test_detect_return_exchange(self):
        """Test detecting return intent from exchange keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("我想换货")

        # Assert
        assert intent == Intent.RETURN

    def test_detect_query_order_keyword(self):
        """Test detecting query_order intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("查一下订单")

        # Assert
        assert intent == Intent.QUERY_ORDER

    def test_detect_query_order_status(self):
        """Test detecting query_order intent from status pattern"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("我的订单状态是什么")

        # Assert
        assert intent == Intent.QUERY_ORDER

    def test_detect_track_shipping_keyword(self):
        """Test detecting track_shipping intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("快递到哪了")

        # Assert
        assert intent == Intent.TRACK_SHIPPING

    def test_detect_track_shipping_logistics(self):
        """Test detecting track_shipping intent from logistics keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("物流查询")

        # Assert
        assert intent == Intent.TRACK_SHIPPING

    def test_detect_complaint_keyword(self):
        """Test detecting complaint intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("我要投诉")

        # Assert
        assert intent == Intent.COMPLAINT

    def test_detect_policy_keyword(self):
        """Test detecting policy intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("退货政策是什么")

        # Assert
        assert intent == Intent.POLICY

    def test_detect_faq_keyword(self):
        """Test detecting FAQ intent from general question keywords"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act - "能不能" is a FAQ keyword that doesn't match task-oriented patterns
        intent = detector.detect("能不能帮我看看")

        # Assert
        assert intent == Intent.FAQ

    def test_detect_greeting_keyword(self):
        """Test detecting greeting intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("你好")

        # Assert
        assert intent == Intent.GREETING

    def test_detect_greeting_english(self):
        """Test detecting greeting intent from English keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("hello")

        # Assert
        assert intent == Intent.GREETING

    def test_detect_chitchat_keyword(self):
        """Test detecting chitchat intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("哈哈")

        # Assert
        assert intent == Intent.CHITCHAT

    def test_detect_confirm_keyword(self):
        """Test detecting confirm intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("是的")

        # Assert
        assert intent == Intent.CONFIRM

    def test_detect_deny_keyword(self):
        """Test detecting deny intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("不是")

        # Assert
        assert intent == Intent.DENY

    def test_detect_cancel_keyword(self):
        """Test detecting cancel intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("取消")

        # Assert
        assert intent == Intent.CANCEL

    def test_detect_unknown(self):
        """Test detecting unknown intent"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("Xylophone zebra yellow")

        # Assert
        assert intent == Intent.UNKNOWN

    def test_detect_with_confidence_high(self):
        """Test detecting intent with high confidence"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        result = detector.detect_with_confidence("我要退款")

        # Assert
        assert result.intent == Intent.REFUND
        assert result.confidence > 0.5
        assert result.confidence <= 1.0

    def test_detect_with_confidence_low(self):
        """Test detecting intent with low confidence"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        result = detector.detect_with_confidence("Xylophone zebra yellow")

        # Assert
        assert result.intent == Intent.UNKNOWN
        assert result.confidence < 0.5

    def test_detect_with_confidence_metadata(self):
        """Test that confidence result includes metadata"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        result = detector.detect_with_confidence("我要退款")

        # Assert
        assert result.intent == Intent.REFUND
        assert result.confidence > 0
        assert result.metadata is not None
        # Metadata should contain info about matched rules
        assert "matched_rules" in result.metadata

    def test_case_insensitive(self):
        """Test that detection is case-insensitive"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent1 = detector.detect("REFUND")
        intent2 = detector.detect("refund")

        # Assert
        assert intent1 == intent2
        assert intent1 == Intent.REFUND

    def test_punctuation_handling(self):
        """Test that punctuation is handled correctly"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("你好！！！")

        # Assert
        assert intent == Intent.GREETING

    def test_empty_query(self):
        """Test handling empty query"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("")

        # Assert
        assert intent == Intent.UNKNOWN

    def test_multiple_keywords_refund_and_return(self):
        """Test query with both refund and return keywords"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act - Both "退款" and "退货" present
        intent = detector.detect("我要退款退货")

        # Assert - Should match one intent based on rule priority
        assert intent in [Intent.REFUND, Intent.RETURN]

    def test_detect_with_context_parameter(self):
        """Test that detect accepts optional context parameter"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("我要退款", context={"previous_messages": []})

        # Assert
        assert intent == Intent.REFUND

    def test_detect_track_shipping_pattern(self):
        """Test detecting track_shipping via regex pattern"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act - matches pattern r"(物流|快递|包裹).{0,4}(到|在|哪|状态|查询)"
        intent = detector.detect("包裹到了吗")

        # Assert
        assert intent == Intent.TRACK_SHIPPING
