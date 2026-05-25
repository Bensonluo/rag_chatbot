"""Tests for Intent detection base interface"""
import pytest
from unittest.mock import Mock
from app.models.enums.intent import Intent


class TestIntentDetector:
    """Test Intent detector base class"""

    def test_base_class_is_abstract(self):
        """Test that IntentDetector cannot be instantiated directly"""
        # Arrange
        from app.services.intent.base import IntentDetector

        # Act & Assert - ABC with abstract methods cannot be instantiated
        with pytest.raises(TypeError, match="abstract"):
            IntentDetector()

    def test_subclass_must_implement_detect(self):
        """Test that subclass must implement detect method"""
        # Arrange
        from app.services.intent.base import IntentDetector

        class IncompleteDetector(IntentDetector):
            pass

        # Act & Assert
        with pytest.raises(TypeError, match="abstract"):
            IncompleteDetector()


class TestIntentEnum:
    """Test Intent enum values"""

    def test_intent_values(self):
        """Test that all expected business intents exist"""
        # Arrange & Act
        from app.models.enums.intent import Intent

        # Assert - Task-oriented intents
        assert Intent.REFUND == "refund"
        assert Intent.RETURN == "return"
        assert Intent.QUERY_ORDER == "query_order"
        assert Intent.TRACK_SHIPPING == "track_shipping"
        assert Intent.COMPLAINT == "complaint"
        # Knowledge intents
        assert Intent.FAQ == "faq"
        assert Intent.POLICY == "policy"
        # Dialogue intents
        assert Intent.CHITCHAT == "chitchat"
        assert Intent.GREETING == "greeting"
        # Meta intents
        assert Intent.CONFIRM == "confirm"
        assert Intent.DENY == "deny"
        assert Intent.CANCEL == "cancel"
        assert Intent.UNKNOWN == "unknown"
        # Graph-related intents
        assert Intent.RELATIONSHIP_QUERY == "relationship_query"
        assert Intent.GLOBAL_SUMMARY == "global_summary"
        assert Intent.ENTITY_LOOKUP == "entity_lookup"

    def test_intent_completeness(self):
        """Test that we have all required intents"""
        # Arrange
        from app.models.enums.intent import Intent

        required_intents = [
            "REFUND",
            "RETURN",
            "QUERY_ORDER",
            "TRACK_SHIPPING",
            "COMPLAINT",
            "FAQ",
            "POLICY",
            "CHITCHAT",
            "GREETING",
            "CONFIRM",
            "DENY",
            "CANCEL",
            "UNKNOWN",
            "RELATIONSHIP_QUERY",
            "GLOBAL_SUMMARY",
            "ENTITY_LOOKUP",
        ]

        # Act & Assert
        for intent_name in required_intents:
            assert hasattr(Intent, intent_name)


class TestIntentResult:
    """Test IntentResult dataclass"""

    def test_create_intent_result(self):
        """Test creating an intent result"""
        # Arrange
        from app.services.intent.base import IntentResult

        # Act
        result = IntentResult(
            intent=Intent.REFUND,
            confidence=0.95
        )

        # Assert
        assert result.intent == Intent.REFUND
        assert result.confidence == 0.95

    def test_create_intent_result_with_metadata(self):
        """Test creating an intent result with metadata"""
        # Arrange
        from app.services.intent.base import IntentResult

        # Act
        result = IntentResult(
            intent=Intent.TRACK_SHIPPING,
            confidence=0.88,
            metadata={"matched_keyword": "快递", "rule_used": "track_shipping_rule_1"}
        )

        # Assert
        assert result.intent == Intent.TRACK_SHIPPING
        assert result.confidence == 0.88
        assert result.metadata["matched_keyword"] == "快递"

    def test_create_intent_result_default_confidence(self):
        """Test creating intent result with default confidence"""
        # Arrange
        from app.services.intent.base import IntentResult

        # Act
        result = IntentResult(intent=Intent.GREETING)

        # Assert
        assert result.intent == Intent.GREETING
        assert result.confidence == 0.0  # Default value

    def test_create_intent_result_default_metadata(self):
        """Test creating intent result with default metadata"""
        # Arrange
        from app.services.intent.base import IntentResult

        # Act
        result = IntentResult(
            intent=Intent.FAQ,
            confidence=0.8
        )

        # Assert
        assert result.metadata is None  # Default is None
