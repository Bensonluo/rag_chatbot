"""Tests for Intent detection base interface"""
import pytest
from unittest.mock import Mock
from app.models.enums.intent import Intent


class TestIntentDetector:
    """Test Intent detector base class"""

    def test_base_class_not_implemented_detect(self):
        """Test that detect raises NotImplementedError in base class"""
        # Arrange
        from app.services.intent.base import IntentDetector

        detector = IntentDetector()

        # Act & Assert
        with pytest.raises(NotImplementedError):
            detector.detect("Hello!")

    def test_base_class_not_implemented_detect_with_confidence(self):
        """Test that detect_with_confidence raises NotImplementedError in base class"""
        # Arrange
        from app.services.intent.base import IntentDetector

        detector = IntentDetector()

        # Act & Assert
        with pytest.raises(NotImplementedError):
            detector.detect_with_confidence("Hello!")


class TestIntentEnum:
    """Test Intent enum values"""

    def test_intent_values(self):
        """Test that all expected intents exist"""
        # Arrange & Act
        from app.models.enums.intent import Intent

        # Assert
        assert Intent.QUESTION == "question"
        assert Intent.COMPARISON == "comparison"
        assert Intent.HOW_TO == "how_to"
        assert Intent.DEFINITION == "definition"
        assert Intent.SUMMARY == "summary"
        assert Intent.CODE_HELP == "code_help"
        assert Intent.CREATIVE == "creative"
        assert Intent.CHITCHAT == "chitchat"
        assert Intent.TASK == "task"
        assert Intent.UNKNOWN == "unknown"

    def test_intent_completeness(self):
        """Test that we have all required intents"""
        # Arrange
        from app.models.enums.intent import Intent

        required_intents = [
            "QUESTION",
            "COMPARISON",
            "HOW_TO",
            "DEFINITION",
            "SUMMARY",
            "CODE_HELP",
            "CREATIVE",
            "CHITCHAT",
            "TASK",
            "UNKNOWN",
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
            intent=Intent.QUESTION,
            confidence=0.95
        )

        # Assert
        assert result.intent == Intent.QUESTION
        assert result.confidence == 0.95

    def test_create_intent_result_with_metadata(self):
        """Test creating an intent result with metadata"""
        # Arrange
        from app.services.intent.base import IntentResult

        # Act
        result = IntentResult(
            intent=Intent.CODE_HELP,
            confidence=0.88,
            metadata={"matched_keyword": "function", "rule_used": "code_help_rule_1"}
        )

        # Assert
        assert result.intent == Intent.CODE_HELP
        assert result.confidence == 0.88
        assert result.metadata["matched_keyword"] == "function"

    def test_create_intent_result_default_confidence(self):
        """Test creating intent result with default confidence"""
        # Arrange
        from app.services.intent.base import IntentResult

        # Act
        result = IntentResult(intent=Intent.CHITCHAT)

        # Assert
        assert result.intent == Intent.CHITCHAT
        assert result.confidence == 0.0  # Default value

    def test_create_intent_result_default_metadata(self):
        """Test creating intent result with default metadata"""
        # Arrange
        from app.services.intent.base import IntentResult

        # Act
        result = IntentResult(
            intent=Intent.QUESTION,
            confidence=0.8
        )

        # Assert
        assert result.metadata is None  # Default is None
