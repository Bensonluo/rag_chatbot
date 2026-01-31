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

    def test_detect_question_what(self):
        """Test detecting 'what' question"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("What is the capital of France?")

        # Assert
        assert intent == Intent.QUESTION

    def test_detect_question_how(self):
        """Test detecting 'how' question that's not a how-to"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("How are you doing today?")

        # Assert
        # This could be QUESTION or CHITCHAT depending on implementation
        assert intent in [Intent.QUESTION, Intent.CHITCHAT]

    def test_detect_how_to(self):
        """Test detecting how-to intent"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("How do I implement binary search in Python?")

        # Assert
        assert intent == Intent.HOW_TO

    def test_detect_comparison(self):
        """Test detecting comparison intent"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("Python vs JavaScript: which is better for web development?")

        # Assert
        assert intent == Intent.COMPARISON

    def test_detect_definition(self):
        """Test detecting definition intent"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("Define machine learning")

        # Assert
        assert intent == Intent.DEFINITION

    def test_detect_summary(self):
        """Test detecting summary intent"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("Summarize the key points of the article")

        # Assert
        assert intent == Intent.SUMMARY

    def test_detect_code_help(self):
        """Test detecting code help intent"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("Help me debug this Python function")

        # Assert
        assert intent == Intent.CODE_HELP

    def test_detect_creative(self):
        """Test detecting creative writing intent"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("Write a poem about spring")

        # Assert
        assert intent == Intent.CREATIVE

    def test_detect_chitchat(self):
        """Test detecting chitchat intent"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("Hello, how are you?")

        # Assert
        assert intent == Intent.CHITCHAT

    def test_detect_task(self):
        """Test detecting task intent"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("Send an email to john@example.com")

        # Assert
        assert intent == Intent.TASK

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
        result = detector.detect_with_confidence("How do I create a function in Python?")

        # Assert
        assert result.intent == Intent.HOW_TO
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
        result = detector.detect_with_confidence("Write a Python function")

        # Assert
        assert result.intent == Intent.CODE_HELP
        assert result.confidence > 0
        assert result.metadata is not None
        # Metadata should contain info about matched rules

    def test_case_insensitive(self):
        """Test that detection is case-insensitive"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent1 = detector.detect("HOW DO I CREATE A FUNCTION?")
        intent2 = detector.detect("How do I create a function?")

        # Assert
        assert intent1 == intent2
        assert intent1 == Intent.HOW_TO

    def test_punctuation_handling(self):
        """Test that punctuation is handled correctly"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("Hello!!!")

        # Assert
        assert intent == Intent.CHITCHAT

    def test_empty_query(self):
        """Test handling empty query"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("")

        # Assert
        assert intent == Intent.UNKNOWN

    def test_multiple_keywords(self):
        """Test query with multiple intent keywords"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act - "how" and "code" both present
        intent = detector.detect("How do I write code for this?")

        # Assert - Should match one intent based on rule priority
        assert intent in [Intent.HOW_TO, Intent.CODE_HELP]
