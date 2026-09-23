"""Emotion screening for human-handoff escalation (deterministic keywords)."""

from app.services.dialogue.emotion import assess_emotion


class TestAnger:
    def test_anger_keyword_escalates(self):
        signal = assess_emotion("你们就是骗子，气死我了")
        assert signal.should_escalate
        assert signal.signal_class == "anger"
        assert "骗子" in signal.matched

    def test_english_anger_escalates(self):
        assert assess_emotion("this is a scam!").should_escalate

    def test_mild_dissatisfaction_does_not_escalate(self):
        """“有点失望” alone is not anger — keep the bot on it."""
        assert not assess_emotion("有点失望，但还能接受").should_escalate


class TestFrustration:
    def test_repeated_failure_escalates(self):
        signal = assess_emotion("还是没收到退款")
        assert signal.should_escalate
        assert signal.signal_class == "frustration"

    def test_completion_marker_required_for_counts(self):
        """“第三次下单” is benign; “第三次了” is exasperation."""
        assert not assess_emotion("第三次下单有优惠吗").should_escalate
        assert assess_emotion("都第三次了还不行").should_escalate

    def test_unresolved_marker_escalates(self):
        assert assess_emotion("问题一直没解决").should_escalate


class TestBenign:
    def test_plain_question_is_neutral(self):
        assert not assess_emotion("退货政策是什么").should_escalate

    def test_greeting_is_neutral(self):
        assert not assess_emotion("你好，我想查一下物流").should_escalate

    def test_empty_message_is_neutral(self):
        signal = assess_emotion("")
        assert not signal.should_escalate
        assert signal.signal_class == ""

    def test_shipping_status_update_not_frustration(self):
        assert not assess_emotion("物流还在运输中，预计明天到").should_escalate
