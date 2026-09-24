"""Sentence-buffered claim gating for token streams (A2 streaming close-out).

The post-hoc claim gate rewrites the persisted text, but on the
token-streamed path violating numbers may already have reached the
consumer — the user screenshots the wrong promise (中消协 liability
direction) while the history stores the corrected one. Industry
consensus for streamed output moderation is chunked, buffered checking
(NeMo Guardrails chunked streaming; "The Guardrail Tax" 2026): buffer
the stream, check each window, release only what passes. Our checker is
clause/sentence-scoped, so the sound window is the SENTENCE — hold
tokens until a terminator (。！？!?；;\\n), gate the complete sentence,
release the corrected text. What the user sees and what persists are
byte-identical again.
"""

from pathlib import Path
from unittest.mock import patch

from app.services.facts.fact_store import FactStore
from app.services.facts.stream_gate import StreamClaimGate, make_stream_gate

FACTS_FILE = Path(__file__).parents[5] / "app/services/facts/policy_facts.json"
STORE = FactStore(FACTS_FILE)


def _refund_gate() -> StreamClaimGate:
    return StreamClaimGate(STORE.subgraph_for("退款多久到账"))


class TestSentenceBuffering:
    def test_nothing_emits_before_terminator(self):
        gate = _refund_gate()
        assert gate.feed("退款将在 10 个工作日内到") == ""

    def test_clean_sentence_passes_through_after_terminator(self):
        gate = _refund_gate()
        assert gate.feed("退款 1-3 个工作日到账。") == "退款 1-3 个工作日到账。"

    def test_violating_sentence_corrected_before_emission(self):
        gate = _refund_gate()
        emitted = gate.feed("退款将在 10 个工作日内到账。")
        assert "10 个工作日" not in emitted
        assert "1-3 个工作日" in emitted

    def test_numbers_split_across_chunks_are_held(self):
        gate = _refund_gate()
        assert gate.feed("退款将在 10 ") == ""
        assert gate.feed("个工作日内到账") == ""
        emitted = gate.feed("。")
        assert "1-3 个工作日" in emitted
        assert "10 个工作日" not in emitted

    def test_flush_gates_and_releases_remainder(self):
        gate = _refund_gate()
        assert gate.feed("退款将在 10 个工作日内到账") == ""
        tail = gate.flush()
        assert "1-3 个工作日" in tail
        assert "10 个工作日" not in tail
        assert gate.flush() == ""  # idempotent at end of stream

    def test_sentences_emit_incrementally(self):
        """First sentence releases before the second completes — the
        latency cost is per-first-sentence, not full-response buffering."""
        gate = _refund_gate()
        assert gate.feed("好的。") == "好的。"
        assert gate.feed("退款将在 5 个工作日内退") == ""
        second = gate.feed("回。")
        assert "1-3 个工作日" in second

    def test_unexecuted_action_softened_in_stream(self):
        gate = _refund_gate()
        emitted = gate.feed("已为您办理退款。")
        assert "已为您办理退款" not in emitted
        assert "订单页面的实际处理进度" in emitted


class TestMakeStreamGate:
    def test_chitchat_message_yields_none(self):
        """Empty fact subgraph → pure token streaming keeps its latency."""
        assert make_stream_gate("你好呀") is None

    def test_checkable_message_yields_gate(self):
        assert make_stream_gate("退款多久到账") is not None

    def test_empty_message_yields_none(self):
        assert make_stream_gate("") is None

    def test_disabled_setting_yields_none(self):
        from app.config.settings import settings

        with patch.object(settings, "FACT_CLAIM_CHECK_ENABLED", False):
            assert make_stream_gate("退款多久到账") is None
