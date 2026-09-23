"""Golden-set ↔ bundled FAQ content coverage contract.

The #19 live answer eval found golden queries missing required points.
Root cause was routing, not content: customer-phrased queries ("怎么
还没发货") scored below the FAQ similarity threshold against every
variant and fell through to the generic RAG answer. Two contracts here
keep that fixed:

1. Routing coverage — each FAQ-referenced golden query must have a
   variant lexically close to it (same core phrase), so real-customer
   phrasings route into the fast path and get the pre-approved answer.
2. Answer coverage — for multi-intent cases the single best-matching
   FAQ answer must already carry every required point, because the
   matcher returns exactly one entry; a second FAQ's answer can't be
   relied on to arrive in the same reply.
"""

import re

from app.services.evaluation.answer_eval import (
    ANSWER_GOLDEN_SET_FILE,
    load_answer_cases,
)
from app.services.faq.store import DEFAULT_FAQ_FILE, FAQEntry, load_faq_entries


def _squeeze(text: str) -> str:
    """Strip whitespace so「72 小时」-style spacing can't break matching."""
    return re.sub(r"\s+", "", text)


class TestGoldenFaqContract:
    @staticmethod
    def _entries() -> dict[str, FAQEntry]:
        return {e.faq_id: e for e in load_faq_entries(DEFAULT_FAQ_FILE)}

    def test_every_golden_faq_reference_resolves(self):
        """A golden case pointing at a missing/renamed FAQ id is a broken eval."""
        entries = self._entries()
        for case in load_answer_cases(ANSWER_GOLDEN_SET_FILE):
            for faq_id in case.faq_ids:
                assert faq_id in entries, f"{case.id}: FAQ '{faq_id}' not in bundled table"

    def test_shipping_delay_query_routes_by_variant(self):
        """「还没发货」phrasings must have a lexically-close variant."""
        entry = self._entries()["track_shipping"]
        joined = _squeeze("".join([entry.question, *entry.variants]))
        for phrase in ("怎么还没发货", "还没发货正常吗"):
            assert phrase in joined, f"track_shipping lacks routing variant containing {phrase}"

    def test_shipping_answer_covers_required_points(self):
        entry = self._entries()["track_shipping"]
        answer = _squeeze(entry.answer)
        assert "72小时" in answer
        assert "短信" in answer and "app" in answer.lower()

    def test_cancel_refund_query_routes_by_variant(self):
        """取消+退款到账 multi-intent phrasings need variants to route."""
        entry = self._entries()["cancel_order"]
        joined = _squeeze("".join([entry.question, *entry.variants]))
        for phrase in ("退款到账时间", "退回微信"):
            assert phrase in joined, f"cancel_order lacks routing variant containing {phrase}"

    def test_cancel_answer_is_self_sufficient_for_multi_intent(self):
        """One cancel_order hit must satisfy all ans-combined-001 points."""
        entry = self._entries()["cancel_order"]
        answer = _squeeze(entry.answer)
        # operation path
        assert "取消订单" in answer and "我的订单" in answer
        # immediate refund for unshipped orders
        assert "立即退款" in answer
        # channel timelines (WeChat 1-3 working days is a required point)
        assert "微信" in answer and "1-3个工作日" in answer
