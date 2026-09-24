"""Fact store for deterministic policy grounding (Phase A2).

Industry consensus (Air Canada / Klarna case studies, commerce-ops
patterns): hard policy numbers never live only in LLM free text — they
live in typed, versioned data, and the LLM's claims are checked against
that data. The store ships the curated fact table as configuration and
pulls an entity-anchored subgraph for each message (GRAG-style), so the
claim checker only ever evaluates facts the user's message actually
invites.
"""

import json
from pathlib import Path

from app.services.facts.fact_store import FactStore

FACTS_FILE = Path(__file__).parents[5] / "app/services/facts/policy_facts.json"
FAQS_FILE = Path(__file__).parents[5] / "app/services/faq/faqs.json"


class TestCuratedFacts:
    def test_loads_curated_fact_table(self):
        store = FactStore(FACTS_FILE)
        assert len(store.facts) >= 12

    def test_fact_ids_are_unique(self):
        store = FactStore(FACTS_FILE)
        ids = [f.id for f in store.facts]
        assert len(ids) == len(set(ids))

    def test_every_fact_is_actionable(self):
        """Each fact must carry anchors (subgraph pull), a statement
        (grounded replacement text), and a topic."""
        store = FactStore(FACTS_FILE)
        for fact in store.facts:
            assert fact.anchors, f"{fact.id} has no anchors"
            assert fact.statement, f"{fact.id} has no statement"
            assert fact.topic, f"{fact.id} has no topic"

    def test_each_topic_has_a_default_subject(self):
        """Claim checking falls back to default-subject facts when the
        response does not name one. Every topic must define at least one,
        and defaults within a topic must share a single subject: several
        dimensions of the SAME subject may all be defaults (coupon
        threshold/cap/validity are all 满减券), but two different subjects
        can never both be the unnamed-subject fallback — that would make
        「退款 X 个工作日」 ambiguous about the channel."""
        store = FactStore(FACTS_FILE)
        defaults: dict[str, set[str]] = {}
        for fact in store.facts:
            if fact.default_subject:
                defaults.setdefault(fact.topic, set()).add(fact.subject)
        all_topics = {f.topic for f in store.facts}
        assert set(defaults) == all_topics
        for topic, subjects in defaults.items():
            assert len(subjects) == 1, f"{topic} has ambiguous default subjects: {subjects}"

    def test_missing_file_degrades_to_empty(self, tmp_path: Path) -> None:
        store = FactStore(tmp_path / "nope.json")
        assert store.facts == []

    def test_malformed_file_degrades_to_empty(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("[{broken", encoding="utf-8")
        store = FactStore(bad)
        assert store.facts == []


class TestSubgraphRetrieval:
    def test_refund_message_pulls_refund_arrival_facts(self):
        store = FactStore(FACTS_FILE)
        subgraph = store.subgraph_for("退款多久到账")
        topics = {f.topic for f in subgraph}
        assert "refund_arrival" in topics
        # Every returned fact's anchor must actually appear in the message.
        for fact in subgraph:
            assert any(a in "退款多久到账" for a in fact.anchors)

    def test_return_message_pulls_return_window(self):
        store = FactStore(FACTS_FILE)
        subgraph = store.subgraph_for("怎么退货")
        assert "return_window" in {f.topic for f in subgraph}

    def test_shipping_insurance_message_pulls_cap_fact(self):
        store = FactStore(FACTS_FILE)
        subgraph = store.subgraph_for("运费险能赔多少")
        caps = [f for f in subgraph if f.topic == "shipping_insurance"]
        assert caps and caps[0].unit == "元"

    def test_coupon_message_pulls_usage_facts(self):
        """One coupon topic, three numeric dimensions (threshold 元 /
        discount cap 元 / validity 天) — grounding then covers all of
        them at once."""
        store = FactStore(FACTS_FILE)
        subgraph = store.subgraph_for("满减券怎么用")
        coupon = [f for f in subgraph if f.topic == "coupon_usage"]
        units = {f.unit for f in coupon}
        assert {"元", "天"} <= units
        assert len(coupon) >= 3

    def test_money_facts_carry_topic_keywords(self):
        """A fact with a value but no topic keywords can never bind to a
        clause — dead config (regression: refund_auto_threshold shipped
        uncheckable for exactly this reason)."""
        store = FactStore(FACTS_FILE)
        for fact in store.facts:
            if fact.value and fact.kind != "statement":
                assert fact.topic_keywords, f"{fact.id} has a value but no topic keywords"

    def test_chitchat_pulls_nothing(self):
        store = FactStore(FACTS_FILE)
        assert store.subgraph_for("你好呀") == []

    def test_default_store_is_cached_singleton(self):
        assert FactStore.load_default() is FactStore.load_default()


class TestFactsStayConsistentWithFaq:
    """No-drift guard: every fact with a faq_id must quote numbers that
    literally appear in the corresponding FAQ answer — the fact table and
    the served FAQ text must never diverge (single source of truth)."""

    def test_numeric_values_appear_in_faq_answers(self):
        store = FactStore(FACTS_FILE)
        faqs = {e["id"]: e["answer"] for e in json.loads(FAQS_FILE.read_text("utf-8"))}
        checked = 0
        for fact in store.facts:
            if not fact.faq_id or not fact.value:
                continue
            answer = faqs[fact.faq_id]
            for number in fact.value.split("-"):
                assert number in answer, (
                    f"{fact.id} value {fact.value} drifted from FAQ {fact.faq_id}: {answer}"
                )
            checked += 1
        assert checked >= 10  # the guard must actually cover the table
