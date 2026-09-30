"""Bilingual FAQ table guard.

The FAQ fast path serves canned answers from shipped configuration
(app/services/faq/faqs.json). The product is global by default: every
curated zh entry must ship an English counterpart (``<id>_en``) whose
hard policy numbers match app/services/facts/policy_facts.json — the
claim gate verifies generated answers against that table, and a canned
EN answer quoting a drifted number would be flagged as a violation on
arrival. Mirrors the demo-corpus no-drift guard.
"""

import json
from pathlib import Path
from typing import Any

FAQ_PATH = Path(__file__).resolve().parents[5] / "app" / "services" / "faq" / "faqs.json"
FACTS_PATH = (
    Path(__file__).resolve().parents[5] / "app" / "services" / "facts" / "policy_facts.json"
)


def _entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = json.loads(FAQ_PATH.read_text(encoding="utf-8"))
    return entries


class TestBilingualStructure:
    def test_every_zh_entry_has_an_english_counterpart(self) -> None:
        entries = _entries()
        ids = {e["id"] for e in entries}
        zh_ids = {i for i in ids if not i.endswith("_en")}
        missing = sorted(i for i in zh_ids if f"{i}_en" not in ids)
        assert not missing, f"zh entries without an English counterpart: {missing}"

    def test_english_entries_are_complete(self) -> None:
        for entry in _entries():
            if not entry["id"].endswith("_en"):
                continue
            assert entry["question"].strip(), entry["id"]
            assert entry["answer"].strip(), entry["id"]
            assert len(entry.get("variants", [])) >= 3, f"{entry['id']}: fewer than 3 variants"


class TestEnglishAnswerFactAlignment:
    """Pin every hard number an EN canned answer quotes to the fact table."""

    PINS: list[tuple[str, str]] = [
        ("returns_policy_en", "7 days"),
        ("returns_policy_en", "15 days"),
        ("refund_timeline_en", "1-3 business days"),
        ("refund_timeline_en", "3-7 business days"),
        ("refund_timeline_en", "7-15 business days"),
        ("track_shipping_en", "48 hours"),
        ("track_shipping_en", "72 hours"),
        ("invoice_en", "90 days"),
        ("payment_failed_en", "10-30 minutes"),
        ("payment_failed_en", "1-7 business days"),
        ("delivery_scope_en", "99 CNY"),
        ("delivery_scope_en", "6-12 CNY"),
        ("authenticity_en", "15 days"),
        ("authenticity_en", "tenfold"),
        ("shipping_insurance_en", "25 CNY"),
        ("shipping_insurance_en", "1-7 business days"),
        ("coupon_usage_en", "300 CNY"),
        ("coupon_usage_en", "50 CNY"),
        ("coupon_usage_en", "30 days"),
        ("cancel_order_en", "1-3 business days"),
        ("cancel_order_en", "3-7 business days"),
    ]

    def test_pinned_numbers_present(self) -> None:
        answers = {e["id"]: e["answer"] for e in _entries()}
        missing = [
            f"{faq_id}: {substring!r}"
            for faq_id, substring in self.PINS
            if substring not in answers[faq_id]
        ]
        assert not missing, f"EN FAQ answers drifted from policy_facts.json: {missing}"

    def test_fact_values_appear_in_both_language_answers(self) -> None:
        """Cross-check: each numeric fact's value appears in the zh answer
        and (in its English phrasing) in the EN answer of the same topic."""
        facts = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
        # fact value → (zh substring, en substring) phrasings, both already
        # pinned individually above; this test catches NEW facts gaining no
        # FAQ coverage and value drift in the zh table itself.
        covered_facts = {
            "refund_arrival_alipay_wechat": ("1-3 个工作日", "1-3 business days"),
            "refund_arrival_bank": ("3-7 个工作日", "3-7 business days"),
            "refund_arrival_credit": ("7-15 个工作日", "7-15 business days"),
            "return_window": ("7 天内无理由退货", "7 days"),
            "exchange_window": ("15 天换货", "15 days"),
            "shipping_free_threshold": ("满 99 元包邮", "99 CNY"),
            "shipping_fee_amount": ("6-12 元", "6-12 CNY"),
            "shipping_normal": ("48 小时内发货", "48 hours"),
            "shipping_peak": ("72 小时", "72 hours"),
            "invoice_window": ("90 天内", "90 days"),
            "payment_retry_wait": ("10-30 分钟", "10-30 minutes"),
            "payment_recovery": ("1-7 个工作日", "1-7 business days"),
            "shipping_insurance_cap": ("最高赔付 25 元", "25 CNY"),
            "coupon_threshold": ("满 300 元", "300 CNY"),
            "coupon_discount_cap": ("最高可抵扣 50 元", "50 CNY"),
            "coupon_validity": ("30 天内有效", "30 days"),
        }
        fact_values = {f["id"]: f["value"] for f in facts}
        topic_by_fact = {
            "refund_arrival_alipay_wechat": "refund_timeline",
            "refund_arrival_bank": "refund_timeline",
            "refund_arrival_credit": "refund_timeline",
            "return_window": "returns_policy",
            "exchange_window": "authenticity",
            "shipping_free_threshold": "delivery_scope",
            "shipping_fee_amount": "delivery_scope",
            "shipping_normal": "track_shipping",
            "shipping_peak": "track_shipping",
            "invoice_window": "invoice",
            "payment_retry_wait": "payment_failed",
            "payment_recovery": "payment_failed",
            "shipping_insurance_cap": "shipping_insurance",
            "coupon_threshold": "coupon_usage",
            "coupon_discount_cap": "coupon_usage",
            "coupon_validity": "coupon_usage",
        }
        answers = {e["id"]: e["answer"] for e in _entries()}
        problems = []
        for fact_id, (zh_sub, en_sub) in covered_facts.items():
            if fact_id not in fact_values:
                problems.append(f"{fact_id} no longer exists in policy_facts.json")
                continue
            topic = topic_by_fact[fact_id]
            if fact_values[fact_id] not in answers[topic]:
                problems.append(f"{fact_id} value {fact_values[fact_id]!r} missing from {topic}")
            if zh_sub not in answers[topic]:
                problems.append(f"zh phrasing {zh_sub!r} missing from {topic}")
            if en_sub not in answers[f"{topic}_en"]:
                problems.append(f"en phrasing {en_sub!r} missing from {topic}_en")
        assert not problems, problems
