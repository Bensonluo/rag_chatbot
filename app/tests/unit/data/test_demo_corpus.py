"""No-drift guard for the synthetic demo knowledge base.

The demo corpus (app/data/demo_kb/demo_corpus.json) is fictional, but
its hard policy numbers are the ones generation quotes in the live
demo — and they are verified against app/services/facts/policy_facts.json
by the claim gate. If the corpus and the fact table diverge, generated
answers get corrected mid-sentence or flagged as violations. These
tests pin the corpus to the fact table so the two cannot silently
diverge (same pattern as the FAQ no-drift pin).
"""

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
CORPUS_PATH = REPO_ROOT / "app" / "data" / "demo_kb" / "demo_corpus.json"
FACTS_PATH = REPO_ROOT / "app" / "services" / "facts" / "policy_facts.json"


def _docs() -> dict[str, dict[str, Any]]:
    data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    return {doc["doc_id"]: doc for doc in data["docs"]}


class TestDemoCorpusStructure:
    def test_synthetic_marker_present(self) -> None:
        raw = CORPUS_PATH.read_text(encoding="utf-8")
        assert "Synthetic demo knowledge base" in raw
        assert "Fictional" in raw

    def test_every_topic_ships_as_zh_en_pair(self) -> None:
        docs = _docs()
        topics_by_lang: dict[str, set[str]] = {"zh": set(), "en": set()}
        for doc in docs.values():
            topics_by_lang[doc["lang"]].add(doc["topic"])
        assert topics_by_lang["zh"] == topics_by_lang["en"]
        assert len(topics_by_lang["zh"]) >= 12

    def test_doc_ids_unique_and_prefixed(self) -> None:
        docs = _docs()
        assert len(docs) == len(json.loads(CORPUS_PATH.read_text(encoding="utf-8"))["docs"])
        assert all(doc_id.startswith("demo_") for doc_id in docs)


class TestDemoCorpusFactAlignment:
    """Every hard number the claim gate can check must match policy_facts.json.

    The substring table below is keyed by (doc_id, substring); each value
    is derived from the corresponding fact's ``value``/``unit`` — if
    policy_facts.json changes a number, this table (and the corpus) must
    change with it.
    """

    PINS: list[tuple[str, str]] = [
        # return_window 7 天 / exchange_window 15 天
        ("demo_returns_policy_zh", "7 天内无理由退货"),
        ("demo_returns_policy_zh", "15 天内可申请换货"),
        ("demo_returns_policy_en", "within 7 days of delivery"),
        ("demo_returns_policy_en", "within 15 days of delivery"),
        # refund_processing 3-5 / arrival 1-3, 3-7, 7-15 / auto 1000
        ("demo_refunds_zh", "3-5 个工作日"),
        ("demo_refunds_zh", "1-3 个工作日"),
        ("demo_refunds_zh", "3-7 个工作日"),
        ("demo_refunds_zh", "7-15 个工作日"),
        ("demo_refunds_zh", "1000 元"),
        ("demo_refunds_en", "3-5 business days"),
        ("demo_refunds_en", "1-3 business days"),
        ("demo_refunds_en", "3-7 business days"),
        ("demo_refunds_en", "7-15 business days"),
        ("demo_refunds_en", "1000 CNY"),
        # shipping_free_threshold 99 / fee 6-12 / dispatch 48h, 72h
        ("demo_shipping_delivery_zh", "满 99 元包邮"),
        ("demo_shipping_delivery_zh", "6-12 元"),
        ("demo_shipping_delivery_zh", "48 小时"),
        ("demo_shipping_delivery_zh", "72 小时"),
        ("demo_shipping_delivery_en", "99 CNY or more"),
        ("demo_shipping_delivery_en", "6-12 CNY"),
        ("demo_shipping_delivery_en", "48 hours"),
        ("demo_shipping_delivery_en", "72 hours"),
        # invoice_window 90
        ("demo_invoices_zh", "90 天内"),
        ("demo_invoices_en", "90 days"),
        # payment_retry_wait 10-30 分钟 / recovery 1-7 工作日
        ("demo_payments_zh", "10-30 分钟"),
        ("demo_payments_zh", "1-7 个工作日"),
        ("demo_payments_en", "10-30 minutes"),
        ("demo_payments_en", "1-7 business days"),
        # coupon_threshold 300 / cap 50 / validity 30 天
        ("demo_coupons_zh", "满 300 元"),
        ("demo_coupons_zh", "最高可抵扣 50 元"),
        ("demo_coupons_zh", "30 天内有效"),
        ("demo_coupons_en", "300 CNY"),
        ("demo_coupons_en", "50 CNY"),
        ("demo_coupons_en", "30 days"),
        # shipping_insurance_cap 25
        ("demo_freight_insurance_zh", "最高赔付 25 元"),
        ("demo_freight_insurance_en", "25 CNY"),
        # authenticity_window 15 / 假一赔十
        ("demo_authenticity_zh", "15 天内申请第三方"),
        ("demo_authenticity_zh", "假一赔十"),
        ("demo_authenticity_en", "15 days of delivery"),
        ("demo_authenticity_en", "tenfold"),
    ]

    def test_pinned_numbers_present(self) -> None:
        docs = _docs()
        missing = [
            f"{doc_id}: {substring!r}"
            for doc_id, substring in self.PINS
            if substring not in docs[doc_id]["content"]
        ]
        assert not missing, f"corpus drifted from policy_facts.json: {missing}"

    def test_every_fact_value_appears_in_its_topic_docs(self) -> None:
        """Cross-check: each numeric fact's value string exists in both
        language versions of the topic doc that carries it."""
        facts = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
        topic_docs = {
            "returns_policy": ["demo_returns_policy_zh", "demo_returns_policy_en"],
            "refunds": ["demo_refunds_zh", "demo_refunds_en"],
            "shipping_delivery": ["demo_shipping_delivery_zh", "demo_shipping_delivery_en"],
            "order_tracking": ["demo_order_tracking_zh", "demo_order_tracking_en"],
            "invoices": ["demo_invoices_zh", "demo_invoices_en"],
            "payments": ["demo_payments_zh", "demo_payments_en"],
            "coupons": ["demo_coupons_zh", "demo_coupons_en"],
            "freight_insurance": ["demo_freight_insurance_zh", "demo_freight_insurance_en"],
            "authenticity": ["demo_authenticity_zh", "demo_authenticity_en"],
        }
        topic_by_fact = {
            "refund_arrival_alipay_wechat": "refunds",
            "refund_arrival_bank": "refunds",
            "refund_arrival_credit": "refunds",
            "refund_processing": "refunds",
            "refund_auto_threshold": "refunds",
            "shipping_insurance_cap": "freight_insurance",
            "shipping_fee_amount": "shipping_delivery",
            "shipping_free_threshold": "shipping_delivery",
            "return_window": "returns_policy",
            "exchange_window": "returns_policy",
            "authenticity_window": "authenticity",
            "shipping_normal": "shipping_delivery",
            "shipping_peak": "shipping_delivery",
            "invoice_window": "invoices",
            "payment_retry_wait": "payments",
            "payment_recovery": "payments",
            "coupon_threshold": "coupons",
            "coupon_discount_cap": "coupons",
            "coupon_validity": "coupons",
        }
        docs = _docs()
        unchecked = []
        for fact in facts:
            if not fact["value"]:
                continue
            topic = topic_by_fact.get(fact["id"])
            if topic is None:
                continue
            for doc_id in topic_docs[topic]:
                # The numeric value itself (e.g. "1-3", "99", "25") must
                # appear in the doc text carrying that policy.
                if fact["value"] not in docs[doc_id]["content"]:
                    unchecked.append(f"{fact['id']} missing value {fact['value']!r} in {doc_id}")
        assert not unchecked, unchecked
