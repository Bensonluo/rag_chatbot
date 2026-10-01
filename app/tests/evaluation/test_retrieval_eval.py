"""Retrieval golden-set baseline harness (review 2026-09-26, #6/#10).

The answer golden set injects the correct faq_ids and grades only
"given the right policy text, can the LLM compose the answer" — it
bypasses routing, FAQ matching, retrieval, cache and tools. This
harness grades the retrieval half deterministically: real KeywordSearch
(Okapi BM25 + CJK bigram) and the real HybridSearchService RRF fusion
over the real seeded demo corpus, with a deterministic lexical stand-in
for the vector leg — no embedding model, no Qdrant, no network, so the
baseline runs in keyless CI like the intent tier.
"""

import pytest

from app.services.evaluation.retrieval_eval import (
    LegacyWholeWordKeywordSearch,
    LexicalVectorClient,
    legacy_wholeword_terms,
    load_demo_corpus,
    load_retrieval_cases,
    ranking_metrics,
    run_retrieval_baseline,
)
from app.services.retrieval.hybrid_search import KeywordSearch
from app.services.retrieval.vector_base import VectorSearchRequest


def test_golden_set_schema_is_valid() -> None:
    """Every case points at a real corpus doc in the case's language."""
    cases = load_retrieval_cases()
    assert len(cases) >= 40, "baseline needs a meaningful sample"

    corpus = load_demo_corpus()
    lang_by_doc = {doc["id"]: doc["metadata"]["lang"] for doc in corpus}

    ids = [case.id for case in cases]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    langs = {case.lang for case in cases}
    assert langs == {"zh", "en"}, "both languages must be covered"
    for case in cases:
        assert case.expected_doc_ids, case.id
        for doc_id in case.expected_doc_ids:
            assert doc_id in lang_by_doc, f"{case.id}: unknown doc {doc_id}"
            assert lang_by_doc[doc_id] == case.lang, (
                f"{case.id} ({case.lang}) expects {doc_id} ({lang_by_doc[doc_id]})"
            )


def test_metrics_math_on_hand_computed_rankings() -> None:
    """recall@k and MRR on a ranking computed by hand.

    case1 ranking [a,b,c] expected [a]  → hit@1, rr=1
    case2 ranking [b,a,c] expected [a]  → miss@1, hit@2, rr=1/2
    case3 ranking [c,d]   expected [b]  → miss everywhere, rr=0
    recall@1 = 1/3, recall@2 = 2/3, MRR = (1 + 0.5 + 0) / 3
    """
    metrics = ranking_metrics(
        rankings=[["a", "b", "c"], ["b", "a", "c"], ["c", "d"]],
        expected=[["a"], ["a"], ["b"]],
        ks=(1, 2),
    )
    assert metrics["recall@1"] == pytest.approx(1 / 3)
    assert metrics["recall@2"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx(0.5)


async def test_lexical_vector_stand_in_is_deterministic() -> None:
    """The stand-in leg must be a fixed function of its inputs — the
    baseline's numbers have to be reproducible run over run."""
    client = LexicalVectorClient(load_demo_corpus())
    request = VectorSearchRequest(query="退款多久到账", top_k=10)
    first = await client.search(request)
    second = await client.search(request)
    assert [r.document_id for r in first] == [r.document_id for r in second]
    assert first, "a keyword-heavy query must rank at least one doc"


async def test_cjk_bigram_beats_legacy_wholeword_on_zh() -> None:
    """The tokenizer A/B the review asked for (#6): on the real corpus,
    the pre-c043c58 whole-word tokenizer collapses contiguous CJK into
    one token, so sub-phrase zh queries cannot intersect a document.
    Bigram recall must be strictly higher on zh; en must be unaffected."""
    report = await run_retrieval_baseline()

    zh_bigram = report["arms"]["keyword_bigram"]["zh"]["recall@10"]
    zh_legacy = report["arms"]["keyword_legacy_wholeword"]["zh"]["recall@10"]
    assert zh_bigram > zh_legacy, (
        f"bigram zh recall@10 ({zh_bigram}) must beat legacy ({zh_legacy})"
    )
    assert zh_bigram >= 0.9, f"bigram zh recall@10 floor: {zh_bigram}"

    en_bigram = report["arms"]["keyword_bigram"]["en"]["recall@10"]
    en_legacy = report["arms"]["keyword_legacy_wholeword"]["en"]["recall@10"]
    assert en_bigram == en_legacy, "ASCII tokenization is identical in both arms"


async def test_baseline_report_structure_and_determinism() -> None:
    """Three arms × (overall, zh, en) × {recall@1, recall@3, recall@10,
    mrr}, and the whole report is reproducible."""
    first = await run_retrieval_baseline()
    second = await run_retrieval_baseline()

    assert set(first["arms"]) == {
        "keyword_bigram",
        "keyword_legacy_wholeword",
        "hybrid_rrf_lexical_vector",
    }
    assert first["cases"] >= 40
    for arm in first["arms"].values():
        assert set(arm) == {"overall", "zh", "en"}
        for slice_ in arm.values():
            assert {"recall@1", "recall@3", "recall@10", "mrr"} <= set(slice_)
    assert first == second, "baseline must be deterministic"


async def test_pool_width_quantified_in_report() -> None:
    """The 2436d2a rationale, measured: recall@10 ≥ recall@3 on every
    arm, and the fused arm must expose a strictly positive pool gain —
    otherwise the candidate pool buys nothing on this corpus."""
    report = await run_retrieval_baseline()
    for name, arm in report["arms"].items():
        for slice_name, slice_ in arm.items():
            assert slice_["recall@10"] >= slice_["recall@3"], (name, slice_name)
    fused = report["arms"]["hybrid_rrf_lexical_vector"]["overall"]
    assert fused["recall@10"] > fused["recall@3"], "candidate pool gain is zero"


async def test_legacy_tokenizer_reproduces_pre_c043c58_behavior() -> None:
    """The legacy arm must reproduce the old failure mode exactly: a
    contiguous CJK run is ONE token, so a sub-phrase query shares zero
    terms with a longer document sentence (the review's 0-hit repro)."""
    terms = legacy_wholeword_terms("商品支持七天无理由退货")
    assert set(terms) == {"商品支持七天无理由退货"}

    legacy = LegacyWholeWordKeywordSearch()
    await legacy.add_documents([{"id": "d1", "content": "商品支持七天无理由退货"}])
    results = await legacy.search(VectorSearchRequest(query="无理由退货", top_k=5))
    assert results == [], "legacy tokenizer must reproduce the 0-hit failure"


class TestLegacyArmIsolation:
    def test_legacy_subclass_only_swaps_tokenizer(self) -> None:
        """The legacy arm must differ from the current KeywordSearch in
        exactly one dimension — the tokenizer — or the A/B measures
        something else."""
        assert issubclass(LegacyWholeWordKeywordSearch, KeywordSearch)
        probe = LegacyWholeWordKeywordSearch()
        assert set(probe._extract_terms("return policy 7 days")) == {
            "return",
            "policy",
            "7",
            "days",
        }
