"""Chinese keyword recall and true BM25 scoring (review #6).

The review reproduced: document ``商品支持七天无理由退货`` with query
``无理由退货`` returned **zero** keyword hits, because ``\\b\\w+\\b``
treats a contiguous CJK run as one term, so a sub-phrase query can
never intersect it. The scoring side was equally hollow: "BM25" was
``len(matches) / len(query_terms)`` — no IDF, no term frequency, no
length normalization, so a term appearing in every document counted
as much as a term appearing in one.

These tests pin the two fixes together: CJK text is indexed as
character bigrams (the Elasticsearch ``cjk_bigram`` baseline — no
tokenizer dependency), ASCII runs stay whole words, and scoring is
real Okapi BM25 (Robertson IDF, saturated TF, length normalization).
"""

from app.services.retrieval.hybrid_search import KeywordSearch
from app.services.retrieval.vector_base import VectorSearchRequest


async def _seeded(docs: list[tuple[str, str]]) -> KeywordSearch:
    leg = KeywordSearch()
    await leg.add_documents([{"id": doc_id, "content": content} for doc_id, content in docs])
    return leg


async def _top_ids(leg: KeywordSearch, query: str, top_k: int = 10) -> list[str]:
    hits = await leg.search(VectorSearchRequest(query=query, top_k=top_k))
    return [h.document_id for h in hits]


class TestChineseRecall:
    async def test_review_repro_subphrase_query_recalls_document(self) -> None:
        """The exact repro from the review: a sub-phrase of a
        contiguous Chinese sentence must recall the document — bigrams
        of the query intersect bigrams of the document."""
        leg = await _seeded(
            [
                ("c1", "商品支持七天无理由退货"),
                ("c2", "运费说明与配送范围"),
            ]
        )

        hits = await _top_ids(leg, "无理由退货")

        assert hits == ["c1"]

    async def test_mixed_ascii_chinese_query_works(self) -> None:
        """ASCII runs stay whole words while CJK runs become bigrams —
        one query can carry both."""
        leg = await _seeded([("c1", "Pro会员 运费免费 上门取件")])

        hits = await _top_ids(leg, "pro 运费免费")

        assert hits == ["c1"]

    async def test_bigram_term_shape(self) -> None:
        """Tokenization contract: CJK runs → character bigrams (a
        lone CJK char stays a unigram), ASCII runs → lowercase words.
        Compared as term-frequency dicts to pin the Counter shape the
        BM25 scorer relies on."""
        leg = KeywordSearch()

        assert dict(leg._extract_terms("无理由退货")) == {
            "无理": 1,
            "理由": 1,
            "由退": 1,
            "退货": 1,
        }
        assert dict(leg._extract_terms("买")) == {"买": 1}
        assert dict(leg._extract_terms("Return Policy 2026")) == {
            "return": 1,
            "policy": 1,
            "2026": 1,
        }


class TestTrueBM25:
    async def test_rare_term_outranks_common_term(self) -> None:
        """IDF must distinguish evidence quality: matching the rare
        query term (appears in one document) beats matching the common
        one (appears in many). The old set-ratio scored both 0.5 and
        left the wrong document first."""
        leg = await _seeded(
            [
                ("f1", "七天可以退货"),
                ("f2", "生鲜不退货"),
                ("f3", "退货流程说明"),
                ("a", "仅支持退货"),  # matches only the common term
                ("b", "支持价保"),  # matches only the rare term
            ]
        )

        hits = await _top_ids(leg, "价保 退货")

        assert hits[0] == "b"

    async def test_term_frequency_breaks_ties(self) -> None:
        """TF must matter within saturation: a document that repeats
        the query term ranks above one that mentions it once. The old
        scorer gave both an identical 1.0."""
        leg = await _seeded(
            [
                ("once", "退货需要审核 通过后退款"),
                ("often", "退货 退货 退货 政策优先"),
            ]
        )

        hits = await _top_ids(leg, "退货")

        assert hits[0] == "often"

    async def test_short_document_is_not_drowned(self) -> None:
        """Length normalization: the same single mention counts for
        more in a two-line answer than in a chapter that buries it
        (both documents mention the term exactly once)."""
        leg = await _seeded(
            [
                (
                    "long",
                    "会员制度 积分规则 售后网点 退货 联系客服 营业时间 配送范围",
                ),
                ("short", "退货请联系客服"),
            ]
        )

        hits = await _top_ids(leg, "退货")

        assert hits[0] == "short"

    async def test_ascii_word_recall_is_unchanged(self) -> None:
        """Regression guard for the other keyword tests: ASCII word
        tokens recall exactly as before the tokenizer change."""
        leg = await _seeded(
            [
                ("a1", "zxqalpha 退货政策第一条"),
                ("b1", "yxwbeta 运费说明"),
            ]
        )

        assert await _top_ids(leg, "zxqalpha") == ["a1"]
        assert await _top_ids(leg, "yxwbeta") == ["b1"]

    async def test_no_overlap_still_returns_nothing(self) -> None:
        """Scoring upgrade must not loosen the no-hit contract: a
        query sharing no term with any document returns an empty
        list, not zero-score filler."""
        leg = await _seeded([("c1", "商品支持七天无理由退货")])

        assert await _top_ids(leg, "发票抬头") == []
