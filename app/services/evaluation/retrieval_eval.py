"""Retrieval golden-set baseline (review 2026-09-26, #6 and #10).

The answer golden set (``answer_eval``) injects the correct faq_ids and
grades only "given the right policy text, can the LLM compose the
answer" — it bypasses intent routing, FAQ matching, retrieval, cache and
tools. This module grades the retrieval half of that gap
deterministically and keylessly: the real ``KeywordSearch`` (Okapi BM25
+ CJK character bigrams) and the real ``HybridSearchService`` RRF
fusion, run over the real seeded demo corpus, with a deterministic
lexical stand-in for the embedding+Qdrant leg — no model download, no
network, so the baseline runs in CI like the intent tier.

Two comparison arms answer the review's explicit questions (#6 建议原文:
"先建立中文分词/字符 n-gram 与真正 BM25 的对照基线，再评估更大候选池的增益"):

- **tokenizer A/B** — current CJK bigram vs the pre-c043c58 whole-word
  tokenizer, quantifying the zh recall gap on the real corpus;
- **pool width** — recall@1/3/10 on every arm, quantifying what the
  10-doc candidate pool (2436d2a) buys over the old top_k=3 request: a
  golden doc fusion ranks 4th-10th is invisible at delivery width 3 and
  rescued by the pool.

Honesty note: the lexical stand-in means the *fused* arm's absolute
numbers are a model, not a production measurement — real embeddings
decorrelate the two legs. The keyword-leg numbers and the tokenizer A/B
are true production-leg measurements. A live run swaps in the real
vector client without touching the golden set or the metrics.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.retrieval.hybrid_search import (
    HybridSearchService,
    KeywordSearch,
)
from app.services.retrieval.vector_base import (
    Document,
    SearchResult,
    VectorClient,
    VectorClientError,
    VectorSearchRequest,
)

RETRIEVAL_GOLDEN_SET_FILE = Path(__file__).parent / "retrieval_golden_set.json"

DEMO_CORPUS_FILE = Path(__file__).resolve().parents[2] / "data" / "demo_kb" / "demo_corpus.json"

# Delivery width (RETRIEVAL_TOP_K), candidate pool width
# (RETRIEVAL_CANDIDATE_POOL), and the legacy request width — the @k
# values the baseline reports.
DEFAULT_KS = (1, 3, 10)

# The pre-c043c58 tokenizer: contiguous word runs, so a CJK sentence is
# one giant token. Kept verbatim for the A/B arm — do not "fix" it.
_LEGACY_TOKEN = re.compile(r"\b\w+\b")


@dataclass
class RetrievalCase:
    """One query with the document(s) retrieval must surface."""

    id: str
    query: str
    lang: str
    expected_doc_ids: list[str]


def load_retrieval_cases(path: str | Path | None = None) -> list[RetrievalCase]:
    """Load the retrieval golden set (validated against the corpus by
    the schema test, so loader errors stay load errors)."""
    source = Path(path) if path else RETRIEVAL_GOLDEN_SET_FILE
    data = json.loads(source.read_text(encoding="utf-8"))
    return [
        RetrievalCase(
            id=case["id"],
            query=case["query"],
            lang=case["lang"],
            expected_doc_ids=list(case["expected_doc_ids"]),
        )
        for case in data["cases"]
    ]


def load_demo_corpus(path: str | Path | None = None) -> list[dict[str, Any]]:
    """The seeded demo KB in KeywordSearch index-document shape.

    One document per corpus doc (docs, not chunks) — the corpus is
    policy-card sized, and doc-level ids are what the golden set pins.
    """
    source = Path(path) if path else DEMO_CORPUS_FILE
    data = json.loads(source.read_text(encoding="utf-8"))
    return [
        {
            "id": doc["doc_id"],
            "content": doc["content"],
            "metadata": {"lang": doc["lang"], "topic": doc["topic"]},
        }
        for doc in data["docs"]
    ]


def legacy_wholeword_terms(text: str) -> Counter[str]:
    """Term frequencies under the pre-c043c58 whole-word tokenizer.

    Contiguous CJK has no word boundaries under ``\\b\\w+\\b``, so a
    whole CJK sentence becomes ONE token — a sub-phrase query then
    shares zero terms with the document (review #6's 0-hit repro). This
    arm exists to quantify exactly that regression baseline.
    """
    return Counter(_LEGACY_TOKEN.findall(text.lower()))


class LegacyWholeWordKeywordSearch(KeywordSearch):
    """KeywordSearch with the pre-c043c58 tokenizer swapped in — the A/B
    arm differs from production in exactly one dimension."""

    def _extract_terms(self, text: str) -> Counter[str]:
        return legacy_wholeword_terms(text)


class LexicalVectorClient(VectorClient):
    """Deterministic stand-in for the embedding+Qdrant leg.

    Ranks by normalized term overlap between the query and each document
    (production tokenizer), ties broken by document id. It exists so
    RRF fusion and pool width are exercised in the production shape
    without a model download or a vector store; a live run passes the
    real client instead. Correlated with the keyword leg by construction
    — see the honesty note in the module docstring.
    """

    def __init__(self, corpus: list[dict[str, Any]]) -> None:
        tokenizer = KeywordSearch()
        self._docs = {doc["id"]: doc for doc in corpus}
        self._doc_terms = {
            doc_id: set(tokenizer._extract_terms(doc["content"]))
            for doc_id, doc in self._docs.items()
        }

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Not supported — the stand-in is read-only over its corpus."""
        return [doc.id for doc in documents]

    async def delete(self, document_ids: list[str]) -> None:  # noqa: ARG002  # VectorClient API conformance
        """Not supported — the stand-in is read-only over its corpus."""

    async def get_document(self, document_id: str) -> Document | None:  # noqa: ARG002  # VectorClient API conformance
        """Not supported — the stand-in is read-only over its corpus."""
        return None

    async def update(self, document: Document) -> None:  # noqa: ARG002  # VectorClient API conformance
        """Not supported — the stand-in is read-only over its corpus."""

    async def search(self, request: VectorSearchRequest) -> list[SearchResult]:
        tokenizer = KeywordSearch()
        query_terms = set(tokenizer._extract_terms(request.query))
        if not query_terms:
            return []
        scored: list[tuple[float, str]] = []
        for doc_id, terms in self._doc_terms.items():
            if not terms:
                continue
            overlap = len(query_terms & terms)
            if overlap > 0:
                scored.append((overlap / len(query_terms), doc_id))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [
            SearchResult(
                document_id=doc_id,
                content=self._docs[doc_id]["content"],
                score=score,
                metadata=self._docs[doc_id]["metadata"],
            )
            for score, doc_id in scored[: request.top_k]
        ]


def ranking_metrics(
    rankings: list[list[str]],
    expected: list[list[str]],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, float]:
    """recall@k (any expected doc in the first k) plus MRR.

    A case's golden doc being unreachable at width k but present at
    width 10 is exactly what the pool-width comparison needs to see, so
    one ranking per case at max width feeds every k by truncation.
    """
    n = len(rankings)
    metrics: dict[str, float] = {f"recall@{k}": 0.0 for k in ks}
    metrics["mrr"] = 0.0
    if n == 0:
        return metrics
    for ranking, expected_ids in zip(rankings, expected, strict=True):
        expected_set = set(expected_ids)
        for k in ks:
            if expected_set & set(ranking[:k]):
                metrics[f"recall@{k}"] += 1.0
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in expected_set:
                metrics["mrr"] += 1.0 / rank
                break
    return {key: value / n for key, value in metrics.items()}


async def _keyword_rankings(
    cases: list[RetrievalCase],
    corpus: list[dict[str, Any]],
    keyword_cls: type[KeywordSearch],
    max_k: int,
) -> list[list[str]]:
    keyword = keyword_cls()
    await keyword.rebuild(corpus)
    rankings: list[list[str]] = []
    for case in cases:
        results = await keyword.search(VectorSearchRequest(query=case.query, top_k=max_k))
        rankings.append([result.document_id for result in results])
    return rankings


async def _hybrid_rankings(
    cases: list[RetrievalCase],
    corpus: list[dict[str, Any]],
    max_k: int,
) -> list[list[str]]:
    keyword = KeywordSearch()
    await keyword.rebuild(corpus)
    hybrid = HybridSearchService(
        vector_client=LexicalVectorClient(corpus),
        keyword_search=keyword,
    )
    rankings: list[list[str]] = []
    for case in cases:
        try:
            results = await hybrid.search(VectorSearchRequest(query=case.query, top_k=max_k))
        except VectorClientError:
            # Both legs empty for this query — an honest zero-recall
            # data point, not a harness crash.
            results = []
        rankings.append([result.document_id for result in results])
    return rankings


def _sliced_metrics(
    cases: list[RetrievalCase],
    rankings: list[list[str]],
    ks: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    expected = [case.expected_doc_ids for case in cases]
    slices = {"overall": ranking_metrics(rankings, expected, ks)}
    for lang in ("zh", "en"):
        indices = [i for i, case in enumerate(cases) if case.lang == lang]
        slices[lang] = ranking_metrics(
            [rankings[i] for i in indices],
            [expected[i] for i in indices],
            ks,
        )
    return slices


async def run_retrieval_baseline(
    cases: list[RetrievalCase] | None = None,
    corpus: list[dict[str, Any]] | None = None,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict[str, Any]:
    """Run the three comparison arms over the golden set.

    Arms:
        keyword_bigram            — production BM25 leg (true numbers)
        keyword_legacy_wholeword  — pre-c043c58 tokenizer A/B
        hybrid_rrf_lexical_vector — both legs + RRF (lexical stand-in)
    """
    cases = cases if cases is not None else load_retrieval_cases()
    corpus = corpus if corpus is not None else load_demo_corpus()
    max_k = max(ks)

    arms: dict[str, dict[str, dict[str, float]]] = {
        "keyword_bigram": _sliced_metrics(
            cases, await _keyword_rankings(cases, corpus, KeywordSearch, max_k), ks
        ),
        "keyword_legacy_wholeword": _sliced_metrics(
            cases,
            await _keyword_rankings(cases, corpus, LegacyWholeWordKeywordSearch, max_k),
            ks,
        ),
    }
    arms["hybrid_rrf_lexical_vector"] = _sliced_metrics(
        cases, await _hybrid_rankings(cases, corpus, max_k), ks
    )
    return {"cases": len(cases), "ks": list(ks), "arms": arms}


def format_baseline_table(report: dict[str, Any]) -> str:
    """Render the baseline as a monospace table for the CLI runner."""
    ks = report["ks"]
    header = (
        f"{'arm':<28} {'slice':<8} "
        + " ".join(f"{'recall@' + str(k):>10}" for k in ks)
        + f" {'mrr':>8}"
    )
    lines = [
        f"retrieval baseline — {report['cases']} cases",
        header,
        "-" * len(header),
    ]
    for arm_name, arm in report["arms"].items():
        for slice_name in ("overall", "zh", "en"):
            slice_ = arm[slice_name]
            cells = " ".join(f"{slice_[f'recall@{k}']:>10.3f}" for k in ks)
            lines.append(f"{arm_name:<28} {slice_name:<8} {cells} {slice_['mrr']:>8.3f}")
    return "\n".join(lines)
