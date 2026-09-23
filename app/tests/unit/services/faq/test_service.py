"""FAQService: curated semantic match, lazy table build, degradation.

Industry baseline (Zendesk canned responses / 阿里小蜜 FAQ 知识库):
top questions get pre-approved deterministic answers. Pinned here:
matching is threshold-gated, the embedding table is built once, and
every failure mode degrades to a miss — the fast path never blocks
the chat pipeline.
"""

import json
from typing import Any

from app.services.faq import FAQEntry, FAQService, create_faq_service, load_faq_entries
from app.services.faq.store import DEFAULT_FAQ_FILE


class FakeEmbeddings:
    """Deterministic embedding service: known strings map to 2-D vectors."""

    def __init__(self, mapping: dict[str, list[float]], fail_on_bulk: bool = False) -> None:
        self.mapping = dict(mapping)
        self.fail_on_bulk = fail_on_bulk
        self.bulk_calls = 0
        self.query_texts: list[str] = []

    async def embed(self, texts: list[str]) -> "SimpleEmbedding":
        if len(texts) > 1:
            self.bulk_calls += 1
            if self.fail_on_bulk:
                raise RuntimeError("embedding provider down")
            return SimpleEmbedding([self._vec(t) for t in texts])
        self.query_texts.append(texts[0])
        return SimpleEmbedding([self._vec(texts[0])])

    def _vec(self, text: str) -> list[float]:
        return self.mapping.get(text, [0.0, 0.0, 1.0])


class SimpleEmbedding:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings


# 3-D anchor space: x = returns axis, y = refund axis, z = "unrelated".
# Unknown strings embed to [0, 0, 1] — orthogonal to every anchor, so an
# unrelated query scores cos 0 against the whole table.
FAQ1 = FAQEntry(
    faq_id="returns",
    question="退货政策是什么",
    answer="支持 7 天无理由退货。",
    variants=["怎么退货", "如何申请退货"],
)
FAQ2 = FAQEntry(
    faq_id="refund_time",
    question="退款多久到账",
    answer="1-3 个工作日原路退回。",
)

MAPPING = {
    FAQ1.question: [1.0, 0.0, 0.0],
    FAQ1.variants[0]: [0.99, 0.05, 0.0],
    FAQ1.variants[1]: [0.95, 0.2, 0.0],
    FAQ2.question: [0.0, 1.0, 0.0],
    "对角线问题": [1.0, 1.0, 0.0],
}


def _service(threshold: float = 0.8, **embed_kwargs: Any) -> tuple[FAQService, FakeEmbeddings]:
    fake = FakeEmbeddings(MAPPING, **embed_kwargs)
    return FAQService(fake, [FAQ1, FAQ2], threshold), fake


# ── Loading ────────────────────────────────────────────────────────────────


class TestLoading:
    def test_bundled_file_loads(self):
        entries = load_faq_entries(DEFAULT_FAQ_FILE)
        assert len(entries) >= 10
        assert all(e.question and e.answer and e.faq_id for e in entries)
        ids = {e.faq_id for e in entries}
        assert "returns_policy" in ids and "refund_timeline" in ids

    def test_missing_file_degrades_to_empty(self, tmp_path):
        assert load_faq_entries(tmp_path / "nope.json") == []

    def test_malformed_json_degrades_to_empty(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_faq_entries(path) == []

    def test_malformed_entries_skipped_and_disabled_filtered(self, tmp_path):
        path = tmp_path / "mixed.json"
        path.write_text(
            json.dumps(
                [
                    {"id": "ok", "question": "q", "answer": "a"},
                    {"id": "no_question", "answer": "a"},  # malformed: skipped
                    {"id": "off", "question": "q2", "answer": "a2", "enabled": False},
                ]
            ),
            encoding="utf-8",
        )
        entries = load_faq_entries(path)
        assert [e.faq_id for e in entries] == ["ok"]

    def test_create_returns_none_when_nothing_to_serve(self, tmp_path):
        empty = tmp_path / "empty.json"
        empty.write_text("[]", encoding="utf-8")
        assert create_faq_service(FakeEmbeddings({}), data_file=str(empty)) is None


# ── Matching ───────────────────────────────────────────────────────────────


class TestMatching:
    async def test_paraphrase_hits_canonical_answer(self):
        service, _ = _service()
        entry = await service.match("怎么退货")
        assert entry is not None and entry.faq_id == "returns"

    async def test_best_match_wins(self):
        service, _ = _service()
        entry = await service.match("退款多久到账")
        assert entry is not None and entry.faq_id == "refund_time"

    async def test_unrelated_query_misses(self):
        service, _ = _service()
        assert await service.match("你们卖手机吗") is None  # z-axis → cos 0 against all

    async def test_threshold_boundary_is_inclusive(self):
        # "对角线问题" → [1,1,0] against the [1,0,0] anchor. The threshold
        # is computed with the service's own _cosine so it is bit-identical
        # to the best score — pinning that the comparison is >=, not >.
        from app.services.faq.store import _cosine

        anchor_only = FAQEntry(faq_id="returns", question=FAQ1.question, answer="a")
        service = FAQService(
            FakeEmbeddings(MAPPING), [anchor_only], _cosine([1.0, 1.0, 0.0], [1.0, 0.0, 0.0])
        )
        entry = await service.match("对角线问题")
        assert entry is not None and entry.faq_id == "returns"

    async def test_blank_message_misses_without_embedding(self):
        service, fake = _service()
        assert await service.match("   ") is None
        assert fake.query_texts == []


# ── Table lifecycle ────────────────────────────────────────────────────────


class TestTableLifecycle:
    async def test_table_embedded_once_across_matches(self):
        service, fake = _service()
        await service.match("怎么退货")
        await service.match("退款多久到账")
        assert fake.bulk_calls == 1

    async def test_build_failure_is_permanent_miss(self):
        service, fake = _service(fail_on_bulk=True)
        assert await service.match("怎么退货") is None
        assert await service.match("怎么退货") is None  # not retried per request
        assert fake.bulk_calls == 1

    async def test_query_embed_failure_does_not_break_table(self):
        # First call fails at query embedding, then works — the table
        # must still be usable afterwards.
        fake = FakeEmbeddings(MAPPING)
        service = FAQService(fake, [FAQ1], 0.8)

        original_embed = fake.embed
        armed = False

        async def flaky_embed(texts: list[str]) -> "SimpleEmbedding":
            nonlocal armed
            if not armed:
                armed = True
                raise RuntimeError("transient")
            return await original_embed(texts)

        fake.embed = flaky_embed  # type: ignore[method-assign]  # noqa: B010
        assert await service.match("怎么退货") is None  # query embed failed
        assert await service.match("怎么退货") is not None  # recovered

    async def test_empty_entries_never_embed(self):
        fake = FakeEmbeddings({})
        service = FAQService(fake, [], 0.8)
        assert await service.match("anything") is None
        assert fake.bulk_calls == 0
        assert fake.query_texts == []
