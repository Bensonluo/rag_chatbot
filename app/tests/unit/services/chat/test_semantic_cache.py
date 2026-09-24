"""L1 semantic answer cache: near-duplicate turns replay without the LLM.

Cache-layering plan L1 (docs/cache-layering-plan.md): rephrased hot
questions (“7天能退吗” vs “七天无理由退货多久”) should replay a cached
answer in milliseconds. The plan's hard boundaries are pinned here:
only allowlisted small-talk/FAQ intents are cached (policy/RAG answers
are L0-exact-only — semantic nearness across paraphrased policy
questions is the “唯一真风险层”), entries are epoch-scoped so a KB
update invalidates them, the store is bounded so it cannot be flooded,
and every failure fails open — the cache is an optimization, never a
dependency.
"""

import base64
import math
from array import array
from typing import Any

from app.services.chat.answer_cache import CachedAnswer
from app.services.chat.semantic_cache import SemanticCacheService


class FakeEmbeddings:
    """Deterministic embedding seam: fixed vector per exact text."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    async def embed(self, texts: list[str]) -> Any:
        return SimpleEmbeddings([self.mapping.get(t, [1.0, 0.0]) for t in texts])


class SimpleEmbeddings:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.embeddings = vectors


class FakeRedis:
    """In-memory hash store sufficient for the semantic cache."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}

    async def hset(self, name: str, key: str, value: str) -> None:
        self.hashes.setdefault(name, {})[key] = value

    async def hget(self, name: str, key: str) -> str | None:
        return self.hashes.get(name, {}).get(key)

    async def hgetall(self, name: str) -> dict[str, str]:
        return dict(self.hashes.get(name, {}))

    async def hlen(self, name: str) -> int:
        return len(self.hashes.get(name, {}))

    async def expire(self, name: str, ttl: int) -> None:
        self.last_expire = (name, ttl)


class RaisingRedis:
    async def hset(self, name: str, key: str, value: str) -> None:
        raise ConnectionError("redis down")

    async def hget(self, name: str, key: str) -> str | None:
        raise ConnectionError("redis down")

    async def hgetall(self, name: str) -> dict[str, str]:
        raise ConnectionError("redis down")

    async def hlen(self, name: str) -> int:
        raise ConnectionError("redis down")

    async def expire(self, name: str, ttl: int) -> None:
        raise ConnectionError("redis down")


def _encode_vec(vector: list[float]) -> str:
    return base64.b64encode(array("f", vector).tobytes()).decode()


def _entry(answer: CachedAnswer, vector: list[float]) -> str:
    import json

    return json.dumps(
        {
            "response": answer.response,
            "sources": answer.sources,
            "intent": answer.intent,
            "vec": _encode_vec(vector),
        }
    )


def _epoch_provider(epoch: str = "e1") -> Any:
    async def provider() -> str:
        return epoch

    return provider


def _make_service(
    redis: Any = None,
    enabled: bool = True,
    threshold: float = 0.92,
    max_entries: int = 256,
    intents: str = "greeting,chitchat",
    epoch: str = "e1",
    embeddings: FakeEmbeddings | None = None,
) -> tuple[SemanticCacheService, FakeEmbeddings]:
    from app.config.settings import settings

    settings.SEMANTIC_CACHE_ENABLED = enabled
    settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD = threshold
    settings.SEMANTIC_CACHE_MAX_ENTRIES = max_entries
    settings.SEMANTIC_CACHE_SERVED_INTENTS = intents
    emb = embeddings or FakeEmbeddings(
        {
            "七天无理由退货是多久": [1.0, 0.0],
            "7天能退货吗": [0.999, 0.0447],
            "今天天气怎么样": [0.0, 1.0],
        }
    )
    service = SemanticCacheService(
        embedding_service=emb,
        redis_client=redis if redis is not None else FakeRedis(),
        epoch_provider=_epoch_provider(epoch),
        settings=settings,
    )
    return service, emb


def _answer(intent: str = "chitchat") -> CachedAnswer:
    return CachedAnswer(
        response="签收后7天内可申请无理由退货。",
        sources=["doc-refund"],
        intent=intent,
    )


class TestSemanticCacheHits:
    async def test_identical_message_replays(self):
        service, _ = _make_service()
        await service.put("七天无理由退货是多久", _answer())

        cached = await service.get("七天无理由退货是多久")
        assert cached is not None
        assert cached.response == "签收后7天内可申请无理由退货。"
        assert cached.sources == ["doc-refund"]
        assert cached.intent == "chitchat"

    async def test_paraphrase_above_threshold_hits(self):
        service, _ = _make_service()
        await service.put("七天无理由退货是多久", _answer())

        cached = await service.get("7天能退货吗")
        assert cached is not None
        assert cached.response == "签收后7天内可申请无理由退货。"


class TestSemanticCacheBoundaries:
    async def test_below_threshold_misses(self):
        service, _ = _make_service()
        await service.put("七天无理由退货是多久", _answer())

        assert await service.get("今天天气怎么样") is None

    async def test_policy_intent_rejected_at_write(self):
        redis = FakeRedis()
        service, _ = _make_service(redis=redis)
        await service.put(
            "退款政策是什么",
            CachedAnswer(response="7天无理由。", sources=["doc"], intent="question"),
        )

        assert await redis.hlen("semantic_answers:e1") == 0

    async def test_disabled_flag_is_full_noop(self):
        redis = FakeRedis()
        service, _ = _make_service(redis=redis, enabled=False)
        await service.put("七天无理由退货是多久", _answer())

        assert await redis.hlen("semantic_answers:e1") == 0
        assert await service.get("七天无理由退货是多久") is None

    async def test_full_store_refuses_new_entries(self):
        redis = FakeRedis()
        service, _ = _make_service(redis=redis, max_entries=1)
        await service.put("七天无理由退货是多久", _answer())
        await service.put(
            "7天能退货吗", CachedAnswer(response="另一条", sources=[], intent="chitchat")
        )

        assert await redis.hlen("semantic_answers:e1") == 1

    async def test_epoch_change_isolates_entries(self):
        # One shared store: isolation must come from the epoch namespace,
        # not from separate storage.
        redis = FakeRedis()
        writer, _ = _make_service(redis=redis, epoch="e1")
        await writer.put("七天无理由退货是多久", _answer())

        reader, _ = _make_service(redis=redis, epoch="e2")
        assert await reader.get("七天无理由退货是多久") is None

    async def test_unavailable_epoch_refuses_reads_and_writes(self):
        # EPOCH_UNAVAILABLE means KB-invalidation is not currently
        # enforceable; populating or serving that namespace would let
        # entries outlive the KB they answer for (L0 doctrine).
        from app.services.retrieval.kb_epoch import EPOCH_UNAVAILABLE

        redis = FakeRedis()
        writer, _ = _make_service(redis=redis, epoch=EPOCH_UNAVAILABLE)
        await writer.put("七天无理由退货是多久", _answer())

        reader, _ = _make_service(redis=redis, epoch=EPOCH_UNAVAILABLE)
        assert await reader.get("七天无理由退货是多久") is None
        assert redis.hashes == {}


class TestSemanticCacheFailOpen:
    async def test_redis_outage_misses_and_never_raises(self):
        service, _ = _make_service(redis=RaisingRedis())
        assert await service.get("七天无理由退货是多久") is None
        await service.put("七天无理由退货是多久", _answer())  # must not raise

    async def test_embedding_outage_misses_and_never_raises(self):
        class BrokenEmbeddings:
            async def embed(self, texts: list[str]) -> Any:
                raise ConnectionError("embedding backend down")

        from app.config.settings import settings

        settings.SEMANTIC_CACHE_ENABLED = True
        service = SemanticCacheService(
            embedding_service=BrokenEmbeddings(),
            redis_client=FakeRedis(),
            epoch_provider=_epoch_provider("e1"),
            settings=settings,
        )
        assert await service.get("任何消息") is None


class TestVectorCodec:
    def test_roundtrip_preserves_cosine(self):
        from app.services.chat.semantic_cache import decode_vector, encode_vector

        v = [0.1, -0.2, 0.3]
        decoded = decode_vector(encode_vector(v))
        assert math.isclose(
            sum(a * b for a, b in zip(v, decoded, strict=True)),
            sum(a * a for a in v),
            rel_tol=1e-5,
        )
