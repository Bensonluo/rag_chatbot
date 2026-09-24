"""Downvote → cache-pyramid eviction contract.

The feedback loop is the user-verified quality signal (the one-shot
north-star proxy): an answer a user downvoted was NOT a resolution —
and without this contract the pyramid kept replaying it from L0/L1 in
~5ms until TTL/epoch rotation, at storm scale re-serving the failure
to every near-duplicate ask. These tests pin the full loop:

- **Write side**: every L0/L1 ``put`` records a response-digest
  reverse index entry (the L0 key digest cannot be derived from the
  downvoted message alone — the feedback layer only knows the
  response text).
- **Eviction**: one downvote removes every indexed entry for that
  response across both layers; the answer stops replaying.
- **Fail-open iron law**: unknown responses, an unreadable epoch, and
  a Redis outage all degrade to a no-op — eviction is an optimization
  of quality, never a dependency of the feedback endpoint.
"""

import hashlib
import json
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from redis.asyncio import Redis

from app.services.chat.answer_cache import AnswerCacheService, normalize_message
from app.services.chat.cache_eviction import (
    evict_downvoted_response,
    response_digest,
)
from app.services.chat.semantic_cache import CachedAnswer, SemanticCacheService


class _FakeRedis:
    """In-memory string+hash surface covering L0, L1, and both indexes."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.fail = False

    async def get(self, key: str) -> Any:
        if self.fail:
            raise ConnectionError("redis down")
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int = 0) -> Any:
        if self.fail:
            raise ConnectionError("redis down")
        self.store[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        if self.fail:
            raise ConnectionError("redis down")
        removed = sum(1 for k in keys if self.store.pop(k, None) is not None)
        return removed

    async def hset(self, name: str, key: str, value: str) -> None:
        if self.fail:
            raise ConnectionError("redis down")
        self.hashes.setdefault(name, {})[key] = value

    async def hget(self, name: str, key: str) -> Any:
        if self.fail:
            raise ConnectionError("redis down")
        return self.hashes.get(name, {}).get(key)

    async def hgetall(self, name: str) -> dict[str, str]:
        if self.fail:
            raise ConnectionError("redis down")
        return dict(self.hashes.get(name, {}))

    async def hlen(self, name: str) -> int:
        if self.fail:
            raise ConnectionError("redis down")
        return len(self.hashes.get(name, {}))

    async def hdel(self, name: str, *keys: str) -> int:
        if self.fail:
            raise ConnectionError("redis down")
        bucket = self.hashes.get(name, {})
        removed = sum(1 for k in keys if bucket.pop(k, None) is not None)
        return removed

    async def expire(self, name: str, ttl: int) -> None:
        if self.fail:
            raise ConnectionError("redis down")


_EPOCH = "epoch-test"


@pytest.fixture(autouse=True)
def _hermetic_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both the L0 service and the evictor must read the same epoch."""
    monkeypatch.setattr(
        "app.services.chat.answer_cache.get_kb_epoch", AsyncMock(return_value=_EPOCH)
    )
    monkeypatch.setattr(
        "app.services.chat.cache_eviction.get_kb_epoch", AsyncMock(return_value=_EPOCH)
    )


def _l0_service(fake: _FakeRedis) -> AnswerCacheService:
    svc = AnswerCacheService(redis_url="redis://localhost:6379/0", ttl_seconds=60)
    svc._redis = cast("Redis[str]", fake)
    return svc


def _l1_service(fake: _FakeRedis) -> SemanticCacheService:
    from app.config.settings import settings

    settings.SEMANTIC_CACHE_ENABLED = True
    settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD = 0.5
    settings.SEMANTIC_CACHE_MAX_ENTRIES = 256
    settings.SEMANTIC_CACHE_SERVED_INTENTS = "chitchat"

    class _FixedEmbeddings:
        async def embed(self, texts: list[str]) -> Any:
            class _R:
                embeddings = [[1.0, 0.0] for _ in texts]

            return _R()

    async def _epoch() -> str:
        return _EPOCH

    return SemanticCacheService(
        embedding_service=_FixedEmbeddings(),
        redis_client=fake,
        epoch_provider=_epoch,
        settings=settings,
    )


def _indexed(fake: _FakeRedis, index_name: str, response: str) -> list[str]:
    raw = fake.hashes.get(index_name, {}).get(response_digest(response))
    return json.loads(raw) if raw else []


class TestResponseDigest:
    def test_digest_is_sha256_hex_of_response(self):
        assert response_digest("bad answer") == hashlib.sha256(b"bad answer").hexdigest()


class TestReverseIndexWrites:
    async def test_l0_put_records_the_answer_key_under_the_response_digest(self):
        fake = _FakeRedis()
        svc = _l0_service(fake)
        await svc.put(
            "退货政策是什么", response="7天无理由退货", sources=["doc-1"], intent="question"
        )
        entries = _indexed(fake, f"answer_resp_index:{_EPOCH}", "7天无理由退货")
        assert len(entries) == 1

    async def test_distinct_queries_with_same_response_are_both_indexed(self):
        fake = _FakeRedis()
        svc = _l0_service(fake)
        await svc.put("q one", response="同一个回答", sources=["d"], intent="question")
        await svc.put("q two", response="同一个回答", sources=["d"], intent="question")
        entries = _indexed(fake, f"answer_resp_index:{_EPOCH}", "同一个回答")
        assert len(entries) == 2
        assert len(set(entries)) == 2

    async def test_l1_put_records_the_field_under_the_response_digest(self):
        fake = _FakeRedis()
        svc = _l1_service(fake)
        await svc.put(
            "你好呀",
            CachedAnswer(response="你好！", sources=["doc-1"], intent="chitchat"),
        )
        entries = _indexed(fake, f"semantic_resp_index:{_EPOCH}", "你好！")
        assert entries == [normalize_message("你好呀")]


class TestEviction:
    async def test_downvote_evicts_the_l0_entry(self):
        fake = _FakeRedis()
        svc = _l0_service(fake)
        await svc.put(
            "退货政策是什么", response="7天无理由退货", sources=["doc-1"], intent="question"
        )
        assert await svc.get("退货政策是什么") is not None

        evicted = await evict_downvoted_response("7天无理由退货", redis_client=fake)

        assert evicted >= 1
        assert await svc.get("退货政策是什么") is None

    async def test_downvote_evicts_the_l1_entry(self):
        fake = _FakeRedis()
        svc = _l1_service(fake)
        await svc.put(
            "你好呀",
            CachedAnswer(response="你好！", sources=["doc-1"], intent="chitchat"),
        )
        assert await svc.get("你好呀") is not None

        evicted = await evict_downvoted_response("你好！", redis_client=fake)

        assert evicted >= 1
        assert await svc.get("你好呀") is None

    async def test_index_field_is_removed_after_eviction(self):
        fake = _FakeRedis()
        svc = _l0_service(fake)
        await svc.put("q", response="a bad answer", sources=["d"], intent="question")

        await evict_downvoted_response("a bad answer", redis_client=fake)

        assert response_digest("a bad answer") not in fake.hashes.get(
            f"answer_resp_index:{_EPOCH}", {}
        )

    async def test_unknown_response_evicts_nothing(self):
        fake = _FakeRedis()
        assert await evict_downvoted_response("从未缓存的回答", redis_client=fake) == 0

    async def test_empty_response_is_a_no_op(self):
        fake = _FakeRedis()
        assert await evict_downvoted_response("", redis_client=fake) == 0

    async def test_unreadable_epoch_evicts_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.services.retrieval.kb_epoch import EPOCH_UNAVAILABLE

        monkeypatch.setattr(
            "app.services.chat.cache_eviction.get_kb_epoch",
            AsyncMock(return_value=EPOCH_UNAVAILABLE),
        )
        fake = _FakeRedis()
        assert await evict_downvoted_response("some answer", redis_client=fake) == 0
        assert fake.hashes == {}

    async def test_redis_outage_is_fail_open_not_raise(self):
        fake = _FakeRedis()
        fake.fail = True
        assert await evict_downvoted_response("some answer", redis_client=fake) == 0
