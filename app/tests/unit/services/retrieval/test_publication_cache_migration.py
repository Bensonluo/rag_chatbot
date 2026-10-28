"""Previously serialized evidence cannot migrate by being replayed as a new answer."""

import hashlib
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from app.services.chat.answer_cache import AnswerCacheService, CachedAnswer
from app.services.chat.semantic_cache import SemanticCacheService, encode_vector
from app.services.retrieval.retrieval_cache import RetrievalCacheService


class LocalStore:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, **kwargs):
        self.values[key] = value

    async def hget(self, name, key):
        return self.hashes.get(name, {}).get(key)

    async def hgetall(self, name):
        return dict(self.hashes.get(name, {}))

    async def hset(self, name, key, value):
        self.hashes.setdefault(name, {})[key] = value

    async def hlen(self, name):
        return len(self.hashes.get(name, {}))

    async def expire(self, *args):
        pass


async def epoch() -> str:
    return "unchanged-epoch"


async def test_l0_old_payload_is_not_served_even_when_redis_epoch_did_not_rotate(monkeypatch):
    monkeypatch.setattr("app.services.chat.answer_cache.get_kb_epoch", epoch)
    monkeypatch.setattr("app.services.chat.answer_cache.record_response_index", AsyncMock())
    store: Any = LocalStore()
    service = AnswerCacheService(redis_url="redis://not-used", ttl_seconds=60)
    service._redis = store
    digest = hashlib.sha256("\x00".join(("q", await epoch(), "", "")).encode()).hexdigest()
    old_key = f"answer:{digest}"
    store.values[old_key] = json.dumps(
        {"response": "old unverified reply", "intent": "chitchat", "sources": ["d"]}
    )
    assert await service.get("q") is None
    await service.put("q", response="new verified reply", sources=["d"], intent="chitchat")
    answer = await service.get("q")
    assert answer is not None and answer.response == "new verified reply"
    assert old_key in store.values


async def test_l1_old_graph_payload_is_not_replayed_under_the_same_epoch(monkeypatch):
    monkeypatch.setattr("app.services.chat.semantic_cache.record_response_index", AsyncMock())
    store = LocalStore()
    store.hashes["semantic_answers:unchanged-epoch"] = {
        "q": json.dumps(
            {
                "response": "old truncated graph reply",
                "sources": ["graph"],
                "intent": "chitchat",
                "lang": "en",
                "vec": encode_vector([1.0, 0.0]),
            }
        )
    }
    settings = SimpleNamespace(
        SEMANTIC_CACHE_ENABLED=True,
        SEMANTIC_CACHE_SIMILARITY_THRESHOLD=0.9,
        SEMANTIC_CACHE_SERVED_INTENTS="chitchat",
        SEMANTIC_CACHE_MAX_ENTRIES=10,
        SEMANTIC_CACHE_TTL_SECONDS=60,
    )
    service = SemanticCacheService(
        SimpleNamespace(embed=AsyncMock(return_value=SimpleNamespace(embeddings=[[1.0, 0.0]]))),
        store,
        epoch_provider=epoch,
        settings=settings,
    )
    assert await service.get("hello") is None
    await service.put("hello", CachedAnswer("new verified reply", ["d"], "chitchat"))
    answer = await service.get("hello")
    assert answer is not None and answer.response == "new verified reply"
    assert "semantic_answers:unchanged-epoch" in store.hashes


async def test_l2_old_document_payload_is_not_used_to_repopulate_new_answer_caches():
    store = LocalStore()
    field = hashlib.sha256("\x00".join(("q", "{}")).encode()).hexdigest()
    store.hashes["retrieval_results:unchanged-epoch"] = {
        field: json.dumps([{"document_id": "unproved", "content": "old"}])
    }
    settings = SimpleNamespace(
        RETRIEVAL_CACHE_ENABLED=True, RETRIEVAL_CACHE_MAX_ENTRIES=10, RETRIEVAL_CACHE_TTL_SECONDS=60
    )
    service = RetrievalCacheService(store, epoch_provider=epoch, settings=settings)
    assert await service.get("q", {}) is None
    docs = [{"document_id": "published", "content": "new", "metadata": {"version_id": "v1"}}]
    await service.put("q", {}, docs)
    assert await service.get("q", {}) == docs
    assert "retrieval_results:unchanged-epoch" in store.hashes
