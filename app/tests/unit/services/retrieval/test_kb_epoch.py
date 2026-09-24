"""KB epoch contract: rotation invalidates epoch-scoped caches, fail-open.

The epoch is the invalidation spine for the L0 answer cache — these
tests pin the two properties everything downstream leans on: a bump
changes the value (stale entries can never match again), and a Redis
outage degrades to a sentinel instead of raising (the epoch is never a
dependency).
"""

from typing import Any

import pytest

from app.services.retrieval import kb_epoch
from app.services.retrieval.kb_epoch import (
    EPOCH_KEY,
    EPOCH_UNAVAILABLE,
    bump_kb_epoch,
    get_kb_epoch,
)


class _FakeRedis:
    """Minimal async surface kb_epoch uses (get / set with nx)."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.fail = False

    async def get(self, key: str) -> Any:
        if self.fail:
            raise ConnectionError("redis down")
        return self.store.get(key)

    async def set(self, key: str, value: str, nx: bool = False) -> Any:
        if self.fail:
            raise ConnectionError("redis down")
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True


@pytest.fixture
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> _FakeRedis:
    fake = _FakeRedis()
    monkeypatch.setattr(kb_epoch, "_client", fake)
    return fake


class TestKBEpoch:
    async def test_first_read_creates_and_is_stable(self, fake_redis: _FakeRedis) -> None:
        first = await get_kb_epoch()
        second = await get_kb_epoch()
        assert first == second
        assert fake_redis.store[EPOCH_KEY] == first

    async def test_bump_rotates_the_epoch(self, fake_redis: _FakeRedis) -> None:
        before = await get_kb_epoch()
        await bump_kb_epoch()
        after = await get_kb_epoch()
        assert after != before

    async def test_redis_outage_returns_sentinel_not_raise(self, fake_redis: _FakeRedis) -> None:
        fake_redis.fail = True
        assert await get_kb_epoch() == EPOCH_UNAVAILABLE

    async def test_redis_outage_bump_does_not_raise(self, fake_redis: _FakeRedis) -> None:
        fake_redis.fail = True
        await bump_kb_epoch()  # must swallow — ingestion must not fail over invalidation
