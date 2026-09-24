"""TTL cache for cross-session user-facts recall (Phase B2 read side).

At 800K-1M daily requests the facts provider is consulted on every
generation call (agent node + every generator). Personalization facts
are slow-stale — extraction runs every USER_FACT_EXTRACTION_INTERVAL
turns — so a short per-user TTL trades bounded staleness for the bulk
of those store hits (AWS caching guidance: per-user namespacing; arXiv
ToolCaching 2026: staleness budget should match the data's decay rate).
These tests pin the cache contract: hits, expiry, isolation, eviction,
and the never-cache-failure rule that keeps never-fail-chat intact.
"""

import asyncio
from collections.abc import Awaitable
from typing import Any

from app.services.chat.user_facts_cache import cached_user_facts_provider


class _CountingProvider:
    """Fake facts store counting calls and returning per-user payloads."""

    def __init__(self, *, fail_first: bool = False) -> None:
        self.calls: list[int] = []
        self.fail_first = fail_first
        self._failed_once = False

    async def __call__(self, user_id: int) -> list[str]:
        self.calls.append(user_id)
        if self.fail_first and not self._failed_once:
            self._failed_once = True
            raise RuntimeError("store down")
        return [f"u{user_id}-fact"]


class TestTTLUserFactsCache:
    async def test_second_recall_within_ttl_hits_cache(self):
        provider = _CountingProvider()
        cached = cached_user_facts_provider(provider, ttl_seconds=60.0)

        first = await cached(7)
        second = await cached(7)

        assert first == ["u7-fact"]
        assert second == first
        assert provider.calls == [7]  # one store hit served two recalls

    async def test_expired_entry_refetches(self):
        provider = _CountingProvider()
        cached = cached_user_facts_provider(provider, ttl_seconds=0.05)

        await cached(7)
        await asyncio.sleep(0.06)
        await cached(7)

        assert provider.calls == [7, 7]

    async def test_zero_ttl_disables_caching(self):
        provider = _CountingProvider()
        cached = cached_user_facts_provider(provider, ttl_seconds=0.0)

        await cached(7)
        await cached(7)

        assert provider.calls == [7, 7]

    async def test_users_are_isolated(self):
        """Per-user namespacing: one user's facts never serve another (AWS)."""
        provider = _CountingProvider()
        cached = cached_user_facts_provider(provider, ttl_seconds=60.0)

        assert await cached(1) == ["u1-fact"]
        assert await cached(2) == ["u2-fact"]
        assert provider.calls == [1, 2]

    async def test_lru_eviction_refetches_cold_user(self):
        provider = _CountingProvider()
        cached = cached_user_facts_provider(provider, ttl_seconds=60.0, maxsize=2)

        await cached(1)
        await cached(2)
        await cached(3)  # evicts user 1 (LRU)
        await cached(1)  # cold again → store hit

        assert provider.calls == [1, 2, 3, 1]

    async def test_lru_touch_keeps_hot_user(self):
        provider = _CountingProvider()
        cached = cached_user_facts_provider(provider, ttl_seconds=60.0, maxsize=2)

        await cached(1)
        await cached(2)
        await cached(1)  # touch: user 1 is now hot, user 2 is LRU
        await cached(3)  # evicts user 2 (not the touched user 1)
        await cached(1)  # still warm — no store call

        assert provider.calls == [1, 2, 3]

    async def test_successful_empty_list_is_cached(self):
        """New users dominate: [] from a good read must not hammer the store."""

        class _Empty:
            def __init__(self) -> None:
                self.calls = 0

            async def __call__(self, user_id: int) -> list[str]:
                self.calls += 1
                return []

        provider = _Empty()
        cached = cached_user_facts_provider(provider, ttl_seconds=60.0)

        assert await cached(9) == []
        assert await cached(9) == []
        assert provider.calls == 1

    async def test_failure_propagates_and_is_never_cached(self):
        """A store outage must not freeze [] for the TTL — pass it through
        so the recall wrapper's never-fail-chat handling stays honest."""
        provider = _CountingProvider(fail_first=True)
        cached = cached_user_facts_provider(provider, ttl_seconds=60.0)

        try:
            await cached(7)
            raised = False
        except RuntimeError:
            raised = True
        assert raised

        assert await cached(7) == ["u7-fact"]  # retry reached the store
        assert provider.calls == [7, 7]

    async def test_wrapped_provider_is_still_async_callable(self):
        """The wrapper must stay a drop-in Callable[[int], Awaitable[list[str]]]."""
        provider = _CountingProvider()
        cached = cached_user_facts_provider(provider, ttl_seconds=60.0)

        result: Any = cached(7)
        assert isinstance(result, Awaitable)
        assert await result == ["u7-fact"]

    async def test_metrics_count_hits_and_misses(self):
        """Hit/miss counters give ops the cache's real absorption rate."""
        from prometheus_client import REGISTRY

        provider = _CountingProvider()
        cached = cached_user_facts_provider(provider, ttl_seconds=60.0)

        hits_before = REGISTRY.get_sample_value("user_facts_cache_hits_total") or 0
        misses_before = REGISTRY.get_sample_value("user_facts_cache_misses_total") or 0

        await cached(7)  # miss
        await cached(7)  # hit

        assert REGISTRY.get_sample_value("user_facts_cache_hits_total") == hits_before + 1
        assert REGISTRY.get_sample_value("user_facts_cache_misses_total") == misses_before + 1
