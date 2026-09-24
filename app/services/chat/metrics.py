"""User-facts recall cache metrics (process-level, Prometheus text exposure)."""

from prometheus_client import REGISTRY, Counter

USER_FACTS_CACHE_HITS = Counter(
    "user_facts_cache_hits_total",
    "User-facts recalls served from the per-worker TTL cache",
    registry=REGISTRY,
)

USER_FACTS_CACHE_MISSES = Counter(
    "user_facts_cache_misses_total",
    "User-facts recalls that had to hit the backing store (miss, expiry, or eviction)",
    registry=REGISTRY,
)
