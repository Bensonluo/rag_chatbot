"""User-facts recall cache metrics (process-level, Prometheus text exposure)."""

from prometheus_client import REGISTRY, Counter, Histogram

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

# Chat stream 时效 (GB/T 47746—2026 响应速度 dimension, AI side). TTFT is
# the canonical perceived-latency metric for streaming bots — industry
# target <1s, drop-off rises ~7-10% per extra waiting second. Buckets are
# dense below the 1s SLO and sparse up to the 120s stream budget.
CHAT_FIRST_TOKEN_SECONDS = Histogram(
    "chat_first_token_seconds",
    "Seconds from stream start to the first content chunk (TTFT; heartbeats excluded)",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0),
    registry=REGISTRY,
)

CHAT_STREAM_DURATION_SECONDS = Histogram(
    "chat_stream_duration_seconds",
    "Total seconds a chat stream stayed open (any exit path)",
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 120.0, 300.0),
    registry=REGISTRY,
)

CHAT_STREAM_OUTCOMES = Counter(
    "chat_stream_outcomes_total",
    "Final outcome of a chat stream by exit path",
    labelnames=["outcome"],
    registry=REGISTRY,
)

# L0 exact-answer cache (docs/cache-layering-plan.md layer 0). Hits
# skip the whole pipeline (~5s → ~5ms); misses are the normal path and
# include Redis-outage degradation. The hit ratio is the demo cache
# panel's number and the capacity story's top-of-funnel lever: storm
# traffic is highly repetitive, so the cache works hardest exactly
# when the pipeline is under the most pressure.
ANSWER_CACHE_HITS = Counter(
    "answer_cache_hits_total",
    "Chat turns served from the L0 exact-answer cache",
    registry=REGISTRY,
)

ANSWER_CACHE_MISSES = Counter(
    "answer_cache_misses_total",
    "L0 answer-cache lookups that missed (no entry, disabled epoch, or Redis outage)",
    registry=REGISTRY,
)
