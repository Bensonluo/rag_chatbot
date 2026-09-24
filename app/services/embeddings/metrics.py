"""Prometheus metrics for the embedding cache (L3 of the cache pyramid).

The embedding cache sits under every semantic operation: L1 semantic
lookups embed the query first, and every RAG turn embeds too. The
service is fail-open in both directions, so a dead cache is invisible
in request outcomes — every text silently re-embeds and TTFT climbs
while the symptom alert (ChatTTFTSlow) carries no root-cause signal.
These counters are the mechanism-level truth, same doctrine as the KB
epoch failure counter.

Hits and misses count *per text* (the honest, traffic-proportional
measure for a batch API), and only on a healthy read: a failed read
increments the failure counter instead, so the hit rate stays a
conditional-on-healthy-read metric rather than smearing outage noise
into the miss count.
"""

from prometheus_client import REGISTRY, Counter

EMBEDDING_CACHE_HITS = Counter(
    "embedding_cache_hits_total",
    "Texts served from the embedding cache (per text, not per call)",
    registry=REGISTRY,
)
EMBEDDING_CACHE_MISSES = Counter(
    "embedding_cache_misses_total",
    "Texts freshly embedded after a healthy cache read found no entry",
    registry=REGISTRY,
)
EMBEDDING_CACHE_FAILURES = Counter(
    "embedding_cache_failures_total",
    "Embedding cache Redis operations that failed, by op (read/write)",
    labelnames=["op"],
    registry=REGISTRY,
)
