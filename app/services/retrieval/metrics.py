"""Retrieval funnel metrics.

The filtered-search path (slot-entity metadata filters with an
unfiltered fallback) is invisible without counters: a fallback rate
pinned at 100% means the chunk-metadata contract is never populated and
the filter is pure overhead; a hit rate that decays means ingestion
stopped writing the keys. These counters close that loop — alert on
`retrieval_filter_fallbacks_total / retrieval_filtered_searches_total`.
"""

from prometheus_client import REGISTRY, Counter

# Metrics belong to the process, not to an individual object graph —
# defining them once avoids duplicate-registration errors when multiple
# app instances or test clients import this module.
RETRIEVAL_FILTERED_SEARCHES = Counter(
    "retrieval_filtered_searches_total",
    "Vector searches issued with slot-derived metadata filters",
    registry=REGISTRY,
)
RETRIEVAL_FILTER_FALLBACKS = Counter(
    "retrieval_filter_fallbacks_total",
    "Filtered searches that returned empty and retried unfiltered",
    registry=REGISTRY,
)
# L2 retrieval-result cache (docs/cache-layering-plan.md): hit rate
# measures how much repeated identical traffic the layer absorbs
# before it reaches Qdrant — the layer's whole reason to exist.
RETRIEVAL_CACHE_HITS = Counter(
    "retrieval_cache_hits_total",
    "Retrieval lookups served from the L2 result cache",
    registry=REGISTRY,
)
RETRIEVAL_CACHE_MISSES = Counter(
    "retrieval_cache_misses_total",
    "Retrieval lookups that fell through to the vector/BM25 legs",
    registry=REGISTRY,
)
# The KB epoch is the invalidation spine of the whole cache pyramid
# (L0/L1/L2 all key on it): a sustained read-failure streak silently
# disables every layer (all refuse reads), and a failed bump serves
# stale answers until TTL. Fail-open means neither is visible in
# request outcomes — this counter is the only signal the spine is
# unhealthy. Alert on any sustained increase.
KB_EPOCH_FAILURES = Counter(
    "kb_epoch_failures_total",
    "KB epoch Redis operations that failed, by op (read/bump)",
    labelnames=["op"],
    registry=REGISTRY,
)
