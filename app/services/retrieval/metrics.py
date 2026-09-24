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
