"""Handoff queue-wait SLA metrics (process-level, Prometheus text exposure).

GB/T 47746—2026 requires 转人工时效 to be observable; HollyCRM's industry
benchmark targets <30s queue wait. These gauges surface both the worst
live wait and the breach count so ops can see a staffing gap while
customers are still holding.
"""

from prometheus_client import REGISTRY, Gauge

# Age of the oldest still-waiting ticket, recomputed on every queue_stats
# poll. None → gauge set to 0 (empty queue, nobody waiting).
HANDOFF_QUEUE_OLDEST_WAIT_SECONDS = Gauge(
    "handoff_queue_oldest_wait_seconds",
    "Age in seconds of the oldest open handoff ticket (worst live queue wait)",
    registry=REGISTRY,
)

# Open tickets that have waited past HANDOFF_SLA_WAIT_SECONDS.
HANDOFF_QUEUE_SLA_BREACHES = Gauge(
    "handoff_queue_sla_breaches",
    "Open handoff tickets currently waiting past the queue-wait SLA threshold",
    registry=REGISTRY,
)
