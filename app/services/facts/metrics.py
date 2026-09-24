"""Claim-gate funnel metrics (process-level, Prometheus text exposure)."""

from prometheus_client import REGISTRY, Counter

CLAIM_CHECKS = Counter(
    "claim_checks_total",
    "Responses whose policy claims were checked against the curated fact table",
    registry=REGISTRY,
)

CLAIM_VIOLATIONS = Counter(
    "claim_violations_total",
    "Policy-claim violations caught by the claim gate",
    labelnames=["reason"],
    registry=REGISTRY,
)
