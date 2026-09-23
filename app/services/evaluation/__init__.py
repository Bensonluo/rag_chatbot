"""Offline evaluation: golden-set quality gate + containment/CSAT KPIs."""

from app.services.evaluation.quality_metrics import (
    QualityMetricsService,
    containment_rate,
    csat_score,
)
from app.services.evaluation.runner import (
    EvalReport,
    GoldenCase,
    load_golden_cases,
    route_bucket,
    run_intent_eval,
)

__all__ = [
    "QualityMetricsService",
    "containment_rate",
    "csat_score",
    "EvalReport",
    "GoldenCase",
    "load_golden_cases",
    "route_bucket",
    "run_intent_eval",
]
