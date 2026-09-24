"""Deterministic policy facts and claim verification (Phase A2).

The LLM proposes; the fact table is the authority; the claim gate
verifies every generated response before it leaves the graph.
"""

from app.services.facts.claim_check import (
    ClaimCheckResult,
    ClaimViolation,
    apply_violations,
    check_policy_claims,
)
from app.services.facts.fact_store import FactStore, PolicyFact

__all__ = [
    "ClaimCheckResult",
    "ClaimViolation",
    "FactStore",
    "PolicyFact",
    "apply_violations",
    "check_policy_claims",
]
