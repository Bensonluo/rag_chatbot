"""Prometheus alert rules contract (anti-rot for the ops alerting layer).

Metrics without alerts are dashboards nobody watches at 3am: #73-#76
added GB/T 47746 时效 + TTFT observability, and this contract keeps the
alerting layer honest. The classic production failure is alert rot —
rules referencing metric names that were renamed or dropped, silently
never firing. These tests pin: the rules file is structurally valid,
every expression references a metric the app actually exports, the
prometheus config loads the rules, and docker-compose mounts them.
"""

import re
from pathlib import Path
from typing import Any

import yaml

# Importing every metrics module registers its collectors, so the
# generate_latest inventory below is the app's true export surface.
import app.middleware.metrics  # noqa: F401
import app.services.agent.metrics  # noqa: F401
import app.services.chat.metrics  # noqa: F401
import app.services.dialogue.funnel_metrics  # noqa: F401
import app.services.embeddings.metrics  # noqa: F401
import app.services.facts.metrics  # noqa: F401
import app.services.handoff.metrics  # noqa: F401
import app.services.handoff.one_shot_metrics  # noqa: F401
import app.services.retrieval.metrics  # noqa: F401

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ALERTS_PATH = _REPO_ROOT / "deploy" / "prometheus" / "alerts.yml"
_PROM_CONFIG_PATH = _REPO_ROOT / "deploy" / "prometheus" / "prometheus.yml"
_COMPOSE_PATH = _REPO_ROOT / "docker-compose.yml"

_PINNED_ALERTS = {
    "HandoffQueueSLABreach",
    "HandoffQueueOldestWaitCritical",
    "HandoffShareHigh",
    "AgentFallbackShareHigh",
    "KBEpochFailuresRising",
    "EmbeddingCacheFailuresRising",
    "FeedbackNegativeShareHigh",
    "OneShotRateLow",
    "ChatTTFTSlow",
    "ChatStreamErrorRateHigh",
    "ClaimGateViolationsRising",
}

# DB-bridged replica-invariant gauges (handoff/metrics.py +
# handoff/one_shot_metrics.py): every replica's refresher sets its own
# copy from the same DB aggregate, and dns_sd scrapes them all
# (test_scale_readiness) — a raw reference evaluates once per replica
# (duplicate alert instances) instead of once for the truth.
_DB_BRIDGED_GAUGES = (
    "chat_one_shot_rate",
    "chat_sessions_served",
    "chat_sessions_escalated",
    "handoff_queue_sla_breaches",
    "handoff_queue_oldest_wait_seconds",
    "handoff_pickup_avg_seconds",
    "handoff_handle_avg_seconds",
)


def _rules() -> list[dict[str, Any]]:
    doc = yaml.safe_load(_ALERTS_PATH.read_text())
    assert isinstance(doc, dict) and doc.get("groups"), "alerts.yml must define groups"
    return [rule for group in doc["groups"] for rule in group.get("rules", [])]


def _exported_metric_names() -> set[str]:
    from prometheus_client import generate_latest

    names: set[str] = set()
    for line in generate_latest().decode().splitlines():
        if line.startswith("# HELP "):
            names.add(line.split()[2])
    return names


def _referenced_metrics(expr: str, known: set[str]) -> set[str]:
    """Metric names appearing in the expression (allowing _bucket/_sum/
    _count/_total sample suffixes, rejecting longer lookalikes)."""
    found = set()
    for name in known:
        pattern = (
            rf"(?<![A-Za-z0-9_:]){re.escape(name)}(?:_(?:bucket|sum|count|total))?(?![A-Za-z0-9_])"
        )
        if re.search(pattern, expr):
            found.add(name)
    return found


class TestAlertRulesContract:
    def test_db_bridged_gauges_are_replica_aggregated(self):
        """A DB-bridged gauge holds replica-invariant truth, but each
        scraped replica contributes its own series — reference it
        through avg() so an alert evaluates the truth once, not N
        times (and a single fail-open frozen replica cannot alone page
        or alone silence the alert)."""
        for rule in _rules():
            expr = str(rule.get("expr", ""))
            for gauge in _DB_BRIDGED_GAUGES:
                if gauge in expr:
                    assert f"avg({gauge})" in expr, (
                        f"alert {rule.get('alert')} references DB-bridged gauge "
                        f"{gauge} without avg(): under dns_sd it evaluates one "
                        "series per replica"
                    )

    def test_rules_file_has_valid_structure(self):
        for rule in _rules():
            assert rule.get("alert"), "every rule needs an alert name"
            assert rule.get("expr"), f"{rule.get('alert')}: expr missing"
            assert rule.get("for"), f"{rule['alert']}: 'for' duration missing (flap guard)"
            labels = rule.get("labels") or {}
            assert labels.get("severity") in {"critical", "warning"}, (
                f"{rule['alert']}: severity must be critical or warning"
            )
            annotations = rule.get("annotations") or {}
            assert annotations.get("summary"), f"{rule['alert']}: summary annotation missing"

    def test_every_rule_references_a_registered_metric(self):
        """Alert rot guard: an expression naming no exported metric can
        never fire — catch it in CI, not in the incident retro."""
        known = _exported_metric_names()
        assert known, "metric inventory came back empty — imports broke"
        for rule in _rules():
            referenced = _referenced_metrics(str(rule["expr"]), known)
            assert referenced, f"{rule['alert']}: expr references no exported metric"
            print(f"{rule['alert']}: {sorted(referenced)}")

    def test_pinned_alerts_exist(self):
        names = {rule["alert"] for rule in _rules()}
        missing = _PINNED_ALERTS - names
        assert not missing, f"pinned alerts missing (renamed?): {missing}"

    def test_prometheus_config_loads_the_rules_file(self):
        doc = yaml.safe_load(_PROM_CONFIG_PATH.read_text())
        rule_files = doc.get("rule_files") or []
        assert rule_files, "prometheus.yml must reference alert rule files"
        for entry in rule_files:
            # Compose mounts the whole deploy/prometheus dir context; the
            # referenced file must exist next to prometheus.yml.
            target = _PROM_CONFIG_PATH.parent / Path(entry).name
            assert target.exists(), f"rule_files entry {entry} has no file"

    def test_docker_compose_mounts_alert_rules(self):
        doc = yaml.safe_load(_COMPOSE_PATH.read_text())
        volumes = doc["services"]["prometheus"]["volumes"]
        assert any("alerts.yml" in v for v in volumes), (
            "prometheus service must mount deploy/prometheus/alerts.yml"
        )
