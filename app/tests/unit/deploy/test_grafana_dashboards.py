"""Grafana dashboard provisioning contract (anti-rot, dashboard edition).

The funnel trilogy is metric → alert → dashboard: ``chat_funnel_layers_total``
and the HandoffShareHigh alert exist, but an empty dashboards directory means
Grafana ships with nothing to look at. These tests pin the same honesty the
alert contract enforces: dashboards parse, panels reference metrics the app
actually exports (renamed metrics silently blank a panel), the provider
config is valid, and docker-compose actually mounts both layers.
"""

import json
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
import app.services.handoff.containment_metrics  # noqa: F401
import app.services.handoff.metrics  # noqa: F401
import app.services.retrieval.metrics  # noqa: F401

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DASHBOARDS_DIR = _REPO_ROOT / "deploy" / "grafana" / "dashboards"
_PROVIDER_DIR = _REPO_ROOT / "deploy" / "grafana" / "provisioning" / "dashboards"
_COMPOSE_PATH = _REPO_ROOT / "docker-compose.yml"

_PINNED_DASHBOARD_TITLE = "CS 漏斗北极星"
_PINNED_PANEL_TITLES = {
    "漏斗分层流量（turns/s）",
    "Handoff 占比（自动处理率反量）",
    "塔顶吸收率（缓存+FAQ）",
    "Agent 终局分布（runs/s）",
    "Agent 回退占比（深层降级）",
    "缓存金字塔命中率",
    "用户差评率（CSAT 反量）",
    "知识缺口占比",
}
_FUNNEL_METRIC = "chat_funnel_layers_total"
_INITIALIZED_REPLICA_FILTER = "(chat_containment_stats_last_success_timestamp_seconds > 0)"
_CONTAINMENT_RATE_QUERY = (
    f"avg(chat_containment_rate and (chat_sessions_served > 0) and {_INITIALIZED_REPLICA_FILTER})"
)

# DB-bridged gauges (handoff/containment_metrics.py, handoff/metrics.py):
# each replica's independent refresher sets its own observation and
# dns_sd scrapes them all — a raw expr renders one series per replica.
_DB_BRIDGED_GAUGES = (
    "chat_containment_rate",
    "chat_sessions_served",
    "chat_sessions_escalated",
    "chat_sessions_with_failed_replies",
    "chat_sessions_with_completed_replies",
    "chat_sessions_latest_reply_failed",
    "handoff_pickup_avg_seconds",
    "handoff_handle_avg_seconds",
)


def _dashboards() -> list[dict[str, Any]]:
    files = sorted(_DASHBOARDS_DIR.glob("*.json"))
    assert files, "deploy/grafana/dashboards must contain at least one dashboard JSON"
    docs = []
    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(doc, dict), f"{path.name}: dashboard must be a JSON object"
        docs.append(doc)
    return docs


def _exported_metric_names() -> set[str]:
    from prometheus_client import generate_latest

    names: set[str] = set()
    for line in generate_latest().decode().splitlines():
        if line.startswith("# HELP "):
            names.add(line.split()[2])
    return names


def _referenced_metrics(expr: str, known: set[str]) -> set[str]:
    found = set()
    for name in known:
        pattern = (
            rf"(?<![A-Za-z0-9_:]){re.escape(name)}(?:_(?:bucket|sum|count|total))?(?![A-Za-z0-9_])"
        )
        if re.search(pattern, expr):
            found.add(name)
    return found


def _panels(doc: dict[str, Any]) -> list[dict[str, Any]]:
    panels = list(doc.get("panels") or [])
    for row in panels:
        panels.extend(row.get("panels") or [])
    return panels


class TestGrafanaDashboardContract:
    def test_dashboards_are_structurally_valid(self):
        for doc in _dashboards():
            assert doc.get("title"), "dashboard needs a title"
            assert doc.get("uid"), "dashboard needs a uid (stable deep-link)"
            assert doc.get("schemaVersion"), "dashboard needs a schemaVersion"
            assert _panels(doc), f"{doc['title']}: dashboard has no panels"
            for panel in _panels(doc):
                assert panel.get("title"), "every panel needs a title"
                targets = panel.get("targets") or []
                assert targets, f"{panel['title']}: panel has no query targets"
                for target in targets:
                    assert target.get("expr"), f"{panel['title']}: target missing expr"

    def test_pinned_funnel_dashboard_exists(self):
        titles = {doc["title"] for doc in _dashboards()}
        assert _PINNED_DASHBOARD_TITLE in titles, (
            f"pinned dashboard missing (renamed?): {_PINNED_DASHBOARD_TITLE}"
        )
        doc = next(d for d in _dashboards() if d["title"] == _PINNED_DASHBOARD_TITLE)
        panel_titles = {p["title"] for p in _panels(doc)}
        missing = _PINNED_PANEL_TITLES - panel_titles
        assert not missing, f"pinned funnel panels missing (renamed?): {missing}"

    def test_every_panel_references_a_registered_metric(self):
        """Panel rot guard: an expression naming no exported metric renders
        an eternally-empty chart — catch it in CI, not in the demo."""
        known = _exported_metric_names()
        assert known, "metric inventory came back empty — imports broke"
        for doc in _dashboards():
            for panel in _panels(doc):
                for target in panel.get("targets") or []:
                    referenced = _referenced_metrics(str(target["expr"]), known)
                    assert referenced, (
                        f"{doc['title']}/{panel['title']}: expr references no exported metric"
                    )

    def test_funnel_dashboard_uses_the_funnel_counter(self):
        """The north-star dashboard must actually show the funnel: at least
        one panel's expressions mention the layer counter."""
        doc = next(d for d in _dashboards() if d["title"] == _PINNED_DASHBOARD_TITLE)
        exprs = [
            str(target["expr"]) for panel in _panels(doc) for target in panel.get("targets") or []
        ]
        assert any(_FUNNEL_METRIC in expr for expr in exprs), (
            "funnel dashboard has no expression using chat_funnel_layers_total"
        )

    def test_db_bridged_gauges_are_replica_aggregated(self):
        """Preserve avg() for independently refreshed DB observations.

        This avoids adding one database's count for every scraped
        replica. It does not make stale and fresh replicas a snapshot.
        """
        for doc in _dashboards():
            exprs = [
                str(target["expr"])
                for panel in _panels(doc)
                for target in panel.get("targets") or []
            ]
            for expr in exprs:
                for gauge in _DB_BRIDGED_GAUGES:
                    if gauge in expr:
                        if "chat_containment_rate" in expr:
                            assert expr == _CONTAINMENT_RATE_QUERY, (
                                "undefined or uninitialized rates must be filtered before averaging"
                            )
                            continue
                        if gauge.startswith("chat_sessions_"):
                            assert expr == f"avg({gauge} and {_INITIALIZED_REPLICA_FILTER})", (
                                "uninitialized replica counts must be excluded before averaging"
                            )
                            continue
                        assert f"avg({gauge})" in expr, (
                            f"panel expr {expr!r} references DB-bridged gauge "
                            f"{gauge} without avg(): under dns_sd it renders one "
                            "series per replica"
                        )

    def test_funnel_dashboard_shows_the_real_containment_metric(self):
        """The session-level containment rate (chat_containment_rate, DB-bridged
        Gauge) is the north star's honest metric — the dashboard must
        carry it next to the turn-level handoff-share proxy panel."""
        doc = next(d for d in _dashboards() if d["title"] == _PINNED_DASHBOARD_TITLE)
        exprs = [
            str(target["expr"]) for panel in _panels(doc) for target in panel.get("targets") or []
        ]
        assert any("chat_containment_rate" in expr for expr in exprs), (
            "funnel dashboard has no expression using chat_containment_rate"
        )

    def test_containment_rate_and_absolute_counts_have_separate_units(self):
        doc = next(d for d in _dashboards() if d["uid"] == "cs-funnel")
        rate = next(panel for panel in _panels(doc) if panel["id"] == 11)
        assert rate["fieldConfig"]["defaults"]["unit"] == "percentunit"
        assert len(rate["targets"]) == 1
        assert rate["targets"][0]["expr"] == _CONTAINMENT_RATE_QUERY
        assert "{{window}}" not in rate["targets"][0]["legendFormat"]
        assert "差评" in rate["description"] and "解决" in rate["description"]
        assert "CONTAINMENT_STATS_WINDOW_DAYS" in rate["description"]
        for metric in _DB_BRIDGED_GAUGES[1:6]:
            matching = [
                panel
                for panel in _panels(doc)
                if any(metric in target["expr"] for target in panel.get("targets", []))
                and "chat_containment_rate"
                not in " ".join(target["expr"] for target in panel.get("targets", []))
            ]
            assert matching, f"missing count panel for {metric}"
            assert all(panel["fieldConfig"]["defaults"]["unit"] == "short" for panel in matching)

    def test_freshness_reports_the_most_recent_successful_replica(self):
        doc = next(d for d in _dashboards() if d["uid"] == "cs-funnel")
        panel = next(panel for panel in _panels(doc) if panel["id"] == 16)
        assert panel["fieldConfig"]["defaults"]["unit"] == "s"
        assert panel["targets"][0]["expr"] == (
            "time() - max(chat_containment_stats_last_success_timestamp_seconds > 0)"
        )
        assert "最近成功" in panel["description"]
        assert "全部副本" in panel["description"]

    def test_pyramid_panel_covers_every_cache_layer(self):
        """The 缓存金字塔命中率 panel is the load-bearing cache view; each
        measured layer's series must survive future panel edits. L3 counts
        per text (batch API), the others per call."""
        doc = next(d for d in _dashboards() if d["title"] == _PINNED_DASHBOARD_TITLE)
        panel = next(p for p in _panels(doc) if p["title"] == "缓存金字塔命中率")
        exprs = [str(t["expr"]) for t in panel.get("targets") or []]
        for layer_metric in (
            "answer_cache_hits_total",  # L0 exact
            "retrieval_cache_hits_total",  # L2 retrieval
            "embedding_cache_hits_total",  # L3 embedding
        ):
            assert any(layer_metric in expr for expr in exprs), (
                f"pyramid panel lost its {layer_metric} series"
            )

    def test_provider_config_is_valid(self):
        files = sorted(_PROVIDER_DIR.glob("*.yml"))
        assert files, "deploy/grafana/provisioning/dashboards must define a provider"
        for path in files:
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert isinstance(doc, dict) and doc.get("apiVersion") == 1
            providers = doc.get("providers") or []
            assert providers, f"{path.name}: no providers listed"
            for provider in providers:
                assert provider.get("type") == "file"
                assert provider.get("options", {}).get("path"), (
                    "provider needs options.path (where compose mounts the JSONs)"
                )

    def test_docker_compose_mounts_dashboards_and_provider(self):
        doc = yaml.safe_load(_COMPOSE_PATH.read_text(encoding="utf-8"))
        volumes = doc["services"]["grafana"]["volumes"]
        joined = " ".join(str(v) for v in volumes)
        assert "deploy/grafana/dashboards" in joined, (
            "grafana must mount the dashboards JSON directory"
        )
        assert "provisioning/dashboards" in joined, (
            "grafana must mount the dashboard provider config"
        )
