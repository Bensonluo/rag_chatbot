"""Scale-readiness contract: ``--scale api=N`` must actually work.

The node-template doctrine (single box = one replica template, 2→N
without re-architecture) makes two compose-level promises that are
easy to silently break:

1. **Bootability** — a fixed host port (``127.0.0.1:8000:8000``)
   makes the second replica fail with "port is already allocated".
   The api service must publish a port *range* so Docker allocates
   one host port per replica.
2. **Observability** — ``static_configs: ["api:8000"]`` resolves the
   service name via Docker's embedded DNS, which round-robins to ONE
   container per lookup: at N replicas Prometheus scrapes a random
   replica each interval, seeing ~1/N of the traffic and interleaving
   per-process counters into garbage. The canonical non-Swarm fix is
   DNS service discovery (``dns_sd_configs type: A``), which asks the
   embedded DNS for ALL container IPs behind the service name.
"""

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE_PATH = _REPO_ROOT / "docker-compose.yml"
_PROM_CONFIG_PATH = _REPO_ROOT / "deploy" / "prometheus" / "prometheus.yml"


def _compose() -> dict[str, Any]:
    return yaml.safe_load(_COMPOSE_PATH.read_text())


def _prom_config() -> dict[str, Any]:
    return yaml.safe_load(_PROM_CONFIG_PATH.read_text())


def _api_ports() -> list[str]:
    ports = _compose()["services"]["api"].get("ports") or []
    return [str(p) for p in ports]


def _api_scrape_job() -> dict[str, Any]:
    jobs = _prom_config().get("scrape_configs") or []
    api_jobs = [j for j in jobs if "api" in str(j.get("job_name", ""))]
    assert api_jobs, "prometheus.yml must define a scrape job for the api service"
    return api_jobs[0]


class TestScaleReadiness:
    def test_api_publishes_a_port_range_not_a_fixed_port(self):
        """A fixed host port allocates on the first replica only; the
        second ``--scale`` replica dies on port conflict. A published
        range lets Docker hand each replica its own host port."""
        ports = _api_ports()
        assert ports, "api service must publish ports (host nginx proxies it)"
        for published in ports:
            # "127.0.0.1:8000-8003:8000" (or "8000-8003:8000"): the host
            # port is the second-to-last colon-separated segment.
            host_side = published.split(":")[-2]
            assert "-" in host_side, (
                f"api publishes fixed host port {published!r}: the second "
                "replica cannot boot. Publish a range (e.g. 127.0.0.1:8000-8003:8000)"
            )

    def test_api_scrape_uses_dns_service_discovery(self):
        """static_configs resolves one round-robin IP per lookup — at N
        replicas Prometheus would see a random 1/N subset. dns_sd_configs
        with type A enumerates every container behind the service name."""
        job = _api_scrape_job()
        assert job.get("dns_sd_configs"), (
            "api scrape job must use dns_sd_configs (static single-target "
            "scrapes one random replica per interval under --scale)"
        )
        assert not job.get("static_configs"), (
            "api scrape job must not mix static_configs with dns_sd_configs"
        )

    def test_dns_sd_targets_the_api_service_on_its_port(self):
        job = _api_scrape_job()
        dns = job["dns_sd_configs"][0]
        assert "api" in (dns.get("names") or []), "dns_sd must resolve the api service name"
        assert dns.get("type") == "A", "Docker embedded DNS serves A records (not SRV)"
        assert dns.get("port") == 8000, "dns_sd must scrape the api container port"
        refresh = dns.get("refresh_interval")
        assert refresh, "dns_sd needs a refresh_interval so scaled-up replicas are discovered"
