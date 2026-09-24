"""Host-nginx upstream template: the 2→N story's last repo-side piece.

docker-compose.yml binds the api service to a 127.0.0.1:8000-8003 host
port RANGE so replicas boot without conflicts, and its comment says to
scale by widening "the range (and the nginx upstream)". The upstream
half of that instruction only exists if the repo carries the config:
deploy/nginx/cs-upstream.conf is the drop-in the host nginx includes
when the second replica goes live. These tests pin the lockstep —
if one side widens and the other doesn't, the gate fails here instead
of a replica silently receiving no traffic.
"""

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONF_PATH = _REPO_ROOT / "deploy" / "nginx" / "cs-upstream.conf"
_COMPOSE_PATH = _REPO_ROOT / "docker-compose.yml"


def _upstream_server_ports() -> list[int]:
    """Ports in the upstream's server lines (order-insensitive set)."""
    text = _CONF_PATH.read_text(encoding="utf-8")
    return [int(p) for p in re.findall(r"^\s*server\s+127\.0\.0\.1:(\d+)", text, re.M)]


def _compose_host_port_range() -> list[int]:
    """Host ports the api service may bind, expanded from the range."""
    text = _COMPOSE_PATH.read_text(encoding="utf-8")
    match = re.search(r'"127\.0\.0\.1:(\d+)-(\d+):\d+"', text)
    assert match, "compose api service must publish a 127.0.0.1 host port range"
    start, end = int(match.group(1)), int(match.group(2))
    return list(range(start, end + 1))


class TestNginxUpstreamContract:
    def test_upstream_ports_match_compose_range_exactly(self):
        """Lockstep: every compose-bindable host port is an upstream
        server, and no extra server points outside the range (a stale
        wider upstream would route to a dead port)."""
        assert _CONF_PATH.exists(), (
            "deploy/nginx/cs-upstream.conf is missing — the compose port "
            "range has no nginx counterpart to widen in lockstep"
        )
        assert sorted(_upstream_server_ports()) == _compose_host_port_range()

    def test_upstream_ejects_dead_replicas(self):
        """A crashed replica must leave rotation by itself: passive
        health markers (max_fails/fail_timeout) on every server, and
        least_conn so N-1 surviving replicas share the load."""
        text = _CONF_PATH.read_text(encoding="utf-8")
        assert "least_conn;" in text, "upstream needs least_conn fan-out"
        server_lines = [line for line in text.splitlines() if re.match(r"\s*server\s", line)]
        assert server_lines, "no upstream servers found"
        for line in server_lines:
            assert "max_fails=" in line, f"server line lacks max_fails: {line.strip()}"
            assert "fail_timeout=" in line, f"server line lacks fail_timeout: {line.strip()}"

    def test_template_documents_streaming_and_vhost_wiring(self):
        """The snippet's guidance is part of the contract: the one-line
        vhost change (proxy_pass to the named upstream) and the SSE
        directives a chat-streaming proxy must carry (buffering off or
        tokens arrive in visible bursts, plus a read timeout that
        survives slow LLM turns)."""
        text = _CONF_PATH.read_text(encoding="utf-8")
        assert "proxy_pass http://cs_api;" in text
        assert "proxy_buffering off;" in text
        assert re.search(r"proxy_read_timeout\s+\d+s;", text)
