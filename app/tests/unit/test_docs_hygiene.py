"""Docs hygiene pins: README requirements stay anchored to the stack files.

The Quick Start's Requirements block documents the default Compose stack's
loopback ports and the Python floor. Both facts live primarily in
``docker-compose.yml`` and ``pyproject.toml``; without a pin either side
drifts silently — a new service port ships undocumented, or a README port
survives after its mapping is gone. These tests fail the suite on drift.
"""

import re
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE = REPO_ROOT / "docker-compose.yml"
README = REPO_ROOT / "README.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _expand_port_token(token: str) -> set[int]:
    """Expand ``8000-8003`` ranges and ``6333/6334`` lists to plain ports."""
    ports: set[int] = set()
    for part in re.split(r"[/,]", token):
        span = re.fullmatch(r"(\d+)[–-](\d+)", part)
        if span:
            ports.update(range(int(span.group(1)), int(span.group(2)) + 1))
        elif part.isdigit():
            ports.add(int(part))
    return ports


def _compose_host_ports() -> set[int]:
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    ports: set[int] = set()
    for service in services.values():
        for entry in service.get("ports", []):
            # "127.0.0.1:8000-8003:8000" — host segment is the middle one;
            # short forms ("8000-8003:8000") put it first.
            segments = str(entry).split(":")
            host = segments[-2] if len(segments) >= 3 else segments[0]
            ports.update(_expand_port_token(host))
    return ports


def _readme_requirements_section() -> str:
    text = README.read_text()
    match = re.search(r"^### Requirements$(.*?)(?=^###? )", text, flags=re.MULTILINE | re.DOTALL)
    assert match, "README has no '### Requirements' section"
    return str(match.group(1))


def _readme_documented_ports() -> set[int]:
    # Only backticked tokens count as port claims; prose digits ("Compose
    # v2", "127.0.0.1") are not documentation of the port map.
    ports: set[int] = set()
    for token in re.findall(r"`([^`]+)`", _readme_requirements_section()):
        ports.update(_expand_port_token(token))
    return ports


def test_readme_port_map_matches_compose_exactly() -> None:
    compose_ports = _compose_host_ports()
    assert compose_ports, "docker-compose.yml publishes no host ports — parser broke"
    documented = _readme_documented_ports()
    drift = documented ^ compose_ports
    assert not drift, (
        f"README Requirements ports drifted from docker-compose.yml: "
        f"symmetric difference {sorted(drift)} — update both sides together"
    )


def test_readme_python_floor_matches_pyproject() -> None:
    with PYPROJECT.open("rb") as handle:
        requires = str(tomllib.load(handle)["project"]["requires-python"])
    floor = requires.removeprefix(">=")
    section = _readme_requirements_section()
    assert f"≥ {floor}" in section, (
        f"README Requirements does not state 'Python ≥ {floor}' "
        f"(pyproject requires-python is {requires})"
    )
