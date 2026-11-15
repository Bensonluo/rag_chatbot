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


def _readme_local_image_refs() -> set[str]:
    """Local image paths README references, markdown and HTML ``<img>`` forms."""
    text = README.read_text()
    refs = re.findall(r"<img[^>]+src=\"([^\"]+)\"", text)
    refs += re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", text)
    return {r for r in refs if not r.startswith(("http://", "https://"))}


def test_readme_local_images_exist() -> None:
    refs = _readme_local_image_refs()
    assert refs, "README references no local image — the hero/demo shots dropped?"
    missing = [r for r in sorted(refs) if not (REPO_ROOT / r).is_file()]
    assert not missing, f"README references images that do not exist: {missing}"


def _zh_requirements_line() -> str:
    """The ZH quickstart condenses Requirements into one bolded line."""
    text = README.read_text()
    match = re.search(r"^\*\*环境要求\*\*：(.*)$", text, flags=re.MULTILINE)
    assert match, "README has no '**环境要求**' line for the ZH quickstart"
    return str(match.group(1))


def test_zh_quickstart_port_map_matches_compose_exactly() -> None:
    compose_ports = _compose_host_ports()
    documented: set[int] = set()
    for token in re.findall(r"`([^`]+)`", _zh_requirements_line()):
        documented.update(_expand_port_token(token))
    drift = documented ^ compose_ports
    assert not drift, (
        f"ZH 环境要求 ports drifted from docker-compose.yml: "
        f"symmetric difference {sorted(drift)} — update both sides together"
    )


def test_zh_quickstart_python_floor_matches_pyproject() -> None:
    with PYPROJECT.open("rb") as handle:
        requires = str(tomllib.load(handle)["project"]["requires-python"])
    floor = requires.removeprefix(">=")
    line = _zh_requirements_line()
    assert f"Python ≥ {floor}" in line, (
        f"ZH 环境要求 does not state 'Python ≥ {floor}' (pyproject requires-python is {requires})"
    )


NGINX_CONFIGS = [
    REPO_ROOT / "frontend" / "nginx.conf",
    REPO_ROOT / "frontend" / "nginx.server.conf",
]

# Regex locations with an optional single-segment prefix: the server unit
# mounts the export under BASE_PATH=/store, so hashed assets arrive as
# /store/_next/static/... — a plain prefix location never matched them
# and the cache headers silently never fired (observed live 2026-11-15).
NGINX_HASHED_ASSET_PATTERN = r"^/(?:[^/]+/)?_next/static/"
NGINX_IMAGE_PATTERN = r"^/(?:[^/]+/)?images/"


def _nginx_location_block(path: Path, pattern: str) -> str:
    text = path.read_text()
    match = re.search(rf"location ~ {re.escape(pattern)} \{{(.*?)\}}", text, flags=re.DOTALL)
    assert match, f"{path.name} has no 'location ~ {pattern}' block"
    return str(match.group(1))


class TestNginxCachingPinned:
    """The frontend's nginx variants serve a Next.js static export; both
    must cache content-hashed build assets as immutable (the header Next.js
    itself sets on ``_next/static`` — with a pure nginx export the server
    must replicate it) and give un-hashed images a bounded TTL. The
    locations must be BASE_PATH-agnostic regexes, or a /store mount
    serves its assets with no cache header at all. Without a pin a config
    edit can silently drop the cache policy again."""

    def test_hashed_build_assets_cache_immutable(self) -> None:
        for conf in NGINX_CONFIGS:
            block = _nginx_location_block(conf, NGINX_HASHED_ASSET_PATTERN)
            assert "max-age=31536000" in block, conf.name
            assert "immutable" in block, conf.name

    def test_unhashed_images_cache_is_bounded(self) -> None:
        for conf in NGINX_CONFIGS:
            block = _nginx_location_block(conf, NGINX_IMAGE_PATTERN)
            assert "max-age=86400" in block, conf.name
            assert "immutable" not in block, conf.name
