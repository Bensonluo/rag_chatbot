"""Redis memory guard contract (anti-rot for the single-box template).

The capacity audit's first armed memory bomb: Redis booted with no
maxmemory ceiling on a 4GB box shared with postgres/qdrant/api —
growth goes to RAM until the kernel OOM killer picks a victim by RSS
(redis is never the biggest, so the API dies first). Every key in
this Redis is TTL-bearing cache or sliding-window state (verified:
answer_cache ``set(ex=)``, embedding cache ``setex``, rate-limiter
zsets re-``EXPIRE`` per write), so allkeys-lru eviction degrades
gracefully. These tests pin the guard so a future compose edit
cannot silently drop it.
"""

import re
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE_PATH = _REPO_ROOT / "docker-compose.yml"


def _redis_service() -> dict[str, Any]:
    compose = yaml.safe_load(_COMPOSE_PATH.read_text())
    return compose["services"]["redis"]


def _to_mb(size: str) -> float:
    """Parse compose/redis size strings ("512m", "256mb") to MB."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kKmMgG])[bB]?", size.strip())
    if not match:
        raise AssertionError(f"Unparseable size string: {size!r}")
    value, unit = match.groups()
    return float(value) * {"k": 1 / 1024, "m": 1, "g": 1024}[unit.lower()]


class TestRedisMemoryGuard:
    def test_maxmemory_capped_with_lru_eviction(self):
        command = _redis_service()["command"]
        assert "--maxmemory" in command, "redis must boot with an explicit ceiling"
        assert "--maxmemory-policy allkeys-lru" in command

    def test_container_limit_leaves_headroom_over_maxmemory(self):
        service = _redis_service()
        limit = service["deploy"]["resources"]["limits"]["memory"]
        command = service["command"]
        maxmemory = re.search(r"--maxmemory\s+(\S+)", command)
        assert maxmemory is not None
        # The container cap must exceed maxmemory: AOF rewrite buffers
        # and allocator fragmentation live in the gap. A 1:1 copy would
        # flip the OOM target back to the container itself.
        assert _to_mb(limit) >= 2 * _to_mb(maxmemory.group(1))
