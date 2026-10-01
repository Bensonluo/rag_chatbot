"""The coverage gate must measure the maintained core (review 2026-09-26, #10).

graph.py / nodes.py / tools.py ARE the main dialogue path. Excluding
them from coverage made the reported "core coverage" un-interpretable
as system coverage — exactly the review's point: the README number
could not be read as "the system's coverage". Un-hiding them showed
the suite exercises them thoroughly (100% / 93.6% / 96.8% when first
measured, 2026-10-01), so the exclusion was hiding healthy numbers,
not protecting a gate.

This pins that the maintained core stays measured: re-adding these
omit lines must fail loudly here instead of silently shrinking what
the quality gate sees. Experimental/disabled-by-default paths
(GraphRAG internals) stay omitted by design — they are not part of
the maintained core quality gate.
"""

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

# The maintained main dialogue path — must stay inside the coverage gate.
MEASURED_CORE = {
    "app/services/dialogue/graph.py",
    "app/services/dialogue/nodes.py",
    "app/services/dialogue/tools.py",
}

# Same category: maintained, default-on service wiring. A factory the
# product always builds is not an experimental path.
MEASURED_WIRING = {"app/services/slot_filling/factory.py"}


def _coverage_omit() -> list[str]:
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return list(config["tool"]["coverage"]["run"]["omit"])


@pytest.mark.parametrize("module", sorted(MEASURED_CORE | MEASURED_WIRING))
def test_maintained_core_is_measured_by_coverage(module: str) -> None:
    omit = _coverage_omit()
    assert module not in omit, (
        f"{module} is maintained core but excluded from coverage — the "
        "gate number stops describing the system (review #10)"
    )


def test_coverage_gate_covers_app_tree() -> None:
    """Sanity: the omit list may only exclude tests, migrations, caches,
    vendored code, and explicitly-experimental paths — never anything
    under the maintained dialogue/slot-filling services."""
    omit = _coverage_omit()
    core_prefixes = (
        "app/services/dialogue/",
        "app/services/slot_filling/",
    )
    offenders = [
        pattern for pattern in omit for prefix in core_prefixes if pattern.startswith(prefix)
    ]
    assert not offenders, f"maintained core paths excluded from coverage: {offenders}"
