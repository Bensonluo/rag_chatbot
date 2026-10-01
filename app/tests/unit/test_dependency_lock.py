"""Dependency lock contract (review 2026-09-26, #10 dependency-health half).

The repository resolved dependencies loosely (``>=`` floors) at every
install: CI, Docker builds, and the server could each pick different
versions on different days — no reproducibility and no stable input for
vulnerability auditing. The runtime lock (``requirements-lock.txt``,
generated from a fresh python-3.11 resolve of the base dependencies)
fixes that; CI audits it with pip-audit.

These pins keep the lock honest:

- fully pinned lines only (``name==version``) — a partial lock is not a
  lock;
- every base dependency declared in pyproject must appear in it —
  adding a dependency without regenerating the lock fails here;
- each pinned version must satisfy its pyproject specifier — a lock
  that contradicts the declared constraints (e.g. qdrant-client >=1.10,
  which needs a newer Qdrant server than the compose stack pins) is
  drift;
- dev tooling must stay out — the lock covers the runtime surface
  (what Docker installs), not someone's ``pip freeze`` of a dev venv.

Regenerate with the command in the lock file header.
"""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"
LOCK_FILE = REPO_ROOT / "requirements-lock.txt"

# The lock pins the RUNTIME surface only. Dev tooling (test runners,
# linters, load-test tools) is resolved fresh by CI's `.[dev]` install
# and must never leak in via a whole-venv freeze.
DEV_ONLY_SENTINELS = ("pytest", "ruff", "mypy", "locust", "pre-commit")


def _base_requirements() -> list[Requirement]:
    with PYPROJECT.open("rb") as fh:
        deps = tomllib.load(fh)["project"]["dependencies"]
    return [Requirement(dep) for dep in deps]


def _locked_versions() -> dict[str, str]:
    locked: dict[str, str] = {}
    for line in LOCK_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, version = line.partition("==")
        assert version, f"unpinned lock line: {line!r} (name==version required)"
        locked[canonicalize_name(name)] = version
    return locked


def test_lock_file_exists() -> None:
    assert LOCK_FILE.is_file(), (
        "requirements-lock.txt is missing — regenerate it (see the command "
        "in the header) whenever pyproject dependencies change"
    )


def test_lock_lines_are_fully_pinned() -> None:
    for name, version in _locked_versions().items():
        assert version and " " not in version and ";" not in version, (
            f"lock entry {name}=={version!r} must be an exact pin"
        )


def test_every_base_dependency_is_locked() -> None:
    locked = _locked_versions()
    missing = [
        str(req) for req in _base_requirements() if canonicalize_name(req.name) not in locked
    ]
    assert not missing, (
        f"base dependencies absent from requirements-lock.txt: {missing} — "
        "regenerate the lock (command in its header)"
    )


def test_locked_versions_satisfy_pyproject_specifiers() -> None:
    locked = _locked_versions()
    violations = []
    for req in _base_requirements():
        pinned = locked.get(canonicalize_name(req.name))
        if pinned is not None and not req.specifier.contains(pinned, prereleases=True):
            violations.append(f"{req.name}=={pinned} violates {req.specifier}")
    assert not violations, (
        f"lock contradicts pyproject constraints (drift between the two): {violations}"
    )


def test_dev_tooling_stays_out_of_the_runtime_lock() -> None:
    locked = _locked_versions()
    leaked = [sentinel for sentinel in DEV_ONLY_SENTINELS if canonicalize_name(sentinel) in locked]
    assert not leaked, (
        f"dev-only tools {leaked} leaked into the runtime lock — the lock "
        "covers what Docker/production installs, not a dev venv freeze"
    )
