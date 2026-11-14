"""Release hygiene pins: one version across pyproject, CHANGELOG and the bundled frontend.

The 2026-11-14 v0.4.0 release shipped with ``pyproject.toml`` still at 0.3.0
while the tag, the CHANGELOG section and the bundled frontend ``package.json``
all said 0.4.0 — ``/health`` and installed-distribution metadata would report
a version the repo had already moved past. The version single source is
``pyproject.toml`` (read at runtime via ``importlib.metadata``); these pins
fail the suite the moment any declared surface drifts from it.
"""

import json
import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
FRONTEND_PACKAGE = REPO_ROOT / "frontend" / "package.json"
CITATION = REPO_ROOT / "CITATION.cff"


def _pyproject_version() -> str:
    with PYPROJECT.open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def _latest_changelog_release() -> str:
    headings = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", CHANGELOG.read_text(), flags=re.MULTILINE)
    assert headings, "CHANGELOG.md has no versioned release headings"
    # Keep-a-Changelog orders releases newest-first; [Unreleased] cannot match.
    return str(headings[0])


def _frontend_version() -> str:
    return str(json.loads(FRONTEND_PACKAGE.read_text())["version"])


def _citation_version() -> str:
    match = re.search(r"^version: (\d+\.\d+\.\d+)$", CITATION.read_text(), flags=re.MULTILINE)
    assert match, "CITATION.cff has no version field"
    return str(match.group(1))


def test_pyproject_version_matches_latest_changelog_release() -> None:
    assert _pyproject_version() == _latest_changelog_release()


def test_pyproject_version_matches_bundled_frontend() -> None:
    assert _pyproject_version() == _frontend_version()


def test_pyproject_version_matches_citation_file() -> None:
    assert _pyproject_version() == _citation_version()
