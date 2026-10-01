#!/usr/bin/env python3
"""Regenerate requirements-lock.txt (runtime dependency lock).

Resolves the BASE dependencies of pyproject.toml under the production
interpreter (python:3.11-slim — the same image Docker and CI use) with
``pip install --dry-run --report``: a fresh, environment-independent
resolve that never touches the local venv. The report's resolved set is
written to requirements-lock.txt as exact pins, consumed downstream as
pip constraints (see the lock file header).

Why a container: a lock resolved under the wrong interpreter can pin a
version whose requires-python excludes 3.11, and a constraints file
containing such a pin breaks every 3.11 install outright.

Usage (needs Docker; the image is pulled if absent):

    python scripts/generate_lock.py

Drift between the lock and pyproject is pinned by
app/tests/unit/test_dependency_lock.py — run the test after
regenerating.
"""

import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = REPO_ROOT / "requirements-lock.txt"

# Same mirror the Dockerfile uses — keeps the resolve fast on CN
# networks and consistent with what production builds fetch.
PYPI_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple/"

HEADER = """\
# Runtime dependency lock — fresh python-3.11 resolve of the BASE
# dependencies in pyproject.toml (the same interpreter the Docker image
# and CI use). Consumed as pip constraints so a differently-shaped
# platform resolve can never be broken by a missing entry:
#
#   docker build / CI / server:  pip install . -c requirements-lock.txt
#   CI dependency audit:         pip-audit -r requirements-lock.txt
#
# Regenerate whenever pyproject [project.dependencies] changes:
#
#   python scripts/generate_lock.py
#
# Drift is pinned by app/tests/unit/test_dependency_lock.py (every base
# dep locked, pins satisfy pyproject specifiers, no dev tooling leaks).
# Generated {generated} against tsinghua PyPI mirror.
"""


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "report.json"
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{REPO_ROOT}:/build",
            "-w",
            "/build",
            "-v",
            f"{tmp}:/out",
            "python:3.11-slim",
            "pip",
            "install",
            "--dry-run",
            "--quiet",
            "--ignore-installed",
            f"--report=/out/{report_path.name}",
            "-i",
            PYPI_INDEX,
            ".",
        ]
        print("resolving base dependencies under python:3.11-slim ...", flush=True)
        subprocess.run(cmd, check=True)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = sorted(
            (item["metadata"]["name"], item["metadata"]["version"])
            for item in report["install"]
            if item["metadata"]["name"] != "rag-chatbot"  # the project itself is not a dependency
        )

    if not rows:
        print("empty resolve — refusing to write a lock", file=sys.stderr)
        return 1

    body = "\n".join(f"{name}=={version}" for name, version in rows)
    LOCK_FILE.write_text(
        HEADER.format(generated=date.today().isoformat()) + body + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} pinned packages to {LOCK_FILE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
