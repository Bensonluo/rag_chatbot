"""Gate completeness: no test file may be invisible to the default run.

pyproject's ``testpaths`` is a hand-grown allowlist — every new test
file had to be registered by hand or it silently never ran. Ten files
accumulated that way (including three API contract suites from this
repo's own recent work), and two of them rotted invisibly for it. This
meta-contract fails the gate the moment a test file exists outside the
collected set: the failure mode becomes loud instead of silent.
"""

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _files_on_disk() -> set[str]:
    return {
        str(p.relative_to(_REPO_ROOT))
        for p in (_REPO_ROOT / "app" / "tests").rglob("test_*.py")
        if "__pycache__" not in p.parts
    }


def _collected_files() -> set[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    files = {
        line.split("::", 1)[0]
        for line in result.stdout.splitlines()
        if "::" in line and line.startswith("app")
    }
    assert files, f"pytest --collect-only produced no node ids:\n{result.stderr[-2000:]}"
    return files


class TestGateCollectsEveryTestFile:
    def test_no_test_file_is_invisible_to_the_gate(self):
        invisible = _files_on_disk() - _collected_files()
        assert not invisible, (
            "test files exist outside the gate's collected set — they never run; "
            f"register them in pyproject testpaths: {sorted(invisible)}"
        )
