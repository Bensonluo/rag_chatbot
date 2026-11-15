"""Rot pin: the README-referenced MCP example client actually runs.

``examples/mcp_shopping_client.py`` is the public face of the MCP surface —
the README points readers at it, and no competitor ships CI-verified
examples. Examples rot silently (an SDK rename, a schema change, a moved
flag) because nothing imports them; this pin launches the real MCP server
as a subprocess on an ephemeral port and drives the example's own ``run()``
through the complete purchase journey over the wire — search → cart →
quote → order → idempotent order replay → sandbox payment — so any drift
between the example, the SDK, and the server fails the suite instead of
shipping a broken demo to readers.
"""

from __future__ import annotations

import importlib.util
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = REPO_ROOT / "examples" / "mcp_shopping_client.py"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _load_example() -> ModuleType:
    """Import the example by path — examples/ is not a package."""
    spec = importlib.util.spec_from_file_location("mcp_shopping_client", EXAMPLE)
    assert spec is not None and spec.loader is not None, f"cannot import {EXAMPLE}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def live_server(tmp_path: Path) -> Iterator[str]:
    """The real MCP server as a subprocess: empty token, throwaway SQLite."""
    port = _free_port()
    proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable,
            "-m",
            "app.commerce_provider.mcp_server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--database",
            str(tmp_path / "rot-pin.sqlite3"),
            "--token",
            "",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20.0
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("mcp_server subprocess died during startup")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.2)
        else:
            raise RuntimeError("mcp_server did not listen within 20s")
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


async def test_example_client_walks_the_purchase_journey(
    live_server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The example self-verifies its journey (idempotent replay, paid status)
    # and exits non-zero on any refusal or check failure; a clean return
    # means the documented walkthrough still works against this server.
    example = _load_example()
    monkeypatch.setenv("DEMO_QUERY", "显示器")  # proven catalog match in the suite
    await example.run(live_server, token=None, auto_yes=True)
