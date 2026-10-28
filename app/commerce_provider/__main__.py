"""Command entrypoint: ``python -m app.commerce_provider``.

Runs the internal virtual commerce provider as a standalone uvicorn service
against a SQLite database file. Every setting defaults to
``COMMERCE_PROVIDER_*`` environment variables; CLI flags override them.
"""

from __future__ import annotations

import argparse

import uvicorn

from app.commerce_provider.api import create_app
from app.commerce_provider.settings import CommerceProviderSettings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.commerce_provider",
        description="Run the internal virtual commerce provider service",
    )
    parser.add_argument("--host", default=None, help="bind address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default 8071)")
    parser.add_argument(
        "--database", dest="database_path", default=None, help="SQLite database file path"
    )
    parser.add_argument(
        "--token", dest="auth_token", default=None, help="Bearer token (empty disables auth)"
    )
    parser.add_argument("--quote-ttl-seconds", type=int, default=None, help="quote validity window")
    parser.add_argument(
        "--faults-enabled",
        action="store_true",
        help="enable the test-only fault injection endpoints",
    )
    args = parser.parse_args(argv)

    settings = CommerceProviderSettings()
    updates: dict[str, object] = {}
    if args.host is not None:
        updates["host"] = args.host
    if args.port is not None:
        updates["port"] = args.port
    if args.database_path is not None:
        updates["database_path"] = args.database_path
    if args.auth_token is not None:
        updates["auth_token"] = args.auth_token
    if args.quote_ttl_seconds is not None:
        updates["quote_ttl_seconds"] = args.quote_ttl_seconds
    if args.faults_enabled:
        updates["faults_enabled"] = True
    if updates:
        settings = settings.model_copy(update=updates)

    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
