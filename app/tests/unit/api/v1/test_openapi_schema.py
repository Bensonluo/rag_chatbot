"""OpenAPI schema contract: the /docs storefront stays honest.

The schema is enriched by ``app.api.v1.openapi``; these tests pin the
invariants that the previous revision of that module silently broke —
placeholder hosts, a hardcoded version, and tag metadata that lagged the
routers actually mounted in the tree.
"""

import json
from importlib.metadata import version as package_version

from fastapi import FastAPI

from app.api.v1.openapi import REPOSITORY_URL, STOREFRONT_URL, setup_openapi


def _schema() -> dict[str, object]:
    # Reuse the production factory so the schema reflects mounted routers
    # (guest/chat/handoff/...) rather than a hand-picked subset.
    from app.main import create_app

    app = create_app()
    return app.openapi()


def test_schema_title_and_version_follow_the_package() -> None:
    schema = _schema()
    info = schema["info"]
    assert info["title"] == "Agent Store"
    assert info["version"] == package_version("rag-chatbot")


def test_no_placeholder_hosts_anywhere_in_the_schema() -> None:
    serialized = json.dumps(_schema())
    assert "example.com" not in serialized
    assert "1.0.0" not in serialized  # the old hardcoded version


def test_every_route_tag_has_metadata() -> None:
    schema = _schema()
    documented = {tag["name"] for tag in schema["tags"]}
    used: set[str] = set()
    for path_item in schema["paths"].values():
        for operation in path_item.values():
            if isinstance(operation, dict):
                used.update(operation.get("tags", []))
    missing = used - documented
    assert not missing, f"tags without metadata: {sorted(missing)}"


def test_servers_license_and_bearer_scheme_are_real() -> None:
    schema = _schema()
    servers = {server["url"]: server["description"] for server in schema["servers"]}
    assert "http://localhost:8000" in servers  # contributors first
    assert STOREFRONT_URL in servers
    assert schema["info"]["license"]["name"] == "MIT"
    assert schema["info"]["contact"]["url"] == f"{REPOSITORY_URL}/issues"
    assert schema["components"]["securitySchemes"]["BearerAuth"]["scheme"] == "bearer"


def test_setup_openapi_is_idempotent_per_app() -> None:
    app = FastAPI(title="t", version="0")
    setup_openapi(app)
    first = app.openapi()
    second = app.openapi()
    assert first is second  # cached on the app instance, not rebuilt per request
