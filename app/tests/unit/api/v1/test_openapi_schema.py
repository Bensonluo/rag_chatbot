"""OpenAPI schema contract: the /docs storefront stays honest.

The schema is enriched by ``app.api.v1.openapi``; these tests pin the
invariants that the previous revision of that module silently broke —
placeholder hosts, a hardcoded version, and tag metadata that lagged the
routers actually mounted in the tree.
"""

import json
from importlib.metadata import version as package_version
from typing import Any

from fastapi import FastAPI

from app.api.v1.openapi import REPOSITORY_URL, STOREFRONT_URL, setup_openapi


def _schema() -> dict[str, Any]:
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


def test_operations_document_their_error_codes() -> None:
    """Guest and authenticated surfaces declare handler-raised codes."""
    paths = _schema()["paths"]
    guest_chat = paths["/api/v1/guest/chat"]["post"]
    assert {"200", "401", "403", "404", "409", "422", "423", "429", "503"} <= set(
        guest_chat["responses"]
    )
    bootstrap = paths["/api/v1/guest/bootstrap"]["post"]
    # 422 is FastAPI's automatic body-validation response (bootstrap takes a
    # request model); the handler-raised codes are the documented 403/503.
    assert set(bootstrap["responses"]) == {"200", "403", "422", "503"}
    assert "Same-origin browser check failed" in bootstrap["responses"]["403"]["description"]
    # The authenticated surface pins its own sets (chat budget, handoff lock,
    # coordination outages; feedback ownership; missing-session update).
    chat = paths["/api/v1/chat"]["post"]
    assert {"401", "403", "404", "409", "423", "429", "503"} <= set(chat["responses"])
    feedback = paths["/api/v1/feedback"]["post"]
    assert {"404", "409", "503"} <= set(feedback["responses"])
    update = paths["/api/v1/sessions/{session_id}"]["put"]
    assert "404" in update["responses"]  # missing session must not surface as 500-only


def test_every_api_operation_documents_error_codes() -> None:
    """No /api/v1 operation ships without at least one documented error code."""
    undocumented = []
    for path, path_item in _schema()["paths"].items():
        if not path.startswith("/api/v1/") or path.endswith("/health"):
            continue  # app-level routes and health checks answer 200 with detail
        for method, operation in path_item.items():
            if not isinstance(operation, dict):
                continue
            error_codes = {
                code
                for code in operation.get("responses", {})
                if code.isdigit() and int(code) >= 400
            }
            if not error_codes:
                undocumented.append(f"{method.upper()} {path}")
    assert not undocumented, f"operations without documented errors: {sorted(undocumented)}"


def test_documented_error_codes_declare_their_body_schema() -> None:
    """Non-422 error docs carry the {"detail": str} body handlers return.

    422 is excluded: FastAPI's body-validation failures return the
    structured ValidationError array, not handler prose.
    """
    schema = _schema()
    error_detail = schema["components"]["schemas"]["ErrorDetail"]
    assert error_detail["properties"]["detail"]["type"] == "string"
    for path, path_item in schema["paths"].items():
        if not path.startswith("/api/v1/") or path.endswith("/health"):
            continue
        for method, operation in path_item.items():
            if not isinstance(operation, dict):
                continue
            for code, response in operation.get("responses", {}).items():
                if not code.isdigit() or int(code) < 400 or code == "422":
                    continue
                content = response.get("content", {})
                assert "application/json" in content, (
                    f"{method.upper()} {path} {code} documents no JSON body"
                )
                ref = content["application/json"]["schema"].get("$ref", "")
                assert ref.endswith("ErrorDetail"), (
                    f"{method.upper()} {path} {code} body is not ErrorDetail"
                )


def test_setup_openapi_is_idempotent_per_app() -> None:
    app = FastAPI(title="t", version="0")
    setup_openapi(app)
    first = app.openapi()
    second = app.openapi()
    assert first is second  # cached on the app instance, not rebuilt per request
