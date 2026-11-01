"""Provider-owned category discovery over the internal wire.

The ``/commerce/categories`` endpoint publishes only authoritative facts:
category id/name/description plus attribute names and allowed values. The
provider's internal search aliases stay private — mapping an open user need
onto standard categories is the LLM Agent's semantic job, never a
provider-published phrase table.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from app.commerce_provider.catalog import CATEGORIES, CATEGORY_COUNT, VirtualCatalog
from app.tests.unit.commerce_provider.test_api import AUTH_HEADERS


def test_categories_publish_the_authoritative_taxonomy(client: TestClient) -> None:
    response = client.get("/commerce/categories", headers=AUTH_HEADERS, params={"owner_user_id": 1})
    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    categories = body["categories"]
    assert len(categories) == CATEGORY_COUNT == len(CATEGORIES)

    assert {category["name"] for category in categories} == {spec.name for spec in CATEGORIES}
    by_id = {category["category_id"]: category for category in categories}
    monitor = by_id["monitor"]
    assert monitor["name"] == "显示器"
    assert monitor["description"]
    sizes = next(attribute for attribute in monitor["attributes"] if attribute["name"] == "尺寸")
    assert sizes["allowed_values"] == ["24英寸", "27英寸", "32英寸", "34英寸"]


def test_category_facts_carry_no_alias_routing(client: TestClient) -> None:
    body: dict[str, Any] = client.get(
        "/commerce/categories", headers=AUTH_HEADERS, params={"owner_user_id": 1}
    ).json()
    for category in body["categories"]:
        # Typed facts only: id, name, description, attribute vocabulary.
        assert set(category) == {"category_id", "name", "description", "attributes"}
        for attribute in category["attributes"]:
            assert set(attribute) == {"name", "allowed_values"}
            assert attribute["allowed_values"]
    assert "aliases" not in repr(body)


def test_categories_require_a_valid_owner_identity(client: TestClient) -> None:
    missing = client.get("/commerce/categories", headers=AUTH_HEADERS)
    assert missing.status_code == 422
    invalid = client.get("/commerce/categories", headers=AUTH_HEADERS, params={"owner_user_id": 0})
    assert invalid.status_code == 422


def test_categories_need_the_bearer_token(client: TestClient) -> None:
    response = client.get("/commerce/categories", params={"owner_user_id": 1})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_core_taxonomy_matches_the_published_catalog() -> None:
    taxonomy = VirtualCatalog().taxonomy()
    assert len(taxonomy.categories) == CATEGORY_COUNT
    names = {category.name for category in taxonomy.categories}
    assert names == {spec.name for spec in CATEGORIES}
    for category in taxonomy.categories:
        spec = next(spec for spec in CATEGORIES if spec.category_id == category.category_id)
        assert category.description == spec.blurb
        assert [(a.name, tuple(a.allowed_values)) for a in category.attributes] == [
            (key, values) for key, values in spec.attrs
        ]


def test_categories_support_a_bounded_id_filter(client: TestClient) -> None:
    response = client.get(
        "/commerce/categories",
        headers=AUTH_HEADERS,
        params={"owner_user_id": 1, "category_ids": "monitor,keyboard"},
    )
    assert response.status_code == 200
    categories = response.json()["categories"]
    assert [category["category_id"] for category in categories] == ["monitor", "keyboard"]
    # Filtered entries publish the same full fact shape as the unfiltered list.
    for category in categories:
        assert set(category) == {"category_id", "name", "description", "attributes"}


def test_categories_filter_matching_nothing_is_not_found(client: TestClient) -> None:
    # Partial match publishes only the existing categories.
    partial = client.get(
        "/commerce/categories",
        headers=AUTH_HEADERS,
        params={"owner_user_id": 1, "category_ids": "monitor,does-not-exist"},
    )
    assert partial.status_code == 200
    assert [category["category_id"] for category in partial.json()["categories"]] == ["monitor"]

    # A filter that matches nothing at all is a definitive refusal.
    empty = client.get(
        "/commerce/categories",
        headers=AUTH_HEADERS,
        params={"owner_user_id": 1, "category_ids": "does-not-exist"},
    )
    assert empty.status_code == 404
    assert empty.json()["error"]["code"] == "not_found"


def test_categories_reject_malformed_id_filters(client: TestClient) -> None:
    for raw in (",", "a" * 65, ",".join(f"id{i}" for i in range(33))):
        response = client.get(
            "/commerce/categories",
            headers=AUTH_HEADERS,
            params={"owner_user_id": 1, "category_ids": raw},
        )
        assert response.status_code == 422, raw
        assert response.json()["error"]["code"] == "rejected"
