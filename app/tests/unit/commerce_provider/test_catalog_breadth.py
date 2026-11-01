"""Breadth and per-domain behavior of the expanded marketplace catalog.

The catalog grew from 25 office-electronics categories to a 151-category
general marketplace across 18 domains. Breadth lives in categories; depth
stays deliberately shallow (32 brands × 8 models × 4 variants per category)
and every category owns its variant axis (颜色/尺码/规格/...), which is
registry-internal metadata, never a public wire field.
"""

from __future__ import annotations

from collections import Counter

import pytest

from app.commerce_provider.catalog import (
    BRAND_COUNT,
    CATEGORIES,
    CATEGORY_COUNT,
    MODELS_PER_PAIR,
    TOTAL_SKUS,
    VARIANTS_PER_MODEL,
    VirtualCatalog,
    default_taxonomy,
)
from app.commerce_provider.errors import ProviderRejectedError
from app.models.schemas.commerce import CatalogFilters


def test_catalog_spans_eighteen_domains_with_151_categories() -> None:
    assert CATEGORY_COUNT == len(CATEGORIES) == 151
    per_domain = Counter(spec.domain for spec in CATEGORIES)
    assert len(per_domain) == 18
    assert per_domain["office_electronics"] == 25  # the original block, verbatim
    for domain in (
        "food",
        "apparel",
        "shoes_bags",
        "accessories",
        "beauty",
        "home_daily",
        "kitchen",
        "home_textile",
        "furniture",
        "appliances",
        "digital",
        "mother_baby",
        "sports",
        "entertainment",
        "pets",
        "auto",
        "health",
    ):
        assert per_domain[domain] >= 3, domain


def test_every_category_declares_a_clean_variant_axis() -> None:
    for spec in CATEGORIES:
        assert spec.variant_key.strip(), spec.category_id
        assert len(spec.variant_values) == VARIANTS_PER_MODEL, spec.category_id
        assert len(set(spec.variant_values)) == VARIANTS_PER_MODEL, spec.category_id
        assert spec.variant_key not in {key for key, _ in spec.attrs}, spec.category_id


def test_sku_space_is_composed_from_the_taxonomy() -> None:
    assert TOTAL_SKUS == CATEGORY_COUNT * BRAND_COUNT * MODELS_PER_PAIR * VARIANTS_PER_MODEL
    catalog = VirtualCatalog()
    assert catalog.category_count == CATEGORY_COUNT
    assert catalog.pair_count == CATEGORY_COUNT * BRAND_COUNT
    assert catalog.total_skus == TOTAL_SKUS


@pytest.mark.parametrize(
    "domain,term,target_id",
    [
        ("office_electronics", "显示器", "monitor"),
        ("food", "海鲜", "seafood"),
        ("apparel", "连衣裙", "dress"),
        ("shoes_bags", "行李箱", "luggage"),
        ("accessories", "腕表", "watches"),
        ("beauty", "口红", "makeup"),
        ("home_daily", "雨伞", "umbrellas"),
        ("kitchen", "餐具", "tableware"),
        ("home_textile", "被芯", "blankets"),
        ("furniture", "沙发", "sofa"),
        ("appliances", "油烟机", "kitchen_appliance"),
        ("digital", "平板电脑", "tablet"),
        ("mother_baby", "尿不湿", "diapers"),
        ("sports", "帐篷", "camping"),
        ("entertainment", "盲盒", "collectibles"),
        ("pets", "猫砂", "pet_supplies"),
        ("auto", "行车记录仪", "car_electronics"),
        ("health", "血糖仪", "health_devices"),
    ],
)
def test_search_smoke_reaches_every_domain(domain: str, term: str, target_id: str) -> None:
    catalog = VirtualCatalog()
    page = catalog.search(query=term, page_size=5)
    assert page.items, f"no hits for {domain}/{term}"
    # Alias matching is deliberately permissive (substring rules), so a term
    # may legitimately reach neighbouring categories; the smoke contract is
    # that every domain stays reachable through natural shopping language.
    hit_ids = {
        CATEGORIES[
            catalog.decompose(catalog.parse_sku_id(item.sku_id) or 0)[0] // BRAND_COUNT
        ].category_id
        for item in page.items
    }
    assert target_id in hit_ids


def test_variant_axis_addresses_and_filters_per_category() -> None:
    catalog = VirtualCatalog()
    tshirt = next(spec for spec in CATEGORIES if spec.category_id == "tshirt")
    start = CATEGORIES.index(tshirt) * BRAND_COUNT * MODELS_PER_PAIR * VARIANTS_PER_MODEL
    item = catalog.materialize(start + 2)  # first brand, first model, variant 2
    assert item.attributes["尺码"] == tshirt.variant_values[2]

    filtered = catalog.search(
        query="T恤",
        filters=CatalogFilters(attributes={"尺码": ["L"]}),
        page_size=5,
    )
    assert filtered.items
    assert all(item.attributes["尺码"] == "L" for item in filtered.items)

    fresh = next(spec for spec in CATEGORIES if spec.category_id == "fresh_produce")
    fresh_start = CATEGORIES.index(fresh) * BRAND_COUNT * MODELS_PER_PAIR * VARIANTS_PER_MODEL
    fresh_item = catalog.materialize(fresh_start + 1)
    assert fresh_item.attributes["规格"] == fresh.variant_values[1]


def test_generated_catalog_rejects_invalid_variant_axes() -> None:
    base = default_taxonomy()

    wrong_count = base.model_copy(deep=True)
    wrong_count.categories[0].variant_values = ["黑色", "白色"]
    with pytest.raises(ProviderRejectedError):
        VirtualCatalog(wrong_count)

    blank_key = base.model_copy(deep=True)
    blank_key.categories[0].variant_key = "  "
    with pytest.raises(ProviderRejectedError):
        VirtualCatalog(blank_key)

    collides = base.model_copy(deep=True)
    collides.categories[0].variant_key = collides.categories[0].attributes[0].name
    with pytest.raises(ProviderRejectedError):
        VirtualCatalog(collides)
