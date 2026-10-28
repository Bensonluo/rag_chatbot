"""Determinism and search behavior of the virtual catalog."""

from __future__ import annotations

from typing import cast

import pytest

from app.commerce_provider.catalog import (
    MAX_WINDOW,
    TOTAL_SKUS,
    VARIANTS_PER_MODEL,
    VirtualCatalog,
)
from app.commerce_provider.errors import ProviderRejectedError
from app.models.schemas.commerce import CatalogFilters


@pytest.fixture
def catalog() -> VirtualCatalog:
    return VirtualCatalog()


def test_sku_space_is_stable_and_large(catalog: VirtualCatalog) -> None:
    assert TOTAL_SKUS >= 100_000_000
    for n in (0, 1, 12_345_678, TOTAL_SKUS - 1):
        assert catalog.parse_sku_id(catalog.sku_id(n)) == n
        pair, model, variant = catalog.decompose(n)
        assert 0 <= pair < 800 and 0 <= model < 31_250 and 0 <= variant < VARIANTS_PER_MODEL
    assert catalog.parse_sku_id("SKU-100000000") is None  # 9 digits: out of space
    assert catalog.parse_sku_id("PRD-000-00000") is None
    assert catalog.parse_sku_id("") is None


def test_facts_are_pure_functions_of_the_index(catalog: VirtualCatalog) -> None:
    first = catalog.materialize(0)
    assert first == catalog.materialize(0)
    assert first.attributes["颜色"] == "黑色"
    assert 1 <= first.max_order_quantity <= 10
    other = catalog.materialize(1)
    assert other.sku_id != first.sku_id


def test_variants_share_product_but_differ_in_sku(catalog: VirtualCatalog) -> None:
    base = catalog.materialize(4)
    neighbour = catalog.materialize(5)
    assert base.product_id == neighbour.product_id
    assert base.sku_id != neighbour.sku_id
    assert base.attributes["颜色"] != neighbour.attributes["颜色"]
    assert base.unit_price_minor == neighbour.unit_price_minor


def test_search_orders_the_unfiltered_stream_by_index(catalog: VirtualCatalog) -> None:
    include_out_of_stock = catalog.search(
        query="", filters=CatalogFilters(in_stock=False), page_size=50
    )
    assert [item.sku_id for item in include_out_of_stock.items] == [
        catalog.sku_id(n) for n in range(50)
    ]
    in_stock = catalog.search(query="", page_size=50)
    assert all(item.available_quantity > 0 for item in in_stock.items)
    expected = [n for n in range(80) if catalog._base_availability(n) > 0][:50]
    assert [catalog.parse_sku_id(item.sku_id) for item in in_stock.items] == expected


def test_search_matches_category_tokens(catalog: VirtualCatalog) -> None:
    page = catalog.search(query="显示器", page_size=10)
    assert page.items
    assert all("显示器" in item.name for item in page.items)
    assert all(item.available_quantity > 0 for item in page.items)


def test_search_intersects_brand_and_category(catalog: VirtualCatalog) -> None:
    page = catalog.search(query="拓普 显示器", page_size=10)
    assert page.items
    for item in page.items:
        assert item.attributes["品牌"] == "拓普"
        assert "显示器" in item.name


def test_search_applies_price_bounds(catalog: VirtualCatalog) -> None:
    page = catalog.search(
        query="耳机",
        filters=CatalogFilters(min_price_minor=10_000, max_price_minor=40_000),
        page_size=20,
    )
    assert page.items
    assert all(10_000 <= item.unit_price_minor <= 40_000 for item in page.items)


def test_search_applies_attribute_filters(catalog: VirtualCatalog) -> None:
    page = catalog.search(
        query="显示器",
        filters=CatalogFilters(attributes={"分辨率": ["4K"]}),
        page_size=20,
    )
    assert page.items
    assert all(item.attributes["分辨率"] == "4K" for item in page.items)

    brand_page = catalog.search(
        query="显示器",
        filters=CatalogFilters(attributes={"品牌": ["拓普"], "颜色": ["黑色"]}),
        page_size=20,
    )
    assert brand_page.items
    for item in brand_page.items:
        assert item.attributes["品牌"] == "拓普"
        assert item.attributes["颜色"] == "黑色"


def test_search_rejects_unknown_attribute_keys_for_a_category(
    catalog: VirtualCatalog,
) -> None:
    page = catalog.search(
        query="显示器",
        filters=CatalogFilters(attributes={"轴体": ["红轴"]}),
        page_size=10,
    )
    # "轴体" only exists on keyboards; the monitor pair cannot satisfy it.
    assert page.items == []


def test_search_paginates_with_bound_tokens(catalog: VirtualCatalog) -> None:
    first = catalog.search(query="耳机", page_size=5)
    assert first.next_page_token is not None
    second = catalog.search(query="耳机", page_size=5, page_token=first.next_page_token)
    first_skus = {item.sku_id for item in first.items}
    second_skus = {item.sku_id for item in second.items}
    assert first_skus and second_skus
    assert first_skus.isdisjoint(second_skus)

    with pytest.raises(ProviderRejectedError):
        catalog.search(query="显示器", page_size=5, page_token=first.next_page_token)

    with pytest.raises(ProviderRejectedError):
        catalog.search(query="耳机", page_size=5, page_token="not-a-token")


def test_search_sorts_by_price_and_newest(catalog: VirtualCatalog) -> None:
    ascending = catalog.search(query="显示器", sort="price_asc", page_size=30)
    prices = [item.unit_price_minor for item in ascending.items]
    assert prices == sorted(prices)

    descending = catalog.search(query="显示器", sort="price_desc", page_size=30)
    prices_desc = [item.unit_price_minor for item in descending.items]
    assert prices_desc == sorted(prices_desc, reverse=True)

    newest = catalog.search(query="显示器", sort="newest", page_size=30)
    parsed = [catalog.parse_sku_id(item.sku_id) for item in newest.items]
    indexes = cast(list[int], [index for index in parsed if index is not None])
    assert indexes == sorted(indexes, reverse=True)


def test_search_returns_empty_for_unmatched_terms(catalog: VirtualCatalog) -> None:
    page = catalog.search(query="绝不存在的商品词汇", page_size=5)
    assert page.items == []
    assert page.total_estimate == 0
    assert page.next_page_token is None


def test_search_window_is_bounded(catalog: VirtualCatalog) -> None:
    assert MAX_WINDOW == 8_192
    page = catalog.search(query="显示器", page_size=50)
    assert page.total_estimate == 1_024  # first-window collection size
