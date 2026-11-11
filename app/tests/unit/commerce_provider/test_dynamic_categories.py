"""Dynamic category minting: validation, dedupe, cap, persistence, purchase."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from app.commerce_provider.catalog import BRAND_COUNT, VirtualCatalog
from app.commerce_provider.core import CommerceProviderCore
from app.commerce_provider.errors import ProviderConflictError, ProviderRejectedError
from app.commerce_provider.faults import FaultController
from app.commerce_provider.storage import Database
from app.models.schemas.commerce import CartLineSelection
from app.tests.unit.commerce_provider.conftest import AUTH_HEADERS

OWNER = 1

_STATIC_PAIRS = VirtualCatalog().pair_count

CHOCOLATE = {
    "name": "糖果巧克力",
    "description": "可可制品甜食",
    "domain": "food",
    "attributes": [("类型", ["黑巧", "牛奶", "白巧", "夹心"])],
    "variant_key": "规格",
    "variant_values": ["45g", "100g", "250g", "礼盒"],
    "base_price_minor": 1500,
    "source_query": "我想吃巧克力",
}


def _proposal(**overrides: Any) -> dict[str, Any]:
    merged = {**CHOCOLATE, **overrides}
    return merged


async def _mint(core: CommerceProviderCore, request_key: str, **overrides: object):
    return await core.propose_category(OWNER, request_key=request_key, **_proposal(**overrides))


async def test_mint_makes_category_immediately_searchable_and_purchasable(
    core: CommerceProviderCore,
) -> None:
    assert (await core.search_products(OWNER, "巧克力")).total_estimate == 0
    result = await _mint(core, "mint-1")
    assert result["status"] == "minted"
    assert result["category"]["category_id"] == "dyn_0001"
    assert result["dynamic_category_count"] == 1

    page = await core.search_products(OWNER, "巧克力")
    assert page.total_estimate > 0
    item = page.items[0]
    assert "糖果巧克力" in item.name
    assert item.available_quantity >= 1

    cart = await core.update_cart_lines(
        OWNER, None, None, [CartLineSelection(sku_id=item.sku_id, quantity=1)], "cart-1"
    )
    quote = await core.prepare_checkout(OWNER, cart.cart_id, cart.cart_version, "quote-1")
    order = await core.create_order(OWNER, quote.quote_id, "order-1")
    payment = await core.create_payment(OWNER, order.order_id, "pay-1")
    paid = await core.mock_pay(OWNER, payment.payment_id, "pay-2")
    assert paid.status == "paid"
    assert (await core.get_order(OWNER, order.order_id)).status == "paid"


async def test_mint_dedupe_returns_existing_twin(core: CommerceProviderCore) -> None:
    first = await _mint(core, "mint-1")
    twin = await core.propose_category(
        OWNER,
        request_key="mint-2",
        **_proposal(name="糖果 巧克力", base_price_minor=9900),
    )
    assert twin["status"] == "existing"
    assert twin["category"]["category_id"] == first["category"]["category_id"]
    # The existing definition wins; the twin's price is not adopted.
    assert twin["category"]["base_price_minor"] == 1500
    assert twin["dynamic_category_count"] == 1


async def test_mint_replay_is_idempotent_by_request_key(
    core: CommerceProviderCore,
) -> None:
    first = await _mint(core, "mint-1")
    replay = await _mint(core, "mint-1")
    assert replay["status"] == "minted"
    assert replay["category"]["category_id"] == first["category"]["category_id"]
    with pytest.raises(ProviderConflictError, match="不同的请求参数"):
        await _mint(core, "mint-1", description="不同的参数")


async def test_mint_rejects_domains_outside_the_fixed_frame(
    core: CommerceProviderCore,
) -> None:
    with pytest.raises(ProviderRejectedError, match="固定域框架"):
        await _mint(core, "mint-1", domain="pharma")


async def test_mint_accepts_general_fallback_domain(core: CommerceProviderCore) -> None:
    result = await _mint(core, "mint-1", name="古琴琴弦", domain="general", base_price_minor=8800)
    assert result["status"] == "minted"
    assert result["category"]["domain"] == "general"


async def test_mint_blocklist(core: CommerceProviderCore) -> None:
    for name, description in (
        ("处方药代购", "正规描述"),
        ("宠物用品", "含活体动物售卖"),
    ):
        with pytest.raises(ProviderRejectedError, match="不支持"):
            await _mint(core, f"mint-{name}", name=name, description=description)


async def test_mint_blocklist_stays_narrow(core: CommerceProviderCore) -> None:
    # Kitchen knives and health supplements must not be collateral damage.
    kitchen = await _mint(
        core,
        "mint-1",
        name="厨房刀具",
        description="砧板菜刀套装",
        domain="kitchen",
        attributes=[("类型", ["菜刀", "水果刀", "剪刀"])],
        variant_key="件数",
        variant_values=["1件", "2件", "4件", "7件"],
        base_price_minor=4900,
    )
    assert kitchen["status"] == "minted"
    supplements = await _mint(
        core,
        "mint-2",
        name="保健食品",
        description="维生素与蛋白粉",
        domain="health",
        base_price_minor=12000,
    )
    assert supplements["status"] == "minted"


async def test_mint_shape_constraints(core: CommerceProviderCore) -> None:
    with pytest.raises(ProviderRejectedError, match="变体维度必须提供 4"):
        await _mint(core, "mint-1", variant_values=["小", "中", "大"])
    with pytest.raises(ProviderRejectedError, match="取值不能重复"):
        await _mint(core, "mint-2", variant_values=["小", "中", "大", "小"])
    with pytest.raises(ProviderRejectedError, match="变体维度不能与属性名重复"):
        await _mint(core, "mint-3", variant_key="规格", attributes=[("规格", ["小"])])
    with pytest.raises(ProviderRejectedError, match="属性轴数量"):
        await _mint(
            core,
            "mint-4",
            attributes=[(f"属性{i}", ["值"]) for i in range(6)],
        )
    with pytest.raises(ProviderRejectedError, match="属性名重复"):
        await _mint(
            core,
            "mint-5",
            attributes=[("口味", ["甜"]), ("口味", ["咸"])],
        )
    with pytest.raises(ProviderRejectedError, match="基准价格"):
        await _mint(core, "mint-6", base_price_minor=10)
    with pytest.raises(ProviderRejectedError, match="基准价格"):
        await _mint(core, "mint-7", base_price_minor=10_000_100)


async def test_mint_cap_refuses_without_evicting(tmp_path: Path) -> None:
    database = Database(str(tmp_path / "capped.sqlite3"))
    await database.startup()
    core = CommerceProviderCore(
        database,
        VirtualCatalog(),
        faults=FaultController(database, enabled=True),
        dynamic_category_limit=1,
    )
    first = await _mint(core, "mint-1")
    assert first["status"] == "minted"
    with pytest.raises(ProviderRejectedError, match="上限"):
        await _mint(core, "mint-2", name="充电宝", domain="digital")
    # The refused mint neither registers nor evicts the existing one.
    assert core.catalog.dynamic_category_count == 1
    assert (await core.search_products(OWNER, "巧克力")).total_estimate > 0


async def test_restart_reload_keeps_addressing_and_orders_stable(
    core: CommerceProviderCore, tmp_path: Path
) -> None:
    await _mint(core, "mint-1")
    second = await _mint(
        core,
        "mint-2",
        name="充电宝",
        description="便携储能电源",
        domain="digital",
        attributes=[("容量", ["5000mAh", "10000mAh", "20000mAh"])],
        variant_key="颜色",
        variant_values=["黑", "白", "蓝", "银"],
        base_price_minor=9900,
    )
    assert second["category"]["category_id"] == "dyn_0002"

    sku = (await core.search_products(OWNER, "充电宝")).items[0].sku_id
    cart = await core.update_cart_lines(
        OWNER, None, None, [CartLineSelection(sku_id=sku, quantity=1)], "cart-1"
    )
    quote = await core.prepare_checkout(OWNER, cart.cart_id, cart.cart_version, "quote-1")
    order = await core.create_order(OWNER, quote.quote_id, "order-1")

    restarted = CommerceProviderCore(
        Database(core.database.path), VirtualCatalog(), dynamic_category_limit=200
    )
    assert await restarted.load_dynamic_categories() == 2
    item = await restarted.get_product(OWNER, sku)
    assert item.sku_id == sku
    assert item.unit_price_minor == (await core.get_product(OWNER, sku)).unit_price_minor
    assert (await restarted.get_order(OWNER, order.order_id)).order_id == order.order_id
    assert restarted.catalog.dynamic_category_count == 2


LARD_STYLES = [
    {"brand": "金穗坊", "series": "小榨原香"},
    {"brand": "澄香", "series": "家庭装"},
    {"brand": "鲁源坊", "series": "轻食"},
    {"brand": "金穗坊", "series": "礼享装"},
    {"brand": "澄香", "series": "经典"},
]


async def test_mint_with_brand_styles_names_shelf_realistic_skus(
    core: CommerceProviderCore,
) -> None:
    """The LLM-authored naming matrix replaces the static brand grid.

    A minted category with brand_styles names its SKUs from the matrix
    (金穗坊 小榨原香 猪油, never 恒锐 Astro A0000 猪油) and one search
    page spans multiple brands instead of a single static brand's series
    line.
    """

    result = await core.propose_category(
        OWNER,
        request_key="mint-styles-1",
        name="猪油",
        description="厨房烹饪用动物油脂",
        domain="food",
        attributes=[("工艺", ["冷榨", "热榨", "精炼"])],
        variant_key="规格",
        variant_values=["250g", "500g", "1kg", "2kg"],
        base_price_minor=2300,
        source_query="有猪油卖吗",
        brand_styles=LARD_STYLES,
    )
    assert result["status"] == "minted"
    assert result["category"]["brand_styles"] == LARD_STYLES

    page = await core.search_products(OWNER, "猪油")
    assert page.total_estimate > 0
    matrix_brands = {style["brand"] for style in LARD_STYLES}
    matrix_pairs = {(style["brand"], style["series"]) for style in LARD_STYLES}
    seen_brands: set[str] = set()
    for item in page.items:
        assert "恒锐" not in item.name and "Astro" not in item.name
        assert not re.search(r"[A-Z]\d{4}", item.name), item.name
        brand = item.attributes["品牌"]
        series = item.attributes["系列"]
        assert (brand, series) in matrix_pairs
        seen_brands.add(brand)
    # One page mixes brands: the reason the static grid felt fake.
    assert len(seen_brands) >= 2
    assert seen_brands <= matrix_brands
    # Brand tokens find the minted category (plan-level brand resolution).
    assert (await core.search_products(OWNER, "金穗坊")).total_estimate > 0
    # A style-authored category occupies exactly one pair.
    assert core.catalog.pair_count == _STATIC_PAIRS + 1


async def test_mint_without_brand_styles_keeps_legacy_grid_naming(
    core: CommerceProviderCore,
) -> None:
    """Legacy proposals (no matrix) keep the historical BRAND_COUNT grid.

    Existing minted categories and their SKU names must never change under
    a live order: absent brand_styles falls back to the static grid.
    """

    result = await _mint(core, "mint-1")
    assert result["category"]["brand_styles"] is None
    item = (await core.search_products(OWNER, "巧克力")).items[0]
    assert "恒锐" in item.name and re.search(r"[A-Z]\d{4}", item.name)
    assert core.catalog.pair_count == _STATIC_PAIRS + BRAND_COUNT


async def test_brand_styles_shape_is_enforced(core: CommerceProviderCore) -> None:
    too_few = LARD_STYLES[:3]
    with pytest.raises(ProviderRejectedError, match="4-8"):
        await _mint(core, "mint-styles-1", name="猪油", brand_styles=too_few)
    single_brand = [{"brand": "金穗坊", "series": f"系列{i}"} for i in range(4)]
    with pytest.raises(ProviderRejectedError, match="品牌"):
        await _mint(core, "mint-styles-2", name="猪油", brand_styles=single_brand)
    duplicated = [
        {"brand": "金穗坊", "series": "小榨原香"},
        {"brand": "澄香", "series": "家庭装"},
        {"brand": "鲁源坊", "series": "轻食"},
        {"brand": "金穗坊 ", "series": " 小榨原香"},
    ]
    with pytest.raises(ProviderRejectedError, match="重复"):
        await _mint(core, "mint-styles-3", name="猪油", brand_styles=duplicated)
    blocklisted = [*LARD_STYLES[:3], {"brand": "处方", "series": "经典"}]
    with pytest.raises(ProviderRejectedError):
        await _mint(core, "mint-styles-4", name="猪油", brand_styles=blocklisted)
    # Nothing above registered: the shelf stays clean for a valid retry.
    assert core.catalog.dynamic_category_count == 0
    valid = await _mint(core, "mint-styles-5", name="猪油", brand_styles=LARD_STYLES)
    assert valid["status"] == "minted"


async def test_brand_styles_survive_restart_with_stable_names(
    core: CommerceProviderCore, tmp_path: Path
) -> None:
    await core.propose_category(
        OWNER,
        request_key="mint-styles-1",
        name="猪油",
        description="厨房烹饪用动物油脂",
        domain="food",
        attributes=[("工艺", ["冷榨", "热榨", "精炼"])],
        variant_key="规格",
        variant_values=["250g", "500g", "1kg", "2kg"],
        base_price_minor=2300,
        brand_styles=LARD_STYLES,
    )
    before = (await core.search_products(OWNER, "猪油")).items[0]

    restarted = CommerceProviderCore(
        Database(core.database.path), VirtualCatalog(), dynamic_category_limit=200
    )
    assert await restarted.load_dynamic_categories() == 1
    after = await restarted.get_product(OWNER, before.sku_id)
    assert after.name == before.name
    assert after.unit_price_minor == before.unit_price_minor
    assert after.attributes["品牌"] == before.attributes["品牌"]
    # A style-authored reload keeps the single-pair footprint.
    assert restarted.catalog.pair_count == _STATIC_PAIRS + 1


def test_dynamic_mint_endpoint_roundtrip(client) -> None:
    response = client.post(
        "/commerce/categories/dynamic",
        headers=AUTH_HEADERS,
        json={
            "owner_user_id": OWNER,
            "request_key": "http-mint-1",
            "name": "糖果巧克力",
            "description": "可可制品甜食",
            "domain": "food",
            "attributes": [{"name": "类型", "allowed_values": ["黑巧", "牛奶"]}],
            "variant_key": "规格",
            "variant_values": ["45g", "100g", "250g", "礼盒"],
            "base_price_minor": 1500,
            "source_query": "我想吃巧克力",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "minted"
    assert body["category"]["category_id"] == "dyn_0001"

    search = client.post(
        "/commerce/search",
        headers=AUTH_HEADERS,
        json={"owner_user_id": OWNER, "query": "巧克力"},
    )
    assert search.status_code == 200
    assert search.json()["total_estimate"] > 0

    twin = client.post(
        "/commerce/categories/dynamic",
        headers=AUTH_HEADERS,
        json={
            "owner_user_id": OWNER,
            "request_key": "http-mint-2",
            "name": "糖果巧克力",
            "description": "同物异名提案",
            "domain": "food",
            "attributes": [{"name": "类型", "allowed_values": ["黑巧"]}],
            "variant_key": "规格",
            "variant_values": ["45g", "100g", "250g", "礼盒"],
            "base_price_minor": 9900,
        },
    )
    assert twin.status_code == 200
    assert twin.json()["status"] == "existing"


def test_dynamic_mint_endpoint_maps_rejections(client) -> None:
    blocked = client.post(
        "/commerce/categories/dynamic",
        headers=AUTH_HEADERS,
        json={
            "owner_user_id": OWNER,
            "request_key": "http-mint-blocked",
            "name": "处方药",
            "description": "违规",
            "domain": "health",
            "attributes": [{"name": "规格", "allowed_values": ["1盒"]}],
            "variant_key": "颜色",
            "variant_values": ["红", "蓝", "白", "黑"],
            "base_price_minor": 1500,
        },
    )
    assert blocked.status_code == 422
    assert blocked.json()["error"]["code"] == "rejected"
