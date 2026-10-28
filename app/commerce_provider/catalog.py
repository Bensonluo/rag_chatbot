"""Deterministic, scalable virtual catalog for the commerce provider.

The catalog is never enumerated or seeded into a database: every fact
(name, description, attributes, price, availability, purchase cap) is a pure
function of a stable numeric SKU index in ``[0, TOTAL_SKUS)`` — exactly
100,000,000 addressable SKUs projected from a fixed vocabulary of 25 office /
electronics categories, 32 brands, 32 product series, 4 color variants and
category-specific attribute lists.

Index decomposition (all constants divide the space exactly)::

    n = (pair_index * MODELS_PER_PAIR + model_index) * VARIANTS_PER_MODEL + variant_index

    pair_index   = category_index * BRAND_COUNT + brand_index   (800 pairs)
    model_index  in [0, 31_250)
    variant_index in [0, 4)

Search walks a bounded candidate stream (ranked pairs, then models in index
order, then variants) and applies filters per candidate, so it never touches
the full space. Price and newest ordering are approximate: they order the
bounded matching window, mirroring how real search backends approximate deep
ordering. ``total_estimate`` is therefore a lower-bound estimate whenever the
scan budget is exhausted before the stream ends.
"""

from __future__ import annotations

import base64
import hashlib
import heapq
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

from app.commerce_provider.errors import ProviderRejectedError
from app.models.schemas.commerce import CatalogFilters, CatalogItem, CatalogPage

CommerceSort = Literal["relevance", "price_asc", "price_desc", "newest"]

_SORT_VALUES: frozenset[str] = frozenset({"relevance", "price_asc", "price_desc", "newest"})

# --- Stable address space ---------------------------------------------------
CATEGORY_COUNT = 25
BRAND_COUNT = 32
PAIR_COUNT = CATEGORY_COUNT * BRAND_COUNT  # 800
VARIANTS_PER_MODEL = 4
TOTAL_SKUS = 100_000_000
MODELS_PER_PAIR = TOTAL_SKUS // (PAIR_COUNT * VARIANTS_PER_MODEL)  # 31_250

assert PAIR_COUNT * MODELS_PER_PAIR * VARIANTS_PER_MODEL == TOTAL_SKUS

# --- Search budgets ----------------------------------------------------------
SCAN_LIMIT = 30_000  # candidate models evaluated per request, worst case
WINDOW_MIN = 1_024  # matches collected for an un-paged first request
MAX_WINDOW = 8_192  # deepest reachable offset + page size
MAX_PAGE_SIZE = 50

_CATALOG_SALT = "rag-chatbot-commerce-provider-v1"
_SKU_PATTERN = re.compile(r"SKU-(\d{8})")
_TOKEN_PATTERN = re.compile(r"[0-9a-z\u4e00-\u9fff]+")


@dataclass(frozen=True)
class CategorySpec:
    """A product category with its search aliases and attribute vocabulary."""

    category_id: str
    name: str
    aliases: tuple[str, ...]
    base_price_minor: int
    blurb: str
    attrs: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class BrandSpec:
    """A virtual brand; the multiplier anchors its price tier."""

    name: str
    slug: str
    multiplier: float


CATEGORIES: tuple[CategorySpec, ...] = (
    CategorySpec(
        "monitor",
        "显示器",
        ("monitor", "display", "显示屏", "屏幕"),
        129_900,
        "适配办公与设计场景",
        (
            ("尺寸", ("24英寸", "27英寸", "32英寸", "34英寸")),
            ("分辨率", ("1080P", "2K", "4K")),
            ("面板", ("IPS", "VA", "TN")),
            ("刷新率", ("60Hz", "144Hz", "165Hz")),
            ("接口", ("HDMI", "DP", "Type-C")),
        ),
    ),
    CategorySpec(
        "keyboard",
        "机械键盘",
        ("keyboard", "键盘"),
        39_900,
        "适合长时间文字输入",
        (
            ("配列", ("68键", "87键", "104键")),
            ("轴体", ("红轴", "茶轴", "青轴")),
            ("连接", ("有线", "2.4G无线", "蓝牙")),
            ("背光", ("无背光", "单色背光", "RGB背光")),
        ),
    ),
    CategorySpec(
        "mouse",
        "鼠标",
        ("mouse", "鼠标"),
        19_900,
        "高精度办公传感器",
        (
            ("连接", ("有线", "2.4G无线", "蓝牙")),
            ("传感器", ("光学", "激光")),
            ("手型", ("对称", "人体工学")),
            ("供电", ("干电池", "USB充电")),
        ),
    ),
    CategorySpec(
        "printer",
        "打印机",
        ("printer", "打印机", "一体机"),
        199_900,
        "满足部门级打印负荷",
        (
            ("技术", ("激光", "喷墨", "针式")),
            ("幅面", ("A4", "A3")),
            ("双面", ("自动双面", "手动双面")),
            ("色彩", ("黑白", "彩色")),
        ),
    ),
    CategorySpec(
        "toner",
        "打印耗材",
        ("toner", "cartridge", "硒鼓", "粉盒", "墨盒", "耗材"),
        39_900,
        "与主流打印机型适配",
        (
            ("类型", ("硒鼓", "粉盒", "墨盒")),
            ("颜色", ("黑色", "三色")),
            ("容量", ("标准容量", "大容量")),
        ),
    ),
    CategorySpec(
        "scanner",
        "扫描仪",
        ("scanner", "扫描仪"),
        169_900,
        "适合档案数字化",
        (
            ("幅面", ("A4", "A3")),
            ("类型", ("平板", "馈纸")),
            ("速度", ("25页每分钟", "40页每分钟")),
        ),
    ),
    CategorySpec(
        "projector",
        "投影仪",
        ("projector", "投影仪", "投影机"),
        299_900,
        "适配会议室投屏",
        (
            ("亮度", ("3300流明", "4000流明", "5000流明")),
            ("分辨率", ("1080P", "4K")),
            ("光源", ("灯泡", "激光", "LED")),
        ),
    ),
    CategorySpec(
        "headphone",
        "耳机",
        ("headphone", "headset", "耳机", "耳麦"),
        29_900,
        "适合会议通话与专注工作",
        (
            ("类型", ("头戴式", "入耳式")),
            ("降噪", ("主动降噪", "无降噪")),
            ("连接", ("有线", "蓝牙")),
        ),
    ),
    CategorySpec(
        "speaker",
        "音箱",
        ("speaker", "音箱", "扬声器"),
        59_900,
        "适配会议室扩音",
        (
            ("声道", ("2.0声道", "2.1声道", "5.1声道")),
            ("连接", ("有线", "蓝牙")),
        ),
    ),
    CategorySpec(
        "microphone",
        "会议麦克风",
        ("microphone", "mic", "麦克风", "全向麦"),
        89_900,
        "覆盖中小型会议室",
        (
            ("拾音", ("全向", "心形指向")),
            ("连接", ("USB", "无线")),
            ("场景", ("会议", "直播")),
        ),
    ),
    CategorySpec(
        "webcam",
        "摄像头",
        ("webcam", "camera", "摄像头", "网络摄像机"),
        49_900,
        "适合视频会议与直播",
        (
            ("分辨率", ("1080P", "4K")),
            ("视角", ("90度", "120度")),
            ("对焦", ("自动对焦", "定焦")),
        ),
    ),
    CategorySpec(
        "dock",
        "扩展坞",
        ("dock", "hub", "扩展坞", "拓展坞", "坞站"),
        34_900,
        "一键扩展桌面接口",
        (
            ("接口", ("8合1", "11合1", "13合1")),
            ("供电", ("65W", "100W")),
            ("连接", ("USB-C", "雷电")),
        ),
    ),
    CategorySpec(
        "cable",
        "数据线",
        ("cable", "数据线", "转接头", "转接线"),
        4_900,
        "稳定传输与快充",
        (
            ("接口", ("USB-C", "Lightning", "Micro-USB")),
            ("长度", ("1米", "2米", "3米")),
            ("功能", ("充电", "数据传输", "视频传输")),
        ),
    ),
    CategorySpec(
        "ssd",
        "固态硬盘",
        ("ssd", "固态硬盘", "固态"),
        49_900,
        "高速读写与掉电保护",
        (
            ("容量", ("512GB", "1TB", "2TB", "4TB")),
            ("接口", ("NVMe", "SATA")),
            ("规格", ("2242", "2280")),
        ),
    ),
    CategorySpec(
        "ram",
        "内存条",
        ("ram", "memory", "内存"),
        39_900,
        "兼容主流办公主机",
        (
            ("容量", ("8GB", "16GB", "32GB")),
            ("规格", ("DDR4", "DDR5")),
            ("频率", ("3200MHz", "5600MHz")),
        ),
    ),
    CategorySpec(
        "laptop",
        "笔记本电脑",
        ("laptop", "notebook", "笔记本", "笔记本电脑"),
        599_900,
        "面向商务办公与出差",
        (
            ("处理器", ("i5", "i7", "i9")),
            ("内存", ("16GB", "32GB", "64GB")),
            ("存储", ("512GB", "1TB", "2TB")),
            ("屏幕", ("14英寸", "15.6英寸", "16英寸")),
        ),
    ),
    CategorySpec(
        "desktop",
        "台式主机",
        ("desktop", "pc", "台式机", "主机"),
        449_900,
        "适合固定工位高强度办公",
        (
            ("处理器", ("i5", "i7", "i9")),
            ("内存", ("16GB", "32GB")),
            ("存储", ("1TB", "2TB")),
            ("显卡", ("核显", "独显")),
        ),
    ),
    CategorySpec(
        "ups",
        "UPS电源",
        ("ups", "不间断电源", "UPS"),
        89_900,
        "断电保护办公设备",
        (
            ("容量", ("600VA", "1000VA", "1500VA")),
            ("电池", ("铅酸", "锂电")),
            ("输出", ("6口", "8口")),
        ),
    ),
    CategorySpec(
        "shredder",
        "碎纸机",
        ("shredder", "碎纸机"),
        129_900,
        "保密销毁纸质文件",
        (
            ("等级", ("4级", "5级", "6级")),
            ("方式", ("条状", "段状", "粉末状")),
            ("容量", ("15升", "20升", "30升")),
        ),
    ),
    CategorySpec(
        "chair",
        "人体工学椅",
        ("chair", "办公椅", "座椅", "人体工学椅", "椅子"),
        159_900,
        "久坐支撑与透气舒适",
        (
            ("材质", ("网布", "皮质")),
            ("腰托", ("可调腰托", "标准腰托")),
            ("头枕", ("带头枕", "无头枕")),
            ("扶手", ("固定扶手", "升降扶手")),
        ),
    ),
    CategorySpec(
        "desk",
        "升降桌",
        ("desk", "table", "办公桌", "升降桌", "桌子"),
        189_900,
        "坐站交替健康办公",
        (
            ("桌面", ("1.2米", "1.4米", "1.6米")),
            ("升降", ("单电机", "双电机", "固定")),
            ("桌板", ("岩板", "实木板", "环保板")),
        ),
    ),
    CategorySpec(
        "cabinet",
        "文件柜",
        ("cabinet", "文件柜", "档案柜", "柜子"),
        79_900,
        "安全存放档案资料",
        (
            ("层数", ("三层", "四层", "五层")),
            ("材质", ("钢制", "木质")),
            ("门型", ("玻璃门", "铁门")),
        ),
    ),
    CategorySpec(
        "whiteboard",
        "白板",
        ("whiteboard", "白板", "书写板"),
        49_900,
        "会议室书写与展示",
        (
            ("尺寸", ("90乘120", "120乘240")),
            ("材质", ("磁性白板", "玻璃白板")),
            ("支架", ("移动支架", "壁挂")),
        ),
    ),
    CategorySpec(
        "binder",
        "装订机",
        ("binder", "装订机", "订书机", "起钉器"),
        12_900,
        "规范装订纸质文档",
        (
            ("类型", ("热熔装订机", "梳式装订机", "厚层订书机")),
            ("装订厚度", ("12毫米", "25毫米", "40毫米")),
        ),
    ),
    CategorySpec(
        "switch",
        "网络交换机",
        ("switch", "交换机", "路由器", "网络设备", "网关"),
        59_900,
        "稳定组网与内网互联",
        (
            ("端口", ("5口", "8口", "16口", "24口")),
            ("速率", ("百兆", "千兆", "2.5G")),
            ("管理", ("非网管", "网管型")),
        ),
    ),
)

BRANDS: tuple[BrandSpec, ...] = (
    BrandSpec("恒锐", "hengrui", 0.78),
    BrandSpec("途胜", "tusheng", 0.82),
    BrandSpec("星澜", "xinglan", 0.85),
    BrandSpec("森格", "senge", 0.88),
    BrandSpec("迅拓", "xuntuo", 0.90),
    BrandSpec("明捷", "mingjie", 0.92),
    BrandSpec("蓝驰", "lanchi", 0.95),
    BrandSpec("越翼", "yueyi", 0.98),
    BrandSpec("拓普", "topu", 1.00),
    BrandSpec("灵动", "lingdong", 1.05),
    BrandSpec("岚图", "lantu", 1.10),
    BrandSpec("澎湃", "pengpai", 1.12),
    BrandSpec("智蓝", "zhilan", 1.15),
    BrandSpec("元启", "yuanqi", 1.20),
    BrandSpec("峰扬", "fengyang", 1.25),
    BrandSpec("沃德", "wode", 1.28),
    BrandSpec("天工", "tiangong", 1.32),
    BrandSpec("卓朗", "zhuolang", 1.38),
    BrandSpec("极目", "jimu", 1.42),
    BrandSpec("启明", "qiming", 1.48),
    BrandSpec("尚品居", "shangpinju", 1.52),
    BrandSpec("铂悦", "boyue", 1.56),
    BrandSpec("曜石", "yaoshi", 1.62),
    BrandSpec("迈恒", "maiheng", 1.68),
    BrandSpec("云枢", "yunshu", 1.74),
    BrandSpec("墨方", "mofang", 1.82),
    BrandSpec("光合", "guanghe", 1.90),
    BrandSpec("北辰", "beichen", 2.00),
    BrandSpec("睿视", "ruishi", 2.10),
    BrandSpec("朗曜", "langyao", 2.20),
    BrandSpec("坤灵", "kunling", 2.35),
    BrandSpec("寰宇", "huanyu", 2.50),
)

SERIES_WORDS: tuple[str, ...] = (
    "Astro",
    "Nova",
    "Prime",
    "Vertex",
    "Orbit",
    "Zenith",
    "Core",
    "Flux",
    "Aero",
    "Terra",
    "Pulse",
    "Spark",
    "Halo",
    "Fusion",
    "Echo",
    "Titan",
    "Nimbus",
    "Quanta",
    "Vela",
    "Lyra",
    "Onyx",
    "Atlas",
    "Iris",
    "Comet",
    "Vector",
    "Delta",
    "Astra",
    "Kilo",
    "Neo",
    "Omni",
    "Pixel",
    "Radian",
)

SERIES_COUNT = len(SERIES_WORDS)
VARIANT_COLORS: tuple[str, ...] = ("黑色", "白色", "银色", "灰色")

assert len(CATEGORIES) == CATEGORY_COUNT
assert len(BRANDS) == BRAND_COUNT
assert len(SERIES_WORDS) == SERIES_COUNT
assert len(VARIANT_COLORS) == VARIANTS_PER_MODEL


@dataclass(frozen=True)
class _TokenPlan:
    """How a tokenized query constrains the candidate stream."""

    pair_indexes: tuple[int, ...]
    series_ids: frozenset[int]
    text_terms: tuple[str, ...]


def _stable_hash(n: int, domain: str) -> int:
    digest = hashlib.blake2b(f"{_CATALOG_SALT}|{domain}|{n}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class VirtualCatalog:
    """Pure-function view over the fixed 100M-SKU address space."""

    total_skus: int = TOTAL_SKUS

    # --- addressing ---------------------------------------------------------
    def sku_id(self, n: int) -> str:
        return f"SKU-{n:08d}"

    def parse_sku_id(self, sku_id: str) -> int | None:
        match = _SKU_PATTERN.fullmatch(sku_id.strip().upper())
        if match is None:
            return None
        n = int(match.group(1))
        return n if n < TOTAL_SKUS else None

    def decompose(self, n: int) -> tuple[int, int, int]:
        """Return ``(pair_index, model_index, variant_index)`` for a valid n."""

        pair_index, rest = divmod(n, MODELS_PER_PAIR * VARIANTS_PER_MODEL)
        model_index, variant_index = divmod(rest, VARIANTS_PER_MODEL)
        return pair_index, model_index, variant_index

    def product_id(self, pair_index: int, model_index: int) -> str:
        return f"PRD-{pair_index:03d}-{model_index:05d}"

    # --- fact projection ----------------------------------------------------
    def _unit_price(self, pair_index: int, model_index: int) -> int:
        category = CATEGORIES[pair_index // BRAND_COUNT]
        brand = BRANDS[pair_index % BRAND_COUNT]
        series_factor = 0.85 + 0.35 * ((model_index % SERIES_COUNT) % 8) / 7
        band_factor = 1 + 0.01 * (model_index % 25)
        price = category.base_price_minor * brand.multiplier * series_factor * band_factor
        return max(100, int(round(price / 100.0)) * 100)

    def _base_availability(self, n: int) -> int:
        digest = _stable_hash(n, "stock")
        return 0 if digest % 41 == 0 else 5 + digest % 60

    def _max_order_quantity(self, n: int) -> int:
        return 1 + _stable_hash(n, "cap") % 10

    def _describe(
        self, pair_index: int, model_index: int, variant_index: int
    ) -> tuple[str, str, dict[str, str]]:
        category = CATEGORIES[pair_index // BRAND_COUNT]
        brand = BRANDS[pair_index % BRAND_COUNT]
        series = SERIES_WORDS[model_index % SERIES_COUNT]
        code = f"{series[0]}{model_index:04d}"
        name = f"{brand.name} {series} {code} {category.name}"
        attributes = {
            "品牌": brand.name,
            "系列": series,
            "颜色": VARIANT_COLORS[variant_index],
        }
        attr_text: list[str] = []
        for ordinal, (key, vocab) in enumerate(category.attrs):
            value = vocab[(model_index + 3 * ordinal) % len(vocab)]
            attributes[key] = value
            attr_text.append(f"{key}{value}")
        description = (
            f"模拟商品：{brand.name}{series}系列{category.name}"
            f"（{'，'.join(attr_text[:4])}），{category.blurb}；用于演示购物流程。"
        )
        return name, description, attributes

    def materialize(self, n: int, stock_delta: int = 0) -> CatalogItem:
        """Project the full catalog item for a SKU index."""

        if not 0 <= n < TOTAL_SKUS:
            raise ProviderRejectedError("模拟商品编号越界")
        pair_index, model_index, variant_index = self.decompose(n)
        name, description, attributes = self._describe(pair_index, model_index, variant_index)
        available = max(0, self._base_availability(n) + stock_delta)
        return CatalogItem(
            product_id=self.product_id(pair_index, model_index),
            sku_id=self.sku_id(n),
            name=name,
            description=description,
            attributes=attributes,
            unit_price_minor=self._unit_price(pair_index, model_index),
            available_quantity=available,
            max_order_quantity=self._max_order_quantity(n),
        )

    # --- search -------------------------------------------------------------
    def search(
        self,
        *,
        query: str,
        filters: CatalogFilters | None = None,
        sort: CommerceSort = "relevance",
        page_size: int = 10,
        page_token: str | None = None,
        stock_deltas: Mapping[str, int] | None = None,
    ) -> CatalogPage:
        """Run a bounded token/filters/sort search over the virtual space."""

        return _search(
            self,
            query=query,
            filters=filters or CatalogFilters(),
            sort=sort,
            page_size=page_size,
            page_token=page_token,
            stock_deltas=stock_deltas or {},
        )

    # --- token helpers ------------------------------------------------------
    def _tokenize(self, query: str) -> list[str]:
        return _TOKEN_PATTERN.findall(query.lower())

    def _token_matches(self, token: str, candidates: tuple[str, ...]) -> bool:
        for raw in candidates:
            candidate = raw.lower()
            if token == candidate:
                return True
            if len(token) >= 2 and (token in candidate or candidate in token):
                return True
        return False

    def _category_pairs(self, category_index: int) -> range:
        start = category_index * BRAND_COUNT
        return range(start, start + BRAND_COUNT)

    def _plan_tokens(self, tokens: list[str]) -> _TokenPlan:
        pair_sets: list[frozenset[int]] = []
        series_ids: set[int] = set()
        text_terms: list[str] = []
        for token in tokens:
            pair_match: set[int] = set()
            for category_index, category in enumerate(CATEGORIES):
                if self._token_matches(token, (category.name, *category.aliases)):
                    pair_match.update(self._category_pairs(category_index))
            for brand_index, brand in enumerate(BRANDS):
                if self._token_matches(token, (brand.name, brand.slug)):
                    pair_match.update(
                        category_index * BRAND_COUNT + brand_index
                        for category_index in range(CATEGORY_COUNT)
                    )
            if pair_match:
                pair_sets.append(frozenset(pair_match))
                continue
            matched_series = {
                index
                for index, word in enumerate(SERIES_WORDS)
                if self._token_matches(token, (word,))
            }
            if matched_series:
                series_ids.update(matched_series)
            else:
                text_terms.append(token)
        pairs = frozenset.intersection(*pair_sets) if pair_sets else frozenset(range(PAIR_COUNT))
        return _TokenPlan(tuple(sorted(pairs)), frozenset(series_ids), tuple(text_terms))

    def _blob(self, pair_index: int, model_index: int, variant_index: int) -> str:
        category = CATEGORIES[pair_index // BRAND_COUNT]
        brand = BRANDS[pair_index % BRAND_COUNT]
        name, description, attributes = self._describe(pair_index, model_index, variant_index)
        parts = [
            name,
            description,
            brand.name,
            brand.slug,
            category.name,
            *category.aliases,
            *attributes.keys(),
            *attributes.values(),
        ]
        return " ".join(parts).lower()


def _encode_page_token(digest: str, offset: int) -> str:
    raw = json.dumps({"d": digest, "o": offset}, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_page_token(token: str) -> tuple[str, int]:
    padded = token + "=" * (-len(token) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()))
        return str(payload["d"]), int(payload["o"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ProviderRejectedError("翻页令牌无效") from exc


def _search(
    catalog: VirtualCatalog,
    *,
    query: str,
    filters: CatalogFilters,
    sort: CommerceSort,
    page_size: int,
    page_token: str | None,
    stock_deltas: Mapping[str, int],
) -> CatalogPage:
    if sort not in _SORT_VALUES:
        raise ProviderRejectedError("不支持的排序方式")
    page_size = max(1, min(page_size, MAX_PAGE_SIZE))
    tokens = catalog._tokenize(query)
    plan = catalog._plan_tokens(tokens)

    context = {
        "query": query,
        "filters": filters.model_dump(mode="json"),
        "sort": sort,
        "page_size": page_size,
    }
    digest = hashlib.sha256(
        json.dumps(context, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()

    offset = 0
    if page_token is not None:
        token_digest, offset = _decode_page_token(page_token)
        if token_digest != digest:
            raise ProviderRejectedError("翻页令牌与当前查询不匹配")
        if offset < 0 or offset + page_size > MAX_WINDOW:
            raise ProviderRejectedError("翻页过深，请缩小查询范围")
    needed = min(MAX_WINDOW, max(WINDOW_MIN, offset + 2 * page_size))

    price_min = filters.min_price_minor
    price_max = filters.max_price_minor
    attr_filters = list(filters.attributes.items())
    brand_filter = filters.attributes.get("品牌")
    series_filter = filters.attributes.get("系列")
    color_filter = filters.attributes.get("颜色")

    matches: list[tuple[int, int, int]] = []  # (n, price, effective availability)
    scanned = 0
    budget_exhausted = False

    for pair_index in plan.pair_indexes:
        category = CATEGORIES[pair_index // BRAND_COUNT]
        brand = BRANDS[pair_index % BRAND_COUNT]
        if brand_filter is not None and brand.name not in brand_filter:
            continue
        # Resolve category-attribute filters once per pair; an unknown key
        # makes the whole pair unable to satisfy the request.
        cat_checks: list[tuple[int, tuple[str, ...], frozenset[str]]] = []
        pair_possible = True
        for key, allowed in attr_filters:
            if key in {"品牌", "系列", "颜色"}:
                continue
            ordinal = next(
                (i for i, (attr_key, _) in enumerate(category.attrs) if attr_key == key), None
            )
            if ordinal is None:
                pair_possible = False
                break
            cat_checks.append((ordinal, category.attrs[ordinal][1], frozenset(allowed)))
        if not pair_possible:
            continue

        series_residues: frozenset[int] = plan.series_ids
        if series_filter is not None:
            filter_residues = frozenset(
                index for index, word in enumerate(SERIES_WORDS) if word in series_filter
            )
            series_residues = (
                filter_residues if not series_residues else series_residues & filter_residues
            )
            if not series_residues:
                continue
        models: Iterator[int]
        if series_residues:
            models = heapq.merge(
                *(range(r, MODELS_PER_PAIR, SERIES_COUNT) for r in sorted(series_residues))
            )
        else:
            models = iter(range(MODELS_PER_PAIR))

        for model_index in models:
            scanned += 1
            if scanned > SCAN_LIMIT:
                budget_exhausted = True
                break
            if any(
                vocab[(model_index + 3 * ordinal) % len(vocab)] not in allowed
                for ordinal, vocab, allowed in cat_checks
            ):
                continue
            price = catalog._unit_price(pair_index, model_index)
            if price_min is not None and price < price_min:
                continue
            if price_max is not None and price > price_max:
                continue
            for variant_index in range(VARIANTS_PER_MODEL):
                if color_filter is not None and VARIANT_COLORS[variant_index] not in color_filter:
                    continue
                n = (pair_index * MODELS_PER_PAIR + model_index) * VARIANTS_PER_MODEL
                n += variant_index
                sku = catalog.sku_id(n)
                available = catalog._base_availability(n) + stock_deltas.get(sku, 0)
                if filters.in_stock and available <= 0:
                    continue
                if plan.text_terms:
                    blob = catalog._blob(pair_index, model_index, variant_index)
                    if any(term not in blob for term in plan.text_terms):
                        continue
                matches.append((n, price, stock_deltas.get(sku, 0)))
            if len(matches) >= needed:
                break
        if len(matches) >= needed or budget_exhausted:
            break
    # A model expands into four variant SKUs, so the inner loop can pass the
    # requested window by up to three entries. Clip before ordering/paging to
    # keep the bounded-search contract exact.
    matches = matches[:needed]

    if sort == "price_asc":
        matches.sort(key=lambda match: (match[1], match[0]))
    elif sort == "price_desc":
        matches.sort(key=lambda match: (-match[1], match[0]))
    elif sort == "newest":
        matches.sort(key=lambda match: -match[0])

    page = matches[offset : offset + page_size]
    items = [catalog.materialize(n, delta) for n, _price, delta in page]
    next_token = (
        _encode_page_token(digest, offset + page_size)
        if offset + page_size < len(matches)
        else None
    )
    return CatalogPage(
        items=items,
        next_page_token=next_token,
        total_estimate=len(matches),
    )
