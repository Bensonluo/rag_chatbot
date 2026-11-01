"""Deterministic, scalable virtual catalog for the commerce provider.

The catalog is never enumerated or seeded into a database: every fact
(name, description, attributes, price, availability, purchase cap) is a pure
function of a stable numeric SKU index in ``[0, TOTAL_SKUS)``. The default
taxonomy projects ``CATEGORY_COUNT`` categories across 18 shopping domains
(office electronics plus 17 daily-life domains), each combined with 32
virtual brands, ``MODELS_PER_PAIR`` product models and a per-category variant
axis of exactly ``VARIANTS_PER_MODEL`` values (colors for electronics, sizes
for apparel, portions for fresh food, and so on).

Index decomposition (all constants divide the space exactly)::

    n = (pair_index * MODELS_PER_PAIR + model_index) * VARIANTS_PER_MODEL + variant_index

    pair_index   = category_index * BRAND_COUNT + brand_index
    model_index  in [0, MODELS_PER_PAIR)
    variant_index in [0, VARIANTS_PER_MODEL)

Every category therefore carries a small, deliberately shallow product set
(32 brands × 8 models × 4 variants = 1,024 SKUs per category) while the
category coverage spans a full general-marketplace taxonomy.

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
from app.models.schemas.commerce import (
    CatalogCategory,
    CatalogFilters,
    CatalogItem,
    CatalogPage,
    CatalogTaxonomy,
    CategoryAttribute,
)
from app.models.schemas.commerce_taxonomy import RegistryCategory, RegistryTaxonomy

CommerceSort = Literal["relevance", "price_asc", "price_desc", "newest"]

_SORT_VALUES: frozenset[str] = frozenset({"relevance", "price_asc", "price_desc", "newest"})

# --- Stable address space ---------------------------------------------------
CATEGORY_COUNT = 151
BRAND_COUNT = 32
PAIR_COUNT = CATEGORY_COUNT * BRAND_COUNT  # 4_832
VARIANTS_PER_MODEL = 4
MODELS_PER_PAIR = 8
TOTAL_SKUS = PAIR_COUNT * MODELS_PER_PAIR * VARIANTS_PER_MODEL  # 154_624

DEFAULT_VARIANT_VALUES: tuple[str, ...] = ("黑色", "白色", "银色", "灰色")

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
    """A product category with its search aliases and attribute vocabulary.

    ``variant_key``/``variant_values`` declare the category's variant axis —
    the fourth SKU address dimension. Electronics default to colors; apparel
    uses sizes, fresh food uses portions. Generated categories must declare
    exactly ``VARIANTS_PER_MODEL`` variant values so the address decomposition
    stays uniform across domains.
    """

    category_id: str
    name: str
    aliases: tuple[str, ...]
    base_price_minor: int
    blurb: str
    attrs: tuple[tuple[str, tuple[str, ...]], ...]
    domain: str = "office_electronics"
    variant_key: str = "颜色"
    variant_values: tuple[str, ...] = DEFAULT_VARIANT_VALUES


@dataclass(frozen=True)
class BrandSpec:
    """A virtual brand; the multiplier anchors its price tier."""

    name: str
    slug: str
    multiplier: float


CATEGORIES: tuple[CategorySpec, ...] = (
    # --- domain: office_electronics 办公电子 (25) ----------------------------
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
            # "色彩" not "颜色": the variant axis already owns 颜色 addressing.
            ("色彩", ("黑色", "三色")),
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
    # --- domain: food 食品生鲜 (10) ------------------------------------------
    CategorySpec(
        "fresh_produce",
        "生鲜果蔬",
        ("fresh", "vegetable", "fruit", "果蔬", "蔬菜", "水果", "生鲜"),
        1_990,
        "产地直采冷链直达",
        (
            ("产地", ("山东", "云南", "海南", "进口")),
            ("等级", ("一级", "特级")),
            ("保存", ("冷藏", "常温")),
        ),
        domain="food",
        variant_key="规格",
        variant_values=("250g", "500g", "1kg", "2kg"),
    ),
    CategorySpec(
        "seafood",
        "海鲜水产",
        ("seafood", "水产", "海鲜", "虾", "鱼", "贝类"),
        4_990,
        "鲜活与冷冻水产",
        (
            ("产地", ("国产", "进口")),
            ("处理", ("鲜活", "冷冻", "干货")),
        ),
        domain="food",
        variant_key="规格",
        variant_values=("250g", "500g", "1kg", "2kg"),
    ),
    CategorySpec(
        "meat_eggs",
        "肉类禽蛋",
        ("meat", "egg", "肉", "禽蛋", "鸡蛋", "牛肉", "猪肉"),
        2_990,
        "冷鲜肉蛋每日达",
        (
            ("类型", ("猪肉", "牛肉", "禽肉", "蛋品")),
            ("处理", ("冷鲜", "冷冻")),
        ),
        domain="food",
        variant_key="规格",
        variant_values=("500g", "1kg", "2kg", "5kg"),
    ),
    CategorySpec(
        "grains",
        "粮油米面",
        ("grain", "rice", "oil", "粮油", "米面", "大米", "面粉", "食用油"),
        5_900,
        "米面粮油基础供给",
        (
            ("类型", ("大米", "面粉", "食用油", "杂粮")),
            ("产地", ("东北", "华北", "进口")),
        ),
        domain="food",
        variant_key="规格",
        variant_values=("1kg", "5kg", "10kg", "礼盒装"),
    ),
    CategorySpec(
        "dairy_bakery",
        "乳品烘焙",
        ("dairy", "milk", "bakery", "乳品", "牛奶", "烘焙", "面包", "蛋糕"),
        1_590,
        "低温乳品与烘焙",
        (
            ("类型", ("鲜奶", "酸奶", "面包", "甜点")),
            ("保存", ("冷藏", "常温")),
        ),
        domain="food",
        variant_key="规格",
        variant_values=("250ml", "500ml", "1L", "家庭装"),
    ),
    CategorySpec(
        "snacks",
        "零食坚果",
        ("snack", "nut", "零食", "坚果", "薯片", "瓜子"),
        3_900,
        "追剧休闲零嘴",
        (("类型", ("坚果炒货", "膨化", "糖果", "肉干")),),
        domain="food",
        variant_key="口味",
        variant_values=("原味", "海盐", "焦糖", "混合"),
    ),
    CategorySpec(
        "beverages",
        "饮品冲调",
        ("drink", "beverage", "饮料", "饮品", "果汁", "汽水", "冲调"),
        1_290,
        "解渴冲调随取随饮",
        (
            ("类型", ("果汁", "汽水", "茶饮", "咖啡冲调")),
            ("温感", ("常温", "冷藏")),
        ),
        domain="food",
        variant_key="规格",
        variant_values=("330ml", "500ml", "1L", "整箱"),
    ),
    CategorySpec(
        "alcohol",
        "酒水",
        ("alcohol", "wine", "beer", "酒", "白酒", "红酒", "啤酒"),
        12_900,
        "聚会宴请用酒",
        (
            ("类型", ("白酒", "红酒", "啤酒", "洋酒")),
            ("产地", ("国产", "进口")),
        ),
        domain="food",
        variant_key="规格",
        variant_values=("单瓶", "双支装", "礼盒装", "整箱"),
    ),
    CategorySpec(
        "tea_coffee",
        "茶叶咖啡",
        ("tea", "coffee", "茶", "茶叶", "咖啡", "咖啡豆", "咖啡粉"),
        8_900,
        "一口回甘一杯提神",
        (
            ("类型", ("绿茶", "红茶", "乌龙", "咖啡豆")),
            ("产地", ("云南", "福建", "进口")),
        ),
        domain="food",
        variant_key="规格",
        variant_values=("50g", "100g", "250g", "500g"),
    ),
    CategorySpec(
        "condiments",
        "调味速食",
        ("condiment", "sauce", "调味", "调料", "酱油", "速食", "方便面"),
        1_090,
        "厨房调味与速食",
        (("类型", ("酱油醋", "香辛料", "酱料", "速食")),),
        domain="food",
        variant_key="规格",
        variant_values=("100g", "500g", "1kg", "组合装"),
    ),
    # --- domain: apparel 服饰内衣 (9) ----------------------------------------
    CategorySpec(
        "tshirt",
        "上衣T恤",
        ("tshirt", "t-shirt", "shirt", "上衣", "T恤", "衬衫"),
        12_900,
        "基础百搭日常上衣",
        (
            ("材质", ("纯棉", "莫代尔", "棉麻")),
            ("袖型", ("短袖", "长袖", "无袖")),
            ("版型", ("修身", "宽松", "常规")),
        ),
        domain="apparel",
        variant_key="尺码",
        variant_values=("S", "M", "L", "XL"),
    ),
    CategorySpec(
        "outerwear",
        "外套夹克",
        ("jacket", "coat", "外套", "夹克", "风衣", "大衣"),
        39_900,
        "换季保暖有型",
        (
            ("材质", ("棉", "聚酯纤维", "羊毛")),
            ("厚度", ("单层", "加厚", "夹棉")),
        ),
        domain="apparel",
        variant_key="尺码",
        variant_values=("S", "M", "L", "XL"),
    ),
    CategorySpec(
        "pants",
        "裤装",
        ("pants", "trousers", "jeans", "裤", "长裤", "牛仔裤", "休闲裤"),
        19_900,
        "合身利落的下半身",
        (
            ("类型", ("牛仔裤", "休闲裤", "西裤", "卫裤")),
            ("材质", ("牛仔", "棉", "西装呢")),
        ),
        domain="apparel",
        variant_key="尺码",
        variant_values=("28", "30", "32", "34"),
    ),
    CategorySpec(
        "dress",
        "连衣裙",
        ("dress", "连衣裙", "裙子", "长裙", "半身裙"),
        25_900,
        "一裙搞定出门造型",
        (
            ("版型", ("A字", "直筒", "蓬蓬", "包臀")),
            ("裙长", ("短裙", "中裙", "长裙")),
        ),
        domain="apparel",
        variant_key="尺码",
        variant_values=("S", "M", "L", "XL"),
    ),
    CategorySpec(
        "suit",
        "西装套装",
        ("suit", "blazer", "西装", "正装", "职业装"),
        59_900,
        "通勤正式场合首选",
        (
            ("场合", ("商务", "休闲", "礼服")),
            ("件数", ("单西", "两件套", "三件套")),
        ),
        domain="apparel",
        variant_key="尺码",
        variant_values=("48", "50", "52", "54"),
    ),
    CategorySpec(
        "underwear",
        "内衣袜品",
        ("underwear", "socks", "内衣", "内裤", "袜子", "文胸"),
        4_900,
        "贴身舒适基础款",
        (
            ("类型", ("内衣", "内裤", "袜子", "文胸")),
            ("材质", ("纯棉", "莫代尔", "冰丝")),
        ),
        domain="apparel",
        variant_key="尺码",
        variant_values=("S", "M", "L", "XL"),
    ),
    CategorySpec(
        "sleepwear",
        "家居服",
        ("pajamas", "loungewear", "家居服", "睡衣", "睡袍"),
        9_900,
        "居家松弛感穿搭",
        (
            ("类型", ("睡衣套装", "睡袍", "家居套装")),
            ("季节", ("春夏", "秋冬")),
        ),
        domain="apparel",
        variant_key="尺码",
        variant_values=("S", "M", "L", "XL"),
    ),
    CategorySpec(
        "down_jacket",
        "羽绒服",
        ("down", "downjacket", "羽绒服", "羽绒", "棉服"),
        49_900,
        "寒冬锁温主力",
        (
            ("含绒量", ("70%", "80%", "90%")),
            ("蓬松度", ("600", "700", "800")),
        ),
        domain="apparel",
        variant_key="尺码",
        variant_values=("S", "M", "L", "XL"),
    ),
    CategorySpec(
        "kids_clothing",
        "童装",
        ("kids", "children", "童装", "儿童", "童上衣", "童裤"),
        7_900,
        "萌娃出行舒适装备",
        (
            ("类型", ("上衣", "裤装", "套装", "裙装")),
            ("材质", ("纯棉", "针织")),
        ),
        domain="apparel",
        variant_key="尺码",
        variant_values=("100cm", "120cm", "140cm", "160cm"),
    ),
    # --- domain: shoes_bags 鞋靴箱包 (6) -------------------------------------
    CategorySpec(
        "sneakers",
        "休闲运动鞋",
        ("sneakers", "shoes", "休闲鞋", "运动鞋", "板鞋"),
        39_900,
        "日常通勤百搭鞋",
        (
            ("鞋面", ("网面", "皮革", "帆布")),
            ("功能", ("透气", "防滑", "缓震")),
        ),
        domain="shoes_bags",
        variant_key="鞋码",
        variant_values=("39", "40", "41", "42"),
    ),
    CategorySpec(
        "leather_shoes",
        "皮鞋",
        ("leathershoes", "皮鞋", "正装鞋", "商务鞋"),
        49_900,
        "正式场合稳重之选",
        (
            ("材质", ("牛皮", "羊皮", "漆皮")),
            ("款式", ("系带", "一脚蹬")),
        ),
        domain="shoes_bags",
        variant_key="鞋码",
        variant_values=("39", "40", "41", "42"),
    ),
    CategorySpec(
        "boots",
        "靴子",
        ("boots", "靴子", "马丁靴", "雪地靴", "短靴"),
        45_900,
        "秋冬造型利器",
        (
            ("类型", ("马丁靴", "雪地靴", "切尔西靴")),
            ("筒高", ("低筒", "中筒", "高筒")),
        ),
        domain="shoes_bags",
        variant_key="鞋码",
        variant_values=("39", "40", "41", "42"),
    ),
    CategorySpec(
        "sandals",
        "凉鞋拖鞋",
        ("sandals", "slippers", "凉鞋", "拖鞋", "人字拖"),
        9_900,
        "清凉宅家两相宜",
        (
            ("类型", ("凉鞋", "拖鞋", "人字拖")),
            ("材质", ("EVA", "橡胶", "皮革")),
        ),
        domain="shoes_bags",
        variant_key="鞋码",
        variant_values=("38", "39", "40", "41"),
    ),
    CategorySpec(
        "bags",
        "箱包",
        ("bag", "handbag", "backpack", "箱包", "背包", "双肩包", "手提包", "单肩包"),
        29_900,
        "通勤出行装得下",
        (
            ("类型", ("双肩包", "单肩包", "手提包", "斜挎包")),
            ("材质", ("尼龙", "皮革", "帆布")),
        ),
        domain="shoes_bags",
    ),
    CategorySpec(
        "luggage",
        "行李箱",
        ("luggage", "suitcase", "行李箱", "拉杆箱", "旅行箱"),
        49_900,
        "出差旅行可靠伙伴",
        (
            ("材质", ("铝框", "PC", "布面")),
            ("轮子", ("万向轮", "飞机轮")),
        ),
        domain="shoes_bags",
        variant_key="尺寸",
        variant_values=("20寸", "24寸", "26寸", "28寸"),
    ),
    # --- domain: accessories 配饰钟表 (6) ------------------------------------
    CategorySpec(
        "watches",
        "手表",
        ("watch", "手表", "腕表", "电子表", "石英表"),
        89_900,
        "腕间时间与品味",
        (
            ("类型", ("石英", "机械", "电子")),
            ("防水", ("30米", "50米", "100米")),
        ),
        domain="accessories",
        variant_key="表带",
        variant_values=("钢带", "皮带", "橡胶", "尼龙"),
    ),
    CategorySpec(
        "jewelry",
        "珠宝首饰",
        ("jewelry", "necklace", "bracelet", "首饰", "项链", "手链", "耳饰"),
        19_900,
        "点亮造型的细节",
        (
            ("类型", ("项链", "手链", "耳饰", "戒指")),
            ("风格", ("简约", "复古", "轻奢")),
        ),
        domain="accessories",
        variant_key="色泽",
        variant_values=("银色", "金色", "玫瑰金", "古铜"),
    ),
    CategorySpec(
        "glasses",
        "眼镜",
        ("glasses", "eyewear", "眼镜", "镜框", "光学镜"),
        25_900,
        "清晰视界从镜开始",
        (
            ("类型", ("光学镜", "防蓝光", "老花")),
            ("材质", ("TR90", "钛架", "板材")),
        ),
        domain="accessories",
        variant_key="镜框",
        variant_values=("全框", "半框", "无框", "圆框"),
    ),
    CategorySpec(
        "sunglasses",
        "太阳镜",
        ("sunglasses", "太阳镜", "墨镜", "遮阳镜"),
        19_900,
        "烈日下的型格防线",
        (
            ("镜片", ("偏光", "变色", "渐变")),
            ("防晒", ("UV400", "UV380")),
        ),
        domain="accessories",
        variant_key="款式",
        variant_values=("飞行员", "方框", "猫眼", "运动"),
    ),
    CategorySpec(
        "hats_scarves",
        "帽子围巾",
        ("hat", "cap", "scarf", "帽子", "围巾", "手套", "针织帽"),
        8_900,
        "保暖防点缀睛",
        (
            ("类型", ("棒球帽", "针织帽", "围巾", "手套")),
            ("材质", ("羊毛", "棉", "腈纶")),
        ),
        domain="accessories",
    ),
    CategorySpec(
        "belts_wallets",
        "皮带钱包",
        ("belt", "wallet", "皮带", "钱包", "卡包"),
        15_900,
        "低调实用随身件",
        (
            ("类型", ("皮带", "钱包", "卡包", "钥匙扣")),
            ("材质", ("头层牛皮", "二层皮", "编织")),
        ),
        domain="accessories",
    ),
    # --- domain: beauty 美妆个护 (8) -----------------------------------------
    CategorySpec(
        "skincare",
        "面部护肤",
        ("skincare", "cream", "护肤", "面霜", "精华", "水乳"),
        12_900,
        "日常护肤基础线",
        (
            ("类型", ("水乳", "面霜", "精华", "面膜")),
            ("肤质", ("干性", "油性", "混合", "敏感")),
        ),
        domain="beauty",
        variant_key="规格",
        variant_values=("30ml", "50ml", "100ml", "套装"),
    ),
    CategorySpec(
        "makeup",
        "彩妆",
        ("makeup", "lipstick", "foundation", "彩妆", "口红", "粉底", "眼影"),
        9_900,
        "妆出好气色",
        (
            ("类型", ("口红", "粉底", "眼影", "腮红")),
            ("质地", ("哑光", "滋润", "轻薄")),
        ),
        domain="beauty",
        variant_key="色号",
        variant_values=("自然色", "象牙白", "暖调", "冷调"),
    ),
    CategorySpec(
        "perfume",
        "香水",
        ("perfume", "fragrance", "香水", "香氛"),
        19_900,
        "气味即记忆",
        (
            ("香调", ("花香", "果香", "木质", "东方")),
            ("浓度", ("EDT", "EDP", "古龙")),
        ),
        domain="beauty",
        variant_key="规格",
        variant_values=("30ml", "50ml", "100ml", "礼盒"),
    ),
    CategorySpec(
        "haircare",
        "洗发护发",
        ("shampoo", "conditioner", "洗发水", "护发", "护发素", "发膜"),
        5_900,
        "洗护一套到位",
        (
            ("类型", ("洗发", "护发", "发膜", "精华")),
            ("功效", ("去屑", "控油", "修护", "柔顺")),
        ),
        domain="beauty",
        variant_key="规格",
        variant_values=("200ml", "400ml", "750ml", "替换装"),
    ),
    CategorySpec(
        "bodycare",
        "身体护理",
        ("bodywash", "lotion", "身体乳", "沐浴露", "香皂", "磨砂膏"),
        4_900,
        "全身清爽滋润",
        (
            ("类型", ("沐浴露", "身体乳", "香皂", "磨砂")),
            ("肤感", ("清爽", "滋润")),
        ),
        domain="beauty",
        variant_key="香型",
        variant_values=("茉莉", "玫瑰", "海洋", "无香"),
    ),
    CategorySpec(
        "oral_care",
        "口腔护理",
        ("toothpaste", "toothbrush", "口腔", "牙膏", "牙刷", "漱口水", "牙线"),
        1_990,
        "口气清新一整天",
        (
            ("类型", ("牙膏", "牙刷", "漱口水", "牙线")),
            ("功效", ("美白", "抗敏", "清新", "护龈")),
        ),
        domain="beauty",
        variant_key="规格",
        variant_values=("单支", "双支装", "家庭装", "旅行装"),
    ),
    CategorySpec(
        "mens_grooming",
        "剃须护理",
        ("shaver", "razor", "剃须", "刮胡刀", "剃须刀", "须后水"),
        19_900,
        "干净利落的清晨",
        (
            ("类型", ("剃须刀", "刀片", "须后", "鼻毛修剪")),
            ("供电", ("充电", "电池", "手动")),
        ),
        domain="beauty",
        variant_key="款式",
        variant_values=("往复式", "旋转式", "手动", "便携"),
    ),
    CategorySpec(
        "beauty_tools",
        "美容仪器",
        ("beautydevice", "美容仪", "洁面仪", "导入仪", "脱毛仪"),
        29_900,
        "居家美容新方式",
        (
            ("功能", ("清洁", "导入", "紧致", "脱毛")),
            ("供电", ("充电", "插电")),
        ),
        domain="beauty",
        variant_key="配置",
        variant_values=("标准版", "进阶版", "尊享版", "套装"),
    ),
    # --- domain: home_daily 家居日用 (9) -------------------------------------
    CategorySpec(
        "home_clean",
        "清洁用品",
        ("cleaning", "detergent", "清洁", "洗衣液", "洗洁精", "消毒"),
        2_900,
        "家务清洁省心力",
        (
            ("类型", ("洗衣液", "洗洁精", "消毒", "洁厕")),
            ("规格", ("补充装", "瓶装", "组合")),
        ),
        domain="home_daily",
        variant_key="香型",
        variant_values=("柠檬", "薰衣草", "原味", "松木"),
    ),
    CategorySpec(
        "paper_tissue",
        "纸品湿巾",
        ("tissue", "papertowel", "纸巾", "抽纸", "卷纸", "湿巾"),
        3_900,
        "生活用纸囤货装",
        (
            ("类型", ("抽纸", "卷纸", "厨房纸", "湿巾")),
            ("层数", ("3层", "4层")),
        ),
        domain="home_daily",
        variant_key="规格",
        variant_values=("10卷", "20卷", "24包", "整箱"),
    ),
    CategorySpec(
        "storage",
        "收纳整理",
        ("storage", "organizer", "收纳", "整理箱", "收纳盒", "置物架"),
        4_900,
        "告别杂乱无章",
        (
            ("类型", ("收纳箱", "收纳盒", "置物架", "真空袋")),
            ("材质", ("塑料", "无纺布", "竹木")),
        ),
        domain="home_daily",
    ),
    CategorySpec(
        "lighting",
        "灯具照明",
        ("light", "lamp", "灯", "台灯", "吸顶灯", "灯泡"),
        8_900,
        "点亮每个角落",
        (
            ("类型", ("台灯", "吸顶灯", "灯泡", "落地灯")),
            ("场景", ("客厅", "卧室", "书房")),
        ),
        domain="home_daily",
        variant_key="色温",
        variant_values=("暖光", "自然光", "冷光", "三档可调"),
    ),
    CategorySpec(
        "home_decor",
        "家居饰品",
        ("decor", "ornament", "装饰", "摆件", "挂画", "花瓶"),
        6_900,
        "家的氛围感担当",
        (
            ("类型", ("摆件", "挂画", "花瓶", "相框")),
            ("材质", ("陶瓷", "木质", "金属", "树脂")),
        ),
        domain="home_daily",
        variant_key="风格",
        variant_values=("北欧", "中式", "现代", "复古"),
    ),
    CategorySpec(
        "towels",
        "毛巾浴巾",
        ("towel", "浴巾", "毛巾", "方巾"),
        2_500,
        "亲肤吸水柔软",
        (
            ("类型", ("毛巾", "浴巾", "方巾", "浴帽")),
            ("克重", ("120g", "180g", "250g")),
        ),
        domain="home_daily",
    ),
    CategorySpec(
        "umbrellas",
        "雨具雨伞",
        ("umbrella", "雨伞", "雨具", "遮阳伞"),
        4_500,
        "晴雨两用不狼狈",
        (
            ("类型", ("长柄", "折叠", "自动", "反向")),
            ("抗风", ("6级", "8级")),
        ),
        domain="home_daily",
    ),
    CategorySpec(
        "hardware",
        "五金工具",
        ("tools", "hardware", "五金", "工具", "螺丝刀", "扳手", "电钻"),
        9_900,
        "家里修缮自己来",
        (
            ("类型", ("套装工具", "螺丝刀", "扳手", "电钻")),
            ("材质", ("铬钒钢", "碳钢", "S2")),
        ),
        domain="home_daily",
        variant_key="件数",
        variant_values=("12件", "32件", "58件", "108件"),
    ),
    CategorySpec(
        "gardening",
        "绿植园艺",
        ("plant", "gardening", "绿植", "花卉", "花盆", "园艺"),
        1_590,
        "给家添一抹绿",
        (
            ("类型", ("绿植", "花卉", "花盆", "营养土")),
            ("养护", ("好养", "需照看")),
        ),
        domain="home_daily",
        variant_key="规格",
        variant_values=("小盆", "中盆", "大盆", "组合"),
    ),
    # --- domain: kitchen 厨房餐具 (7) ----------------------------------------
    CategorySpec(
        "cookware",
        "锅具",
        ("cookware", "pan", "pot", "锅", "炒锅", "煎锅", "汤锅"),
        19_900,
        "好锅成就好菜",
        (
            ("类型", ("炒锅", "煎锅", "汤锅", "奶锅")),
            ("材质", ("铁锅", "不粘", "不锈钢", "麦饭石")),
        ),
        domain="kitchen",
        variant_key="尺寸",
        variant_values=("26cm", "28cm", "30cm", "32cm"),
    ),
    CategorySpec(
        "tableware",
        "餐具",
        ("tableware", "dinnerware", "餐具", "碗", "盘"),
        9_900,
        "一日三餐的好载体",
        (
            ("类型", ("碗盘", "筷勺", "刀具", "砧板")),
            ("材质", ("陶瓷", "骨瓷", "不锈钢", "竹木")),
        ),
        domain="kitchen",
        variant_key="规格",
        variant_values=("单人份", "双人份", "四人份", "六人份"),
    ),
    CategorySpec(
        "cups",
        "水具杯壶",
        ("cup", "mug", "bottle", "杯子", "水杯", "保温杯", "壶"),
        5_900,
        "喝水也要有仪式感",
        (
            ("类型", ("水杯", "保温杯", "茶壶", "凉水壶")),
            ("材质", ("玻璃", "不锈钢", "陶瓷", "塑料")),
        ),
        domain="kitchen",
        variant_key="容量",
        variant_values=("350ml", "500ml", "750ml", "1L"),
    ),
    CategorySpec(
        "kitchen_tools",
        "厨房小工具",
        ("kitchentools", "厨房", "厨具", "削皮器", "开罐器", "打蛋器"),
        3_900,
        "下厨效率小帮手",
        (
            ("类型", ("削皮", "开罐", "打蛋", "计量")),
            ("材质", ("不锈钢", "塑料", "硅胶")),
        ),
        domain="kitchen",
        variant_key="规格",
        variant_values=("单件", "三件套", "五件套", "八件套"),
    ),
    CategorySpec(
        "food_containers",
        "保鲜盒",
        ("container", "保鲜盒", "饭盒", "便当盒", "密封罐"),
        4_900,
        "食材收纳不串味",
        (
            ("类型", ("保鲜盒", "便当盒", "密封罐", "保鲜袋")),
            ("材质", ("玻璃", "塑料", "不锈钢")),
        ),
        domain="kitchen",
        variant_key="规格",
        variant_values=("300ml", "800ml", "1.5L", "套装"),
    ),
    CategorySpec(
        "chopsticks",
        "筷子勺铲",
        ("chopsticks", "筷子", "勺子", "铲子", "汤勺"),
        1_290,
        "一日三餐握在手里",
        (
            ("类型", ("筷子", "勺子", "铲勺", "漏勺")),
            ("材质", ("竹", "木", "不锈钢", "合金")),
        ),
        domain="kitchen",
        variant_key="规格",
        variant_values=("两双装", "五双装", "十双装", "家庭套装"),
    ),
    CategorySpec(
        "tea_sets",
        "茶具咖啡具",
        ("teaset", "茶具", "茶壶", "咖啡具", "手冲", "茶杯"),
        15_900,
        "泡茶手冲都讲究",
        (
            ("类型", ("茶壶", "茶杯", "手冲壶", "咖啡杯")),
            ("材质", ("玻璃", "陶瓷", "紫砂")),
        ),
        domain="kitchen",
        variant_key="规格",
        variant_values=("单人", "双人", "四人", "礼盒"),
    ),
    # --- domain: home_textile 家纺床品 (3) -----------------------------------
    CategorySpec(
        "bedding",
        "床品套件",
        ("bedding", "sheet", "四件套", "床品", "被套", "床单"),
        19_900,
        "一夜好眠的基础",
        (
            ("材质", ("纯棉", "天丝", "磨毛")),
            ("支数", ("40支", "60支", "100支")),
        ),
        domain="home_textile",
        variant_key="尺码",
        variant_values=("1.2米", "1.5米", "1.8米", "2.0米"),
    ),
    CategorySpec(
        "pillows",
        "枕头",
        ("pillow", "枕头", "枕芯", "靠垫"),
        8_900,
        "承托颈椎每一夜",
        (
            ("类型", ("枕芯", "靠垫", "抱枕")),
            ("材质", ("记忆棉", "乳胶", "羽绒", "荞麦")),
        ),
        domain="home_textile",
        variant_key="高度",
        variant_values=("低枕", "中枕", "高枕", "可调"),
    ),
    CategorySpec(
        "blankets",
        "被子毯类",
        ("blanket", "quilt", "被子", "毯子", "被芯", "羽绒被"),
        25_900,
        "四季被毯温暖相伴",
        (
            ("类型", ("羽绒被", "蚕丝被", "纤维被", "毯子")),
            ("保暖", ("春秋被", "冬被", "夏凉被")),
        ),
        domain="home_textile",
        variant_key="尺码",
        variant_values=("150乘200", "180乘220", "200乘230", "220乘240"),
    ),
    # --- domain: furniture 大件家具 (6) --------------------------------------
    CategorySpec(
        "sofa",
        "沙发",
        ("sofa", "沙发", "布艺沙发", "皮质沙发"),
        399_900,
        "客厅的核心阵地",
        (
            ("材质", ("布艺", "皮质", "科技布")),
            ("结构", ("两人位", "三人位", "L型")),
        ),
        domain="furniture",
    ),
    CategorySpec(
        "bed_frames",
        "床架",
        ("bedframe", "bed", "床", "床架", "板式床", "实木床"),
        199_900,
        "安稳睡眠的承托",
        (
            ("材质", ("实木", "板式", "铁艺")),
            ("储物", ("带储物", "不带储物")),
        ),
        domain="furniture",
        variant_key="尺码",
        variant_values=("1.2米", "1.5米", "1.8米", "2.0米"),
    ),
    CategorySpec(
        "mattress",
        "床垫",
        ("mattress", "床垫", "弹簧床垫", "乳胶床垫"),
        159_900,
        "好睡眠的核心投资",
        (
            ("材质", ("弹簧", "乳胶", "记忆棉", "棕")),
            ("厚度", ("5cm", "15cm", "22cm", "28cm")),
        ),
        domain="furniture",
        variant_key="尺码",
        variant_values=("1.2米", "1.5米", "1.8米", "2.0米"),
    ),
    CategorySpec(
        "dining_sets",
        "餐桌椅",
        ("diningtable", "餐桌", "餐椅", "餐桌椅"),
        129_900,
        "围坐吃饭的仪式感",
        (
            ("材质", ("实木", "岩板", "玻璃", "大理石")),
            ("结构", ("一桌四椅", "一桌六椅", "仅桌子")),
        ),
        domain="furniture",
        variant_key="尺码",
        variant_values=("四人位", "六人位", "八人位", "可伸缩"),
    ),
    CategorySpec(
        "wardrobes",
        "衣柜",
        ("wardrobe", "closet", "衣柜", "衣橱", "斗柜"),
        159_900,
        "衣物收纳主力",
        (
            ("材质", ("实木", "板式", "金属")),
            ("门型", ("平开门", "推拉门", "开放式")),
        ),
        domain="furniture",
        variant_key="门数",
        variant_values=("两门", "三门", "四门", "六门"),
    ),
    CategorySpec(
        "coffee_tables",
        "茶几电视柜",
        ("coffeetable", "tvstand", "茶几", "电视柜", "边几"),
        89_900,
        "客厅收纳与陈列",
        (
            ("类型", ("茶几", "电视柜", "边几", "组合")),
            ("材质", ("岩板", "实木", "玻璃")),
        ),
        domain="furniture",
        variant_key="尺码",
        variant_values=("小号", "中号", "大号", "组合"),
    ),
    # --- domain: appliances 家用电器 (12) ------------------------------------
    CategorySpec(
        "air_conditioner",
        "空调",
        ("airconditioner", "ac", "空调", "挂机", "柜机", "中央空调"),
        259_900,
        "冬暖夏凉全靠它",
        (
            ("类型", ("挂机", "柜机", "中央")),
            ("能效", ("一级", "二级", "三级")),
        ),
        domain="appliances",
        variant_key="匹数",
        variant_values=("1匹", "1.5匹", "2匹", "3匹"),
    ),
    CategorySpec(
        "washing_machine",
        "洗衣机",
        ("washingmachine", "washer", "洗衣机", "滚筒", "波轮"),
        199_900,
        "解放双手的日常",
        (
            ("类型", ("滚筒", "波轮", "洗烘一体")),
            ("电机", ("变频", "定频")),
        ),
        domain="appliances",
        variant_key="容量",
        variant_values=("8kg", "10kg", "12kg", "15kg"),
    ),
    CategorySpec(
        "refrigerator",
        "冰箱",
        ("refrigerator", "fridge", "冰箱", "冷柜", "双开门"),
        259_900,
        "食材新鲜大管家",
        (
            ("类型", ("双开门", "三门", "十字对开", "冷柜")),
            ("制冷", ("风冷", "直冷")),
        ),
        domain="appliances",
        variant_key="容量",
        variant_values=("200L", "300L", "450L", "600L"),
    ),
    CategorySpec(
        "tv",
        "电视",
        ("tv", "television", "电视", "液晶电视", "智慧屏"),
        229_900,
        "客厅影音中心",
        (
            ("类型", ("LCD", "OLED", "QLED")),
            ("刷新率", ("60Hz", "120Hz", "144Hz")),
        ),
        domain="appliances",
        variant_key="尺寸",
        variant_values=("43英寸", "55英寸", "65英寸", "75英寸"),
    ),
    CategorySpec(
        "kitchen_appliance",
        "厨房大电",
        ("kitchenappliance", "油烟机", "燃气灶", "洗碗机", "消毒柜", "集成灶"),
        189_900,
        "下厨环境大升级",
        (
            ("类型", ("油烟机", "燃气灶", "洗碗机", "消毒柜")),
            ("安装", ("嵌入式", "独立式")),
        ),
        domain="appliances",
        variant_key="规格",
        variant_values=("标准款", "大吸力", "静音款", "套装"),
    ),
    CategorySpec(
        "water_heater",
        "热水器",
        ("waterheater", "热水器", "电热水器", "燃气热水器"),
        159_900,
        "随时享用热水澡",
        (
            ("类型", ("电储水", "燃气", "即热式")),
            ("能效", ("一级", "二级")),
        ),
        domain="appliances",
        variant_key="规格",
        variant_values=("50L", "60L", "80L", "100L"),
    ),
    CategorySpec(
        "vacuum",
        "吸尘器",
        ("vacuum", "吸尘器", "扫地机器人", "洗地机", "除螨仪"),
        129_900,
        "地面清洁自动化",
        (
            ("类型", ("手持", "扫地机", "洗地机", "除螨")),
            ("续航", ("45分钟", "60分钟", "90分钟")),
        ),
        domain="appliances",
        variant_key="档位",
        variant_values=("标准版", "高配版", "旗舰版", "套装"),
    ),
    CategorySpec(
        "air_purifier",
        "空气净化器",
        ("airpurifier", "净化器", "空气净化", "除甲醛"),
        99_900,
        "呼吸安心每一口",
        (
            ("功能", ("除醛", "除霾", "除菌", "加湿净化")),
            ("CADR", ("300", "400", "600", "800")),
        ),
        domain="appliances",
        variant_key="档位",
        variant_values=("卧室款", "客厅款", "大屏款", "全屋款"),
    ),
    CategorySpec(
        "fans",
        "风扇冷风扇",
        ("fan", "风扇", "电风扇", "冷风扇", "循环扇"),
        39_900,
        "夏日通风降温",
        (
            ("类型", ("落地", "台式", "循环", "冷风扇")),
            ("控制", ("遥控", "APP", "机械")),
        ),
        domain="appliances",
        variant_key="档位",
        variant_values=("基础版", "遥控版", "智能版", "旗舰版"),
    ),
    CategorySpec(
        "humidifier",
        "加湿器",
        ("humidifier", "加湿器", "除湿器", "除湿机"),
        69_900,
        "四季湿度平衡",
        (
            ("类型", ("加湿", "除湿", "净化加湿")),
            ("水箱", ("3L", "5L", "7L")),
        ),
        domain="appliances",
        variant_key="档位",
        variant_values=("基础版", "智能版", "大容量", "套装"),
    ),
    CategorySpec(
        "microwave_oven",
        "微波炉烤箱",
        ("microwave", "oven", "微波炉", "烤箱", "蒸烤箱", "空气炸锅"),
        49_900,
        "快速加热与烘焙",
        (
            ("类型", ("微波炉", "烤箱", "蒸烤一体", "空气炸锅")),
            ("容量分级", ("小型", "中型", "大型")),
        ),
        domain="appliances",
        variant_key="容量",
        variant_values=("20L", "23L", "30L", "60L"),
    ),
    CategorySpec(
        "rice_cooker",
        "电饭煲",
        ("ricecooker", "电饭煲", "电饭锅", "压力锅", "电炖锅"),
        29_900,
        "一日三餐主食担当",
        (
            ("类型", ("电饭煲", "压力锅", "电炖锅")),
            ("加热", ("底盘", "IH电磁")),
        ),
        domain="appliances",
        variant_key="容量",
        variant_values=("3L", "4L", "5L", "6L"),
    ),
    # --- domain: digital 数码通讯 (8) ----------------------------------------
    CategorySpec(
        "smartphone",
        "手机",
        ("phone", "smartphone", "手机", "5G手机"),
        399_900,
        "口袋里的全能终端",
        (
            ("屏幕", ("直屏", "曲面", "折叠")),
            ("快充", ("65W", "100W", "120W")),
        ),
        domain="digital",
        variant_key="存储",
        variant_values=("128GB", "256GB", "512GB", "1TB"),
    ),
    CategorySpec(
        "tablet",
        "平板电脑",
        ("tablet", "ipad", "平板", "平板电脑", "学习平板"),
        229_900,
        "大屏娱乐与轻办公",
        (
            ("类型", ("娱乐平板", "学习平板", "办公平板")),
            ("网络", ("WiFi版", "插卡版")),
        ),
        domain="digital",
        variant_key="存储",
        variant_values=("64GB", "128GB", "256GB", "512GB"),
    ),
    CategorySpec(
        "smartwatch",
        "智能手表",
        ("smartwatch", "智能手表", "电话手表", "儿童手表"),
        129_900,
        "腕上健康管家",
        (
            ("功能", ("健康监测", "GPS", "独立通话", "eSIM")),
            ("防水", ("50米", "IP68")),
        ),
        domain="digital",
        variant_key="表径",
        variant_values=("41mm", "45mm", "46mm", "49mm"),
    ),
    CategorySpec(
        "smart_band",
        "智能手环",
        ("band", "fitnessband", "手环", "智能手环", "运动手环"),
        29_900,
        "轻装运动记录",
        (
            ("功能", ("心率", "血氧", "睡眠", "泳姿")),
            ("续航", ("14天", "21天", "28天")),
        ),
        domain="digital",
    ),
    CategorySpec(
        "camera",
        "相机摄影",
        ("camera", "相机", "微单", "单反", "镜头", "摄影"),
        599_900,
        "记录生活的仪式感",
        (
            ("类型", ("微单", "单反", "卡片机", "镜头")),
            ("画幅", ("APS-C", "全画幅")),
        ),
        domain="digital",
        variant_key="套装",
        variant_values=("机身", "套机镜头", "双镜头", "创作者套装"),
    ),
    CategorySpec(
        "game_console",
        "游戏主机",
        ("console", "gaming", "游戏机", "主机", "手柄", "Switch", "PS5"),
        259_900,
        "客厅娱乐新维度",
        (
            ("类型", ("掌机", "家用机", "掌机配件")),
            ("存储", ("64GB", "256GB", "512GB", "1TB")),
        ),
        domain="digital",
        variant_key="版本",
        variant_values=("国行", "日版", "港版", "美版"),
    ),
    CategorySpec(
        "smart_speaker",
        "智能音箱",
        ("smartspeaker", "智能音箱", "语音助手", "闹钟音箱"),
        19_900,
        "一句话控制全屋",
        (
            ("功能", ("语音助手", "屏幕", "闹钟", "网关")),
            ("音质", ("单声道", "立体声")),
        ),
        domain="digital",
        variant_key="款式",
        variant_values=("迷你版", "标准版", "屏显版", "旗舰版"),
    ),
    CategorySpec(
        "e_reader",
        "电子书阅读器",
        ("ereader", "kindle", "电子书", "阅读器", "电纸书"),
        99_900,
        "沉浸阅读不伤眼",
        (
            ("屏幕", ("6英寸", "7英寸", "10英寸")),
            ("灯光", ("阅读灯", "暖光", "无灯")),
        ),
        domain="digital",
        variant_key="存储",
        variant_values=("8GB", "16GB", "32GB", "64GB"),
    ),
    # --- domain: mother_baby 母婴玩具 (9) ------------------------------------
    CategorySpec(
        "baby_formula",
        "婴儿奶粉",
        ("formula", "milkpowder", "奶粉", "婴儿奶粉", "幼儿奶粉"),
        259_900,
        "宝宝口粮严选",
        (
            ("奶源", ("国产", "欧洲", "新西兰", "澳洲")),
            ("配方", ("普通", "有机", "A2")),
        ),
        domain="mother_baby",
        variant_key="段数",
        variant_values=("1段", "2段", "3段", "4段"),
    ),
    CategorySpec(
        "diapers",
        "纸尿裤",
        ("diaper", "纸尿裤", "拉拉裤", "尿不湿"),
        89_900,
        "干爽一夜安睡",
        (
            ("类型", ("纸尿裤", "拉拉裤", "游泳裤")),
            ("特点", ("超薄", "透气", "敏感肌")),
        ),
        domain="mother_baby",
        variant_key="尺码",
        variant_values=("NB", "S", "M", "L"),
    ),
    CategorySpec(
        "baby_toys",
        "婴儿玩具",
        ("babytoy", "婴儿玩具", "安抚玩具", "牙胶", "爬行垫"),
        9_900,
        "陪宝宝长大",
        (
            ("类型", ("牙胶", "安抚", "爬行垫", "健身架")),
            ("材质", ("食品级硅胶", "PP", "布艺")),
        ),
        domain="mother_baby",
        variant_key="适龄",
        variant_values=("0-6个月", "6-12个月", "1-3岁", "3-6岁"),
    ),
    CategorySpec(
        "strollers",
        "婴儿推车",
        ("stroller", "推车", "婴儿车", "遛娃神器", "伞车"),
        129_900,
        "带娃出行更轻松",
        (
            ("类型", ("高景观", "轻便伞车", "双向", "多功能")),
            ("减震", ("四轮减震", "充气轮")),
        ),
        domain="mother_baby",
        variant_key="款式",
        variant_values=("标准版", "轻便版", "高配版", "双人版"),
    ),
    CategorySpec(
        "car_seats",
        "安全座椅",
        ("carseat", "安全座椅", "儿童座椅", "提篮"),
        159_900,
        "出行安全第一步",
        (
            ("安装", ("ISOFIX", "安全带")),
            ("朝向", ("双向", "朝后", "朝前")),
        ),
        domain="mother_baby",
        variant_key="组别",
        variant_values=("婴儿组", "幼儿组", "成长组", "全能组"),
    ),
    CategorySpec(
        "baby_care",
        "婴儿洗护",
        ("babycare", "婴儿洗护", "宝宝霜", "沐浴露", "爽身粉"),
        4_500,
        "温和呵护娇嫩肌",
        (
            ("类型", ("沐浴", "润肤", "护臀", "洗发")),
            ("功效", ("保湿", "舒缓", "无泪配方")),
        ),
        domain="mother_baby",
        variant_key="规格",
        variant_values=("100ml", "200ml", "300ml", "套装"),
    ),
    CategorySpec(
        "maternity",
        "孕产用品",
        ("maternity", "孕产", "孕妇装", "待产包", "月子服", "哺乳内衣"),
        9_900,
        "准妈妈的贴心装备",
        (
            ("类型", ("孕妇装", "哺乳内衣", "待产包", "月子帽")),
            ("材质", ("纯棉", "莫代尔")),
        ),
        domain="mother_baby",
        variant_key="尺码",
        variant_values=("S", "M", "L", "XL"),
    ),
    CategorySpec(
        "building_toys",
        "积木益智",
        ("buildingblocks", "积木", "拼装", "益智玩具", "乐高", "磁力片"),
        19_900,
        "动手动脑玩起来",
        (
            ("类型", ("积木", "磁力片", "拼图", "编程机器人")),
            ("材质", ("ABS", "木质", "磁力")),
        ),
        domain="mother_baby",
        variant_key="适龄",
        variant_values=("1-3岁", "3-6岁", "6-9岁", "9岁+"),
    ),
    CategorySpec(
        "kids_shoes",
        "童鞋童袜",
        ("kidsshoes", "童鞋", "童袜", "儿童鞋"),
        6_900,
        "小脚丫舒适合脚",
        (
            ("类型", ("运动鞋", "皮鞋", "凉鞋", "袜子")),
            ("材质", ("网面", "皮革", "纯棉")),
        ),
        domain="mother_baby",
        variant_key="尺码",
        variant_values=("26码", "30码", "34码", "38码"),
    ),
    # --- domain: sports 运动户外 (10) ----------------------------------------
    CategorySpec(
        "sportswear",
        "运动服饰",
        ("sportswear", "运动服", "速干衣", "运动T恤", "运动裤"),
        12_900,
        "运动出汗也自在",
        (
            ("类型", ("上衣", "裤装", "套装", "外套")),
            ("材质", ("速干", "棉", "针织")),
        ),
        domain="sports",
        variant_key="尺码",
        variant_values=("S", "M", "L", "XL"),
    ),
    CategorySpec(
        "running_shoes",
        "跑步鞋",
        ("runningshoes", "跑鞋", "跑步鞋", "慢跑鞋", "竞速鞋"),
        45_900,
        "每一步都有回弹",
        (
            ("类型", ("慢跑", "竞速", "越野", "训练")),
            ("缓震", ("入门", "进阶", "顶级")),
        ),
        domain="sports",
        variant_key="鞋码",
        variant_values=("40", "41", "42", "43"),
    ),
    CategorySpec(
        "fitness_equipment",
        "健身器材",
        ("fitness", "哑铃", "瑜伽垫", "跳绳", "健身器材", "弹力带", "健腹轮"),
        8_900,
        "居家锻炼随时开练",
        (
            ("类型", ("哑铃", "瑜伽垫", "跳绳", "弹力带")),
            ("材质", ("包胶", "纯钢", "NBR")),
        ),
        domain="sports",
        variant_key="规格",
        variant_values=("入门套装", "进阶套装", "单件", "家庭装"),
    ),
    CategorySpec(
        "outdoor_jacket",
        "冲锋衣户外服",
        ("outdoorjacket", "冲锋衣", "抓绒衣", "防晒衣", "皮肤衣"),
        39_900,
        "风雨户外都不怕",
        (
            ("类型", ("冲锋衣", "抓绒", "防晒", "皮肤衣")),
            ("防水", ("5000mm", "10000mm")),
        ),
        domain="sports",
        variant_key="尺码",
        variant_values=("S", "M", "L", "XL"),
    ),
    CategorySpec(
        "camping",
        "露营装备",
        ("camping", "帐篷", "露营", "天幕", "露营桌椅", "睡袋"),
        29_900,
        "把家搬进大自然",
        (
            ("类型", ("帐篷", "天幕", "睡袋", "桌椅")),
            ("季节", ("三季", "四季")),
        ),
        domain="sports",
        variant_key="规格",
        variant_values=("双人", "三人", "四人", "六人"),
    ),
    CategorySpec(
        "hiking",
        "徒步装备",
        ("hiking", "徒步", "登山杖", "登山鞋", "户外背包", "驮包"),
        15_900,
        "说走就走的徒步",
        (
            ("类型", ("登山鞋", "登山杖", "背包", "水袋")),
            ("防水", ("防泼水", "全防水")),
        ),
        domain="sports",
        variant_key="容量",
        variant_values=("20L", "35L", "50L", "65L"),
    ),
    CategorySpec(
        "cycling",
        "骑行装备",
        ("cycling", "骑行", "自行车", "头盔", "骑行服", "码表"),
        19_900,
        "骑乐无穷装备全",
        (
            ("类型", ("整车", "头盔", "骑行服", "码表")),
            ("变速", ("单速", "18速", "24速", "27速")),
        ),
        domain="sports",
        variant_key="车架",
        variant_values=("S", "M", "L", "XL"),
    ),
    CategorySpec(
        "fishing",
        "垂钓渔具",
        ("fishing", "渔具", "鱼竿", "钓箱", "鱼线", "路亚"),
        12_900,
        "静候鱼儿上钩",
        (
            ("类型", ("鱼竿", "鱼线", "钓箱", "路亚")),
            ("材质", ("碳素", "玻璃钢")),
        ),
        domain="sports",
        variant_key="规格",
        variant_values=("3.6米", "4.5米", "5.4米", "套装"),
    ),
    CategorySpec(
        "ball_sports",
        "球类运动",
        ("ball", "basketball", "football", "篮球", "足球", "羽毛球", "乒乓球", "排球"),
        7_900,
        "约球走起",
        (
            ("类型", ("篮球", "足球", "羽毛球", "乒乓球")),
            ("场地", ("室内", "室外")),
        ),
        domain="sports",
        variant_key="规格",
        variant_values=("3号球", "4号球", "5号球", "7号球"),
    ),
    CategorySpec(
        "swimming",
        "泳装泳具",
        ("swimming", "泳衣", "泳镜", "泳帽", "泳圈", "浮板"),
        9_900,
        "畅游一夏",
        (
            ("类型", ("泳衣", "泳镜", "泳帽", "泳圈")),
            ("场景", ("泳池", "户外", "训练")),
        ),
        domain="sports",
        variant_key="尺码",
        variant_values=("S", "M", "L", "XL"),
    ),
    # --- domain: entertainment 图书文娱 (6) ----------------------------------
    CategorySpec(
        "books",
        "图书",
        ("book", "图书", "书籍", "小说", "绘本", "教材"),
        4_500,
        "阅读点亮生活",
        (
            ("类型", ("文学", "社科", "童书", "教材")),
            ("版次", ("新版", "经典版")),
        ),
        domain="entertainment",
        variant_key="装帧",
        variant_values=("平装", "精装", "套装", "典藏"),
    ),
    CategorySpec(
        "stationery",
        "文具",
        ("stationery", "文具", "笔", "笔记本", "文具盒", "尺子"),
        900,
        "书写整齐有条理",
        (
            ("类型", ("笔类", "本册", "收纳", "修正")),
            ("材质", ("塑料", "金属", "纸质")),
        ),
        domain="entertainment",
    ),
    CategorySpec(
        "art_supplies",
        "绘画用品",
        ("painting", "绘画", "画笔", "颜料", "画板", "素描"),
        1_900,
        "释放创作灵感",
        (
            ("类型", ("画笔", "颜料", "画板", "素描")),
            ("适用", ("入门", "进阶", "专业")),
        ),
        domain="entertainment",
        variant_key="规格",
        variant_values=("入门套装", "标准套装", "专业套装", "单套补充"),
    ),
    CategorySpec(
        "musical_instruments",
        "乐器",
        ("instrument", "guitar", "乐器", "吉他", "钢琴", "电子琴", "尤克里里", "笛子"),
        89_900,
        "奏响生活乐章",
        (
            ("类型", ("吉他", "键盘", "管乐", "弦乐")),
            ("适用", ("入门", "进阶", "专业")),
        ),
        domain="entertainment",
        variant_key="款式",
        variant_values=("入门款", "标准款", "进阶款", "演出款"),
    ),
    CategorySpec(
        "collectibles",
        "手办潮玩",
        ("figure", "collectible", "手办", "潮玩", "盲盒", "模型", "扭蛋"),
        12_900,
        "收藏热爱每一份",
        (
            ("类型", ("手办", "盲盒", "模型", "积木潮玩")),
            ("比例", ("1/7", "1/8", "1/12")),
        ),
        domain="entertainment",
        variant_key="版本",
        variant_values=("普通版", "豪华版", "限量版", "签名版"),
    ),
    CategorySpec(
        "board_games",
        "桌游卡牌",
        ("boardgame", "桌游", "卡牌", "UNO", "狼人杀", "剧本杀", "三国杀"),
        9_900,
        "聚会破冰神器",
        (
            ("类型", ("桌游", "卡牌", "剧本", "益智")),
            ("时长", ("30分钟", "60分钟", "90分钟")),
        ),
        domain="entertainment",
        variant_key="人数",
        variant_values=("2人", "3-4人", "5-6人", "8人以上"),
    ),
    # --- domain: pets 宠物生活 (6) --------------------------------------------
    CategorySpec(
        "pet_food",
        "宠物主粮",
        ("petfood", "猫粮", "狗粮", "主粮", "宠粮"),
        12_900,
        "毛孩子的主食口粮",
        (
            ("适用", ("猫", "狗", "幼犬", "幼猫")),
            ("配方", ("无谷", "冻干", "处方", "普通")),
        ),
        domain="pets",
        variant_key="规格",
        variant_values=("1.5kg", "3kg", "7kg", "15kg"),
    ),
    CategorySpec(
        "pet_treats",
        "宠物零食",
        ("pettreats", "冻干", "肉干", "磨牙棒", "罐头"),
        3_900,
        "训练奖励小零食",
        (
            ("类型", ("冻干", "肉干", "罐头", "磨牙棒")),
            ("适用", ("猫", "狗")),
        ),
        domain="pets",
        variant_key="规格",
        variant_values=("50g", "100g", "250g", "500g"),
    ),
    CategorySpec(
        "pet_supplies",
        "宠物用品",
        ("petsupplies", "猫砂", "猫爬架", "狗窝", "牵引绳", "猫砂盆"),
        4_900,
        "养宠日常好帮手",
        (
            ("类型", ("猫砂", "猫爬架", "狗窝", "牵引绳")),
            ("特点", ("除臭", "低尘", "稳固")),
        ),
        domain="pets",
        variant_key="尺码",
        variant_values=("S", "M", "L", "XL"),
    ),
    CategorySpec(
        "pet_toys",
        "宠物玩具",
        ("pettoys", "逗猫棒", "宠物玩具", "咬咬玩具"),
        1_590,
        "释放毛孩精力",
        (
            ("类型", ("逗猫棒", "咬咬", "球类", "漏食")),
            ("材质", ("毛绒", "橡胶", "麻绳")),
        ),
        domain="pets",
        variant_key="尺码",
        variant_values=("S", "M", "L", "加大"),
    ),
    CategorySpec(
        "pet_clean",
        "宠物清洁",
        ("petclean", "宠物香波", "除臭剂", "宠物沐浴", "梳毛"),
        2_500,
        "干净毛孩子人人爱",
        (
            ("类型", ("香波", "除臭", "梳毛", "湿巾")),
            ("功效", ("除蚤", "柔顺", "抑菌")),
        ),
        domain="pets",
        variant_key="规格",
        variant_values=("250ml", "500ml", "1L", "补充装"),
    ),
    CategorySpec(
        "aquarium",
        "水族用品",
        ("aquarium", "鱼缸", "水族", "过滤器", "鱼粮", "加热棒"),
        9_900,
        "一方水中生态",
        (
            ("类型", ("鱼缸", "过滤器", "鱼粮", "灯具")),
            ("材质", ("玻璃", "亚克力")),
        ),
        domain="pets",
        variant_key="规格",
        variant_values=("小型", "中型", "大型", "套装"),
    ),
    # --- domain: auto 汽车用品 (6) --------------------------------------------
    CategorySpec(
        "car_care",
        "洗车养护",
        ("carcare", "洗车", "打蜡", "镀膜", "玻璃水", "洗车机"),
        3_900,
        "自己动手养车",
        (
            ("类型", ("洗车液", "打蜡", "镀膜", "玻璃水")),
            ("适用", ("漆面", "玻璃", "内饰")),
        ),
        domain="auto",
        variant_key="规格",
        variant_values=("500ml", "1L", "2L", "套装"),
    ),
    CategorySpec(
        "car_interior",
        "车内饰品",
        ("carinterior", "脚垫", "座套", "方向盘套", "车内香水"),
        8_900,
        "车里也有仪式感",
        (
            ("类型", ("脚垫", "座套", "方向盘套", "摆件")),
            ("材质", ("皮革", "丝圈", "TPE")),
        ),
        domain="auto",
    ),
    CategorySpec(
        "car_electronics",
        "车载电子",
        ("carelectronics", "行车记录仪", "车载充电", "导航", "车载吸尘器", "胎压监测"),
        49_900,
        "行车安全与便利",
        (
            ("类型", ("记录仪", "充电", "吸尘", "胎压")),
            ("供电", ("点烟器", "USB", "太阳能")),
        ),
        domain="auto",
        variant_key="分辨率",
        variant_values=("720P", "1080P", "2K", "4K"),
    ),
    CategorySpec(
        "car_maintenance",
        "机油养护",
        ("motoroil", "机油", "添加剂", "机油滤芯", "防冻液"),
        8_900,
        "动力持久养护",
        (
            ("类型", ("机油", "滤芯", "添加剂", "防冻液")),
            ("粘度", ("0W-20", "5W-30", "5W-40", "10W-40")),
        ),
        domain="auto",
        variant_key="规格",
        variant_values=("1L", "4L", "组合装", "礼盒装"),
    ),
    CategorySpec(
        "car_exterior",
        "车外装饰",
        ("carexterior", "车贴", "雨挡", "挡泥板", "车衣", "尾翼"),
        4_900,
        "个性与防护兼得",
        (
            ("类型", ("车贴", "雨挡", "挡泥板", "车衣")),
            ("材质", ("ABS", "不锈钢", "布")),
        ),
        domain="auto",
        variant_key="款式",
        variant_values=("通用款", "专车定制", "运动款", "基础款"),
    ),
    CategorySpec(
        "riding_gear",
        "摩托骑行装备",
        ("ridinggear", "摩托车", "骑行服", "头盔", "护具"),
        39_900,
        "安全帅气的骑行",
        (
            ("类型", ("头盔", "骑行服", "护具", "手套")),
            ("认证", ("3C", "DOT", "ECE")),
        ),
        domain="auto",
        variant_key="尺码",
        variant_values=("S", "M", "L", "XL"),
    ),
    # --- domain: health 医疗健康 (5) -------------------------------------------
    CategorySpec(
        "health_devices",
        "健康器械",
        ("healthdevice", "血压计", "血糖仪", "体温计", "血氧仪", "雾化器"),
        9_900,
        "家庭健康小卫士",
        (
            ("类型", ("血压计", "血糖仪", "体温计", "血氧仪")),
            ("供电", ("电池", "充电", "插电")),
        ),
        domain="health",
        variant_key="款式",
        variant_values=("基础版", "语音版", "蓝牙版", "大屏版"),
    ),
    CategorySpec(
        "massage",
        "按摩仪",
        ("massage", "按摩", "按摩仪", "颈椎按摩", "按摩椅", "足浴盆"),
        19_900,
        "舒缓一天的疲惫",
        (
            ("类型", ("颈椎", "眼部", "腰部", "足浴")),
            ("功能", ("加热", "脉冲", "揉捏", "气压")),
        ),
        domain="health",
        variant_key="款式",
        variant_values=("基础版", "加热版", "旗舰版", "至尊版"),
    ),
    CategorySpec(
        "vitamins",
        "保健品",
        ("vitamins", "维生素", "钙片", "鱼油", "褪黑素"),
        12_900,
        "日常营养补充",
        (
            ("类型", ("维生素", "矿物质", "鱼油", "助眠")),
            ("人群", ("成人", "女性", "男性", "中老年")),
        ),
        domain="health",
        variant_key="规格",
        variant_values=("30粒", "60粒", "90粒", "礼盒装"),
    ),
    CategorySpec(
        "sports_nutrition",
        "运动营养",
        ("sportsnutrition", "蛋白粉", "肌酸", "左旋", "能量胶", "电解质"),
        15_900,
        "训练补给跟得上",
        (
            ("类型", ("蛋白粉", "肌酸", "能量胶", "电解质")),
            ("适用", ("增肌", "减脂", "耐力")),
        ),
        domain="health",
        variant_key="口味",
        variant_values=("原味", "巧克力", "香草", "水果"),
    ),
    CategorySpec(
        "vision_care",
        "隐形眼镜护理",
        ("contacts", "contactlens", "隐形眼镜", "护理液", "美瞳", "润眼液"),
        6_900,
        "清晰视界隐形无忧",
        (
            ("类型", ("日抛", "月抛", "护理液", "润眼液")),
            ("材质", ("硅水凝胶", "水凝胶")),
        ),
        domain="health",
        variant_key="规格",
        variant_values=("日抛10片", "日抛30片", "月抛2片", "月抛6片"),
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

# One series word per model index keeps the model dimension shallow: each
# (category, brand) pair exposes exactly MODELS_PER_PAIR distinct series.
SERIES_WORDS: tuple[str, ...] = (
    "Astro",
    "Nova",
    "Prime",
    "Vertex",
    "Zenith",
    "Titan",
    "Atlas",
    "Omni",
)

SERIES_COUNT = len(SERIES_WORDS)

assert len(CATEGORIES) == CATEGORY_COUNT
assert len(BRANDS) == BRAND_COUNT
assert len(SERIES_WORDS) == SERIES_COUNT
assert MODELS_PER_PAIR == SERIES_COUNT


def taxonomy_from_specs(
    specs: tuple[CategorySpec, ...] = CATEGORIES,
    *,
    taxonomy_version: str = "static-v3",
) -> RegistryTaxonomy:
    """Build the registry taxonomy from generated-category specs."""

    return RegistryTaxonomy(
        taxonomy_version=taxonomy_version,
        categories=[
            RegistryCategory(
                category_id=spec.category_id,
                name=spec.name,
                description=spec.blurb,
                attributes=[
                    CategoryAttribute(name=key, allowed_values=list(values))
                    for key, values in spec.attrs
                ],
                domain=spec.domain,
                provider_policy="generated",
                status="active",
                provider_aliases=list(spec.aliases),
                base_price_minor=spec.base_price_minor,
                variant_key=spec.variant_key,
                variant_values=list(spec.variant_values),
            )
            for spec in specs
        ],
    )


def default_taxonomy() -> RegistryTaxonomy:
    """Return the built-in taxonomy used when no registry version is active."""

    return taxonomy_from_specs(CATEGORIES)


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
    """Pure-function view over a bounded deterministic SKU address space."""

    def __init__(self, taxonomy: CatalogTaxonomy | RegistryTaxonomy | None = None) -> None:
        source = taxonomy or default_taxonomy()
        registry_source = (
            source
            if isinstance(source, RegistryTaxonomy)
            else RegistryTaxonomy.model_validate(source)
        )
        categories = [
            category
            for category in registry_source.categories
            if category.status == "active" and category.provider_policy == "generated"
        ]
        if not categories:
            raise ProviderRejectedError("Generated catalog requires at least one active category")
        for category in categories:
            if not category.variant_key.strip():
                raise ProviderRejectedError(f"类目 {category.category_id} 缺少变体维度名称")
            if len(category.variant_values) != VARIANTS_PER_MODEL:
                raise ProviderRejectedError(
                    f"类目 {category.category_id} 必须声明 {VARIANTS_PER_MODEL} 个变体值"
                )
            if category.variant_key in {attribute.name for attribute in category.attributes}:
                raise ProviderRejectedError(
                    f"类目 {category.category_id} 的变体维度 {category.variant_key} 不能与属性名重复"
                )
        pair_count = len(categories) * BRAND_COUNT
        models_per_pair = max(1, TOTAL_SKUS // (pair_count * VARIANTS_PER_MODEL))
        self.taxonomy_version = registry_source.taxonomy_version
        self._registry_taxonomy = RegistryTaxonomy(
            taxonomy_version=registry_source.taxonomy_version,
            categories=categories,
        )
        self._categories = categories
        self._dynamic_count = 0
        self.category_count = len(categories)
        self.pair_count = pair_count
        self.models_per_pair = models_per_pair
        self.total_skus = pair_count * models_per_pair * VARIANTS_PER_MODEL
        # Pair -> category mapping. Static categories own BRAND_COUNT pairs
        # each (uniform grid); a minted category with an LLM-authored
        # brand_styles matrix owns exactly ONE pair whose model slots are
        # the (brand, series) styles, so a search page spans brands instead
        # of one static brand's whole series line.
        self._pair_category_index: list[int] = [
            category_index
            for category_index in range(len(categories))
            for _ in range(BRAND_COUNT)
        ]
        self._category_pair_start: list[int] = [
            category_index * BRAND_COUNT for category_index in range(len(categories))
        ]
        self._category_pair_width: list[int] = [BRAND_COUNT] * len(categories)

    def _category_of_pair(self, pair_index: int) -> RegistryCategory:
        """Resolve a pair slot to its category across static and minted blocks."""

        return self._categories[self._pair_category_index[pair_index]]

    # --- addressing ---------------------------------------------------------
    def sku_id(self, n: int) -> str:
        return f"SKU-{n:08d}"

    def parse_sku_id(self, sku_id: str) -> int | None:
        match = _SKU_PATTERN.fullmatch(sku_id.strip().upper())
        if match is None:
            return None
        n = int(match.group(1))
        return n if n < self.total_skus else None

    def decompose(self, n: int) -> tuple[int, int, int]:
        """Return ``(pair_index, model_index, variant_index)`` for a valid n."""

        pair_index, rest = divmod(n, self.models_per_pair * VARIANTS_PER_MODEL)
        model_index, variant_index = divmod(rest, VARIANTS_PER_MODEL)
        return pair_index, model_index, variant_index

    def product_id(self, pair_index: int, model_index: int) -> str:
        return f"PRD-{pair_index:03d}-{model_index:05d}"

    def taxonomy(self) -> CatalogTaxonomy:
        """Publish this provider's authoritative generated-category taxonomy."""

        return self._registry_taxonomy.to_catalog_taxonomy()

    def registry_taxonomy(self) -> RegistryTaxonomy:
        """Publish the internal registry taxonomy with variant axes and domains."""

        return self._registry_taxonomy

    # --- dynamic categories --------------------------------------------------
    @staticmethod
    def normalize_name(name: str) -> str:
        """Case/whitespace-insensitive category name key for dedupe."""

        return "".join(name.split()).lower()

    @property
    def dynamic_category_count(self) -> int:
        return self._dynamic_count

    def find_category_by_name(self, name: str) -> RegistryCategory | None:
        key = self.normalize_name(name)
        for category in self._categories:
            if self.normalize_name(category.name) == key:
                return category
        return None

    def register_dynamic(self, category: RegistryCategory) -> None:
        """Append a runtime-minted category without disturbing SKU addressing.

        ``models_per_pair`` is frozen at construction, so every existing
        ``n -> (pair, model, variant)`` mapping stays valid; minted categories
        continue the numeric SKU space after the static block. Reads are
        copy-on-write: searches see either the old or the new category list.

        A category carrying its own ``brand_styles`` matrix occupies a single
        pair (model slots ARE the styles); a legacy proposal without one keeps
        the historical BRAND_COUNT-pair block so existing SKU names never
        change under a live order.
        """

        if category.category_id in {item.category_id for item in self._categories}:
            raise ProviderRejectedError(f"类目标识已存在: {category.category_id}")
        if not category.variant_values or len(category.variant_values) != VARIANTS_PER_MODEL:
            raise ProviderRejectedError(
                f"动态类目 {category.category_id} 必须声明 {VARIANTS_PER_MODEL} 个变体值"
            )
        if category.variant_key in {attribute.name for attribute in category.attributes}:
            raise ProviderRejectedError(
                f"动态类目 {category.category_id} 的变体维度 {category.variant_key} 不能与属性名重复"
            )
        if category.brand_styles and len(category.brand_styles) > self.models_per_pair:
            raise ProviderRejectedError(
                f"动态类目 {category.category_id} 的品牌款式矩阵不得超过 "
                f"{self.models_per_pair} 个组合"
            )
        if self.find_category_by_name(category.name) is not None:
            raise ProviderRejectedError(f"类目名称已存在: {category.name}")
        pairs_added = 1 if category.brand_styles else BRAND_COUNT
        new_index = len(self._categories)
        self._categories = [*self._categories, category]
        self._category_pair_start.append(self.pair_count)
        self._category_pair_width.append(pairs_added)
        self._pair_category_index = [*self._pair_category_index, *([new_index] * pairs_added)]
        self._dynamic_count += 1
        self.category_count += 1
        self.pair_count += pairs_added
        self.total_skus = self.pair_count * self.models_per_pair * VARIANTS_PER_MODEL
        self._registry_taxonomy = RegistryTaxonomy(
            taxonomy_version=self.taxonomy_version,
            categories=list(self._categories),
        )

    # --- fact projection ----------------------------------------------------
    @staticmethod
    def _category_attributes(category: CatalogCategory) -> list[tuple[str, tuple[str, ...]]]:
        return [
            (attribute.name, tuple(attribute.allowed_values)) for attribute in category.attributes
        ]

    def _category_base_price(self, category: RegistryCategory) -> int:
        if category.base_price_minor is not None:
            return category.base_price_minor
        return 4_900 + _stable_hash(abs(hash(category.category_id)), "category-price") % 196_000

    def _unit_price(self, pair_index: int, model_index: int) -> int:
        category = self._category_of_pair(pair_index)
        if category.brand_styles:
            # Style-authored categories price by model position: a gentle
            # ladder spreads the (brand, series) styles across price points
            # the way a real shelf would, still anchored on the base price.
            ladder = max(len(category.brand_styles) - 1, 1)
            price = self._category_base_price(category) * (0.85 + 0.45 * model_index / ladder)
            return max(100, int(round(price / 100.0)) * 100)
        brand = BRANDS[pair_index % BRAND_COUNT]
        series_factor = 0.85 + 0.35 * ((model_index % SERIES_COUNT) % 8) / 7
        band_factor = 1 + 0.01 * (model_index % 25)
        price = self._category_base_price(category) * brand.multiplier * series_factor * band_factor
        return max(100, int(round(price / 100.0)) * 100)

    def _base_availability(self, n: int) -> int:
        digest = _stable_hash(n, "stock")
        return 0 if digest % 41 == 0 else 5 + digest % 60

    def _max_order_quantity(self, n: int) -> int:
        return 1 + _stable_hash(n, "cap") % 10

    def _describe(
        self, pair_index: int, model_index: int, variant_index: int
    ) -> tuple[str, str, dict[str, str]]:
        category = self._category_of_pair(pair_index)
        if category.brand_styles:
            # Minted naming matrix: model slots ARE the LLM-authored
            # (brand, series) styles — a shelf-like name with no synthetic
            # letter codes.
            if model_index >= len(category.brand_styles):
                raise ProviderRejectedError("该商品款式不存在")
            style = category.brand_styles[model_index]
            brand_name, series = style.brand, style.series
            name = f"{brand_name} {series} {category.name}"
        else:
            brand = BRANDS[pair_index % BRAND_COUNT]
            series = SERIES_WORDS[model_index % SERIES_COUNT]
            code = f"{series[0]}{model_index:04d}"
            brand_name = brand.name
            name = f"{brand_name} {series} {code} {category.name}"
        attributes = {
            "品牌": brand_name,
            "系列": series,
            category.variant_key: category.variant_values[variant_index],
        }
        attr_text: list[str] = []
        for ordinal, (key, vocab) in enumerate(self._category_attributes(category)):
            value = vocab[(model_index + 3 * ordinal) % len(vocab)]
            attributes[key] = value
            attr_text.append(f"{key}{value}")
        description = (
            f"模拟商品：{brand_name}{series}系列{category.name}"
            f"（{'，'.join(attr_text[:4])}），{category.description}；用于演示购物流程。"
        )
        return name, description, attributes

    def materialize(self, n: int, stock_delta: int = 0) -> CatalogItem:
        """Project the full catalog item for a SKU index."""

        if not 0 <= n < self.total_skus:
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
        start = self._category_pair_start[category_index]
        return range(start, start + self._category_pair_width[category_index])

    def _plan_tokens(self, tokens: list[str]) -> _TokenPlan:
        pair_sets: list[frozenset[int]] = []
        series_ids: set[int] = set()
        text_terms: list[str] = []
        for token in tokens:
            pair_match: set[int] = set()
            for category_index, category in enumerate(self._categories):
                aliases = (category.name, *category.provider_aliases)
                if self._token_matches(token, aliases):
                    pair_match.update(self._category_pairs(category_index))
            for brand_index, brand in enumerate(BRANDS):
                if self._token_matches(token, (brand.name, brand.slug)):
                    for category_index, category in enumerate(self._categories):
                        # Static-grid categories own one pair per brand; a
                        # brand_styles category brands its models, not its
                        # pairs, so a static-brand token cannot select it.
                        if not category.brand_styles:
                            pair_match.add(
                                self._category_pair_start[category_index] + brand_index
                            )
            for category_index, category in enumerate(self._categories):
                # LLM-authored brand names resolve at category granularity:
                # the whole (single-pair) category matches, the model loop
                # narrows down to the exact style.
                if category.brand_styles and any(
                    self._token_matches(token, (style.brand, style.series))
                    for style in category.brand_styles
                ):
                    pair_match.update(self._category_pairs(category_index))
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
        pairs = (
            frozenset.intersection(*pair_sets) if pair_sets else frozenset(range(self.pair_count))
        )
        return _TokenPlan(tuple(sorted(pairs)), frozenset(series_ids), tuple(text_terms))

    def _blob(self, pair_index: int, model_index: int, variant_index: int) -> str:
        category = self._category_of_pair(pair_index)
        name, description, attributes = self._describe(pair_index, model_index, variant_index)
        parts = [
            name,
            description,
            category.name,
            *category.provider_aliases,
            *attributes.keys(),
            *attributes.values(),
        ]
        if category.brand_styles:
            parts.extend(
                word
                for style in category.brand_styles
                for word in (style.brand, style.series)
            )
        else:
            brand = BRANDS[pair_index % BRAND_COUNT]
            parts.extend((brand.name, brand.slug))
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

    matches: list[tuple[int, int, int]] = []  # (n, price, effective availability)
    scanned = 0
    budget_exhausted = False

    for pair_index in plan.pair_indexes:
        category = catalog._category_of_pair(pair_index)
        styles = category.brand_styles
        if styles is None:
            brand = BRANDS[pair_index % BRAND_COUNT]
            if brand_filter is not None and brand.name not in brand_filter:
                continue
        # The variant axis (颜色/尺码/规格/...) filters at the variant loop;
        # every other attribute key resolves against the category vocabulary
        # and an unknown key makes the whole pair unable to satisfy it.
        variant_allowed = filters.attributes.get(category.variant_key)
        cat_checks: list[tuple[int, tuple[str, ...], frozenset[str]]] = []
        pair_possible = True
        for key, allowed in attr_filters:
            if key in {"品牌", "系列"} or key == category.variant_key:
                continue
            ordinal = next(
                (
                    i
                    for i, (attr_key, _) in enumerate(catalog._category_attributes(category))
                    if attr_key == key
                ),
                None,
            )
            if ordinal is None:
                pair_possible = False
                break
            attr_vocab = catalog._category_attributes(category)[ordinal][1]
            cat_checks.append((ordinal, attr_vocab, frozenset(allowed)))
        if not pair_possible:
            continue

        models: Iterator[int]
        if styles is not None:
            # Style-authored category: model slots ARE the (brand, series)
            # styles; static SERIES_WORDS residues do not apply and brand /
            # series filters resolve per style at the model loop.
            models = iter(range(len(styles)))
        else:
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
            if series_residues:
                models = heapq.merge(
                    *(range(r, catalog.models_per_pair, SERIES_COUNT) for r in sorted(series_residues))
                )
            else:
                models = iter(range(catalog.models_per_pair))

        for model_index in models:
            if styles is not None:
                style = styles[model_index]
                if brand_filter is not None and style.brand not in brand_filter:
                    continue
                if series_filter is not None and style.series not in series_filter:
                    continue
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
                if (
                    variant_allowed is not None
                    and category.variant_values[variant_index] not in variant_allowed
                ):
                    continue
                n = (pair_index * catalog.models_per_pair + model_index) * VARIANTS_PER_MODEL
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
