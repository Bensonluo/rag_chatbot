"""Deterministic validation for runtime-minted dynamic categories.

Smart understanding lives API-side (the agent's LLM proposes a category
definition); this module is the provider-side enforcement that never calls a
model: the domain must belong to the fixed frame (curated domains plus the
``general`` fallback), the name must be new after normalization, the shape
must mirror curated categories, and blocklisted goods are refused outright.

Together this makes the catalog *open at leaf level* (any everyday physical
good can be minted) while the *domain frame stays fixed* — the taxonomy map
consumers browse never restructures itself at runtime.
"""

from __future__ import annotations

import re
from typing import Any

from app.commerce_provider.catalog import VirtualCatalog
from app.commerce_provider.errors import ProviderRejectedError

# The fallback domain: legit everyday goods that fit no curated domain land
# here instead of being refused for being unclassifiable.
FALLBACK_DOMAIN = "general"

MAX_DYNAMIC_NAME_LENGTH = 32
MAX_DYNAMIC_DESCRIPTION_LENGTH = 200
MAX_DYNAMIC_ATTRIBUTES = 5
MAX_DYNAMIC_ATTRIBUTE_VALUES = 4
MAX_DYNAMIC_ATTRIBUTE_NAME_LENGTH = 16
MAX_DYNAMIC_VALUE_LENGTH = 24
MIN_BASE_PRICE_MINOR = 100  # ¥1
MAX_BASE_PRICE_MINOR = 1_000_000  # ¥10,000
# LLM-authored naming matrix: enough (brand, series) pairings to cover a
# search page across 2+ brands, capped by the model slots per pair.
MIN_BRAND_STYLES = 4
MAX_BRAND_STYLES = 8
MIN_BRANDS_IN_STYLES = 2
MAX_BRAND_OR_SERIES_LENGTH = 16

# Policy refusals: prescription/medical claims, weapons, adult content,
# virtual goods and services, living animals. Deliberately narrow so kitchen
# knives or 保健食品 are not collateral damage; the LLM-side judgement is the
# first gate, this regex is the provider's independent second gate.
_BLOCKLIST_PATTERN = re.compile(
    "|".join(
        (
            "处方",
            "药品",
            "药物",
            "疫苗",
            "处方药",
            "枪",
            "弹药",
            "子弹",
            "管制刀",
            "爆炸",
            "成人用品",
            "情色",
            "色情",
            "账号",
            "充值卡",
            "代练",
            "外挂",
            "香烟",
            "烟草",
            "活体动物",
            "器官",
            "代孕",
            "彩票",
            "博彩",
        )
    )
)


def allowed_domains(catalog: VirtualCatalog) -> frozenset[str]:
    """The fixed domain frame: curated domains plus the general fallback."""

    domains = {category.domain for category in catalog.registry_taxonomy().categories}
    domains.add(FALLBACK_DOMAIN)
    return frozenset(domains)


def _validate_brand_styles(
    brand_styles: list[dict[str, Any]] | None,
) -> list[dict[str, str]] | None:
    """Normalize the LLM-authored naming matrix, or accept ``None`` as legacy.

    Malformed proposals are refused (the mint is an explicit agent action,
    not silent best-effort); an absent matrix keeps the fallback naming
    grid so older callers and legacy rows behave exactly as before.
    """

    if brand_styles is None:
        return None
    if (
        not isinstance(brand_styles, list)
        or not MIN_BRAND_STYLES <= len(brand_styles) <= MAX_BRAND_STYLES
    ):
        raise ProviderRejectedError(
            f"品牌款式矩阵必须提供 {MIN_BRAND_STYLES}-{MAX_BRAND_STYLES} 个组合"
        )
    normalized: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for entry in brand_styles:
        if not isinstance(entry, dict):
            raise ProviderRejectedError("品牌款式矩阵每项须为 brand/series 对象")
        brand = str(entry.get("brand", "")).strip()
        series = str(entry.get("series", "")).strip()
        if (
            not brand
            or not series
            or len(brand) > MAX_BRAND_OR_SERIES_LENGTH
            or len(series) > MAX_BRAND_OR_SERIES_LENGTH
        ):
            raise ProviderRejectedError(
                f"品牌与系列名长度必须在 1-{MAX_BRAND_OR_SERIES_LENGTH} 个字符之间"
            )
        if _BLOCKLIST_PATTERN.search(brand) or _BLOCKLIST_PATTERN.search(series):
            raise ProviderRejectedError("品牌或系列名属于平台不支持的类目范围")
        pair = ("".join(brand.split()).lower(), "".join(series.split()).lower())
        if pair in seen_pairs:
            raise ProviderRejectedError(f"品牌款式组合重复: {brand} {series}")
        seen_pairs.add(pair)
        normalized.append({"brand": brand, "series": series})
    if len({item["brand"] for item in normalized}) < MIN_BRANDS_IN_STYLES:
        raise ProviderRejectedError(
            f"品牌款式矩阵必须覆盖至少 {MIN_BRANDS_IN_STYLES} 个不同品牌，避免单一品牌霸屏"
        )
    return normalized


def validate_proposal(
    *,
    name: str,
    description: str,
    domain: str,
    allowed_domains: frozenset[str],
    attributes: list[tuple[str, list[str]]],
    variant_key: str,
    variant_values: list[str],
    base_price_minor: int,
    brand_styles: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate a mint proposal and return normalized construction fields.

    Raises ``ProviderRejectedError`` on any rule violation; commits nothing.
    The sequenced ``category_id`` is reserved by the caller's transaction and
    combined with these fields to build the final ``RegistryCategory``.
    """

    name = name.strip()
    description = description.strip()
    variant_key = variant_key.strip()
    if not name or len(name) > MAX_DYNAMIC_NAME_LENGTH:
        raise ProviderRejectedError("动态类目名称长度必须在 1-32 个字符之间")
    if len(description) > MAX_DYNAMIC_DESCRIPTION_LENGTH:
        raise ProviderRejectedError("动态类目描述过长")
    if _BLOCKLIST_PATTERN.search(name) or _BLOCKLIST_PATTERN.search(description):
        raise ProviderRejectedError("该商品属于平台不支持的类目范围")
    if domain not in allowed_domains:
        raise ProviderRejectedError(f"类目域 {domain} 不在固定域框架内")
    if not variant_key or len(variant_key) > MAX_DYNAMIC_ATTRIBUTE_NAME_LENGTH:
        raise ProviderRejectedError("变体维度名称长度必须在 1-16 个字符之间")
    if len(variant_values) != 4 or any(
        (not value.strip()) or len(value) > MAX_DYNAMIC_VALUE_LENGTH for value in variant_values
    ):
        raise ProviderRejectedError("变体维度必须提供 4 个非空且不超过 24 字符的取值")
    if not MIN_BASE_PRICE_MINOR <= base_price_minor <= MAX_BASE_PRICE_MINOR:
        raise ProviderRejectedError(
            f"基准价格必须在 {MIN_BASE_PRICE_MINOR // 100}-{MAX_BASE_PRICE_MINOR // 100} 元之间"
        )
    if len({"".join(value.split()).lower() for value in variant_values}) != 4:
        raise ProviderRejectedError("变体维度取值不能重复")
    if not attributes or len(attributes) > MAX_DYNAMIC_ATTRIBUTES:
        raise ProviderRejectedError(f"属性轴数量必须在 1-{MAX_DYNAMIC_ATTRIBUTES} 之间")
    if variant_key in {attribute_name for attribute_name, _ in attributes}:
        raise ProviderRejectedError("变体维度不能与属性名重复")
    seen_attribute_names: set[str] = set()
    normalized_attributes: list[tuple[str, list[str]]] = []
    for attribute_name, values in attributes:
        attribute_name = attribute_name.strip()
        if not attribute_name or len(attribute_name) > MAX_DYNAMIC_ATTRIBUTE_NAME_LENGTH:
            raise ProviderRejectedError("属性名长度必须在 1-16 个字符之间")
        normalized = "".join(attribute_name.split()).lower()
        if normalized in seen_attribute_names:
            raise ProviderRejectedError(f"属性名重复: {attribute_name}")
        seen_attribute_names.add(normalized)
        if not values or len(values) > MAX_DYNAMIC_ATTRIBUTE_VALUES:
            raise ProviderRejectedError(
                f"属性 {attribute_name} 取值数量必须在 1-{MAX_DYNAMIC_ATTRIBUTE_VALUES} 之间"
            )
        cleaned: list[str] = []
        for value in values:
            value = value.strip()
            if not value or len(value) > MAX_DYNAMIC_VALUE_LENGTH:
                raise ProviderRejectedError(f"属性 {attribute_name} 取值无效")
            cleaned.append(value)
        if len({"".join(value.split()).lower() for value in cleaned}) != len(cleaned):
            raise ProviderRejectedError(f"属性 {attribute_name} 取值不能重复")
        normalized_attributes.append((attribute_name, cleaned))
    return {
        "name": name,
        "description": description,
        "domain": domain,
        "provider_policy": "generated",
        "status": "active",
        "provider_aliases": [],
        "base_price_minor": base_price_minor,
        "variant_key": variant_key,
        "variant_values": variant_values,
        "brand_styles": _validate_brand_styles(brand_styles),
        "attributes": [
            {"name": attribute_name, "allowed_values": values}
            for attribute_name, values in normalized_attributes
        ],
    }


__all__ = [
    "FALLBACK_DOMAIN",
    "allowed_domains",
    "validate_proposal",
]
