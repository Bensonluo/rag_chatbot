"""
Domain schema definition for the knowledge graph.

Defines entity types, relationship types, and their properties for the
pharmaceutical market analysis domain.
"""

ENTITY_TYPES: dict[str, dict[str, str | list[str]]] = {
    "Drug": {
        "description": "A pharmaceutical product",
        "properties": [
            "generic_name", "brand_name", "approval_date",
            "therapeutic_class", "dosage_form",
        ],
    },
    "Company": {
        "description": "A pharmaceutical company",
        "properties": ["headquarters", "company_type", "stock_ticker"],
    },
    "Channel": {
        "description": "A sales or distribution channel",
        "properties": ["channel_type"],
    },
    "Region": {
        "description": "A geographic market region",
        "properties": ["country", "province", "city"],
    },
    "Indication": {
        "description": "A disease or medical condition",
        "properties": ["icd_code", "specialty"],
    },
    "ActiveIngredient": {
        "description": "An active pharmaceutical ingredient (API)",
        "properties": ["molecular_formula", "mechanism_of_action"],
    },
    "SalesMetric": {
        "description": "A sales data point (revenue, volume, etc.)",
        "properties": ["revenue", "volume", "growth_rate", "period", "currency"],
    },
    "TimePeriod": {
        "description": "A time reference (year, quarter, month)",
        "properties": ["year", "quarter", "month"],
    },
}

RELATION_TYPES: dict[str, dict[str, str]] = {
    "MANUFACTURES": {
        "source": "Company",
        "target": "Drug",
        "description": "Company manufactures this drug",
    },
    "SOLD_IN": {
        "source": "Drug",
        "target": "Channel",
        "description": "Drug is sold through this channel",
    },
    "SOLD_IN_REGION": {
        "source": "Drug",
        "target": "Region",
        "description": "Drug is available in this region",
    },
    "TREATS": {
        "source": "Drug",
        "target": "Indication",
        "description": "Drug is used to treat this condition",
    },
    "CONTAINS": {
        "source": "Drug",
        "target": "ActiveIngredient",
        "description": "Drug contains this active ingredient",
    },
    "HAS_SALES": {
        "source": "Drug",
        "target": "SalesMetric",
        "description": "Drug has this sales data",
    },
    "DURING": {
        "source": "SalesMetric",
        "target": "TimePeriod",
        "description": "Sales data is for this time period",
    },
    "COMPETES_WITH": {
        "source": "Drug",
        "target": "Drug",
        "description": "Drugs compete in the same therapeutic market",
    },
    "OPERATES_IN": {
        "source": "Company",
        "target": "Region",
        "description": "Company operates in this region",
    },
    "MARKET_OPPORTUNITY": {
        "source": "Drug",
        "target": "Region",
        "description": "Potential market opportunity for drug in region",
    },
}


def get_schema_prompt_text() -> str:
    """Format the schema as text suitable for LLM prompts."""
    lines = ["Entity Types:"]

    for name, info in ENTITY_TYPES.items():
        props = ", ".join(info["properties"])  # type: ignore[union-attr]
        lines.append(f"  {name}: {info['description']}. Properties: [{props}]")

    lines.append("\nRelationship Types:")
    for name, info in RELATION_TYPES.items():
        lines.append(
            f"  ({info['source']})-[{name}]->({info['target']}): "
            f"{info['description']}"
        )

    return "\n".join(lines)
