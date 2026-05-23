"""
Domain schema definition for the knowledge graph.

Defines entity types, relationship types, and their properties for the
intelligent customer service domain.
"""

ENTITY_TYPES: dict[str, dict[str, str | list[str]]] = {
    "Product": {
        "description": "A product or service offered by the company",
        "properties": [
            "product_name", "version", "product_type", "release_date",
        ],
    },
    "Feature": {
        "description": "A feature or capability of a product",
        "properties": ["feature_name", "feature_type", "status"],
    },
    "Issue": {
        "description": "A customer-reported problem or bug",
        "properties": ["symptom", "severity", "frequency", "error_code"],
    },
    "Solution": {
        "description": "A troubleshooting step or resolution for an issue",
        "properties": ["step", "solution_type", "difficulty"],
    },
    "FAQ": {
        "description": "A frequently asked question and its answer",
        "properties": ["question", "answer", "category"],
    },
    "Category": {
        "description": "A product or issue classification",
        "properties": ["category_name", "level", "parent_category"],
    },
    "Platform": {
        "description": "A platform or operating environment",
        "properties": ["platform_name", "version", "os_type"],
    },
}

RELATION_TYPES: dict[str, dict[str, str]] = {
    "HAS_FEATURE": {
        "source": "Product",
        "target": "Feature",
        "description": "Product has this feature",
    },
    "REPORTS": {
        "source": "Issue",
        "target": "Product",
        "description": "Issue is reported for this product",
    },
    "RESOLVES": {
        "source": "Solution",
        "target": "Issue",
        "description": "Solution resolves this issue",
    },
    "BELONGS_TO": {
        "source": "Product",
        "target": "Category",
        "description": "Product belongs to this category",
    },
    "APPLIES_TO": {
        "source": "Solution",
        "target": "Platform",
        "description": "Solution applies to this platform",
    },
    "RELATED_ISSUE": {
        "source": "Issue",
        "target": "Issue",
        "description": "Issues are related or share common cause",
    },
    "ANSWERS": {
        "source": "FAQ",
        "target": "Product",
        "description": "FAQ answers question about this product",
    },
    "AFFECTS": {
        "source": "Issue",
        "target": "Platform",
        "description": "Issue affects this platform",
    },
    "REQUIRES": {
        "source": "Solution",
        "target": "Feature",
        "description": "Solution requires this feature to be enabled",
    },
    "SIMILAR_TO": {
        "source": "Product",
        "target": "Product",
        "description": "Products are similar or share functionality",
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
