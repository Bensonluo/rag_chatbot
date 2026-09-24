"""Pyramid-funnel layer telemetry (per-turn serving layer).

The north-star funnel story requires knowing which layer actually
served each turn: the L0 exact-answer cache and FAQ absorb the hot
top, RAG grounds policy answers, the agent tool loop serves the deep
"real need" traffic, and handoff is the deliberate exit to humans.
One labeled counter per layer makes the funnel distribution — and
the one-shot-resolution proxy (served turns vs handoff turns) —
computable in a single PromQL query. Layer touches are counted, not
exclusive: an agent turn that also retrieves touches both, which is
the honest measure of layer pressure.
"""

from prometheus_client import REGISTRY, Counter

FUNNEL_LAYERS = Counter(
    "chat_funnel_layers_total",
    "Chat turns by pyramid-funnel layer that served or engaged them",
    labelnames=["layer"],
    registry=REGISTRY,
)

# Canonical layer names — renaming breaks dashboards and queries.
LAYER_L0_CACHE = "l0_cache"
LAYER_L1_SEMANTIC = "l1_semantic"
LAYER_FAQ = "faq"
LAYER_RAG = "rag"
LAYER_AGENT_TOOL = "agent_tool"
LAYER_HANDOFF = "handoff"
LAYER_DIRECT = "direct"


def record_funnel_layer(layer: str) -> None:
    """Record that a funnel layer served (or engaged in) one chat turn."""
    FUNNEL_LAYERS.labels(layer=layer).inc()
