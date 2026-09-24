"""LangGraph node config-injection contract.

langgraph 1.x inspects node signatures and only injects ``config`` when the
parameter annotation — possibly stringified by ``from __future__ import
annotations`` — matches an accepted spelling. A node outside that set runs
with ``config=None``, which silently drops the per-request ``stream_queue``:
SSE content tokens vanish while trace events (ContextVar-based, no config)
keep flowing. That exact asymmetry shipped unnoticed because unit tests call
nodes directly and integration tests mock the graph.

This pins every graph-registered node against the regression.
"""

import inspect
from typing import Optional

from langchain_core.runnables import RunnableConfig

from app.services.dialogue.nodes import NodeFactory

# Mirror of langgraph._internal._runnable.KWARGS_CONFIG_KEYS for "config".
ACCEPTED_CONFIG_ANNOTATIONS = (
    RunnableConfig,
    "RunnableConfig",
    Optional[RunnableConfig],
    "Optional[RunnableConfig]",
    inspect.Parameter.empty,
)

# The 13 nodes registered in app/services/dialogue/graph.py::create_dialogue_graph.
GRAPH_NODE_METHODS = (
    "guardrail_node",
    "answer_cache_lookup_node",
    "detect_intent_node",
    "handle_switch_node",
    "route_intent_node",
    "collect_slots_node",
    "execute_tool_node",
    "rag_lookup_node",
    "generate_response_node",
    "direct_response_node",
    "handle_handoff_node",
    "handle_agent_node",
    "faq_lookup_node",
)

# Nodes that emit response content onto the per-request stream queue —
# they cannot do their job without an injected config.
NODES_REQUIRING_CONFIG = (
    "execute_tool_node",
    "answer_cache_lookup_node",
    "faq_lookup_node",
    "generate_response_node",
    "direct_response_node",
    "handle_handoff_node",
    "handle_agent_node",
)


class TestNodeConfigInjection:
    def test_every_graph_node_config_annotation_is_injectable(self):
        """Any node declaring a config param must stringify to a form
        langgraph accepts, or it silently runs with config=None."""
        rejected = []
        for name in GRAPH_NODE_METHODS:
            fn = getattr(NodeFactory, name, None)
            assert fn is not None, f"graph node method missing: {name}"
            # inspect.signature follows the traced_stage __wrapped__ chain.
            params = inspect.signature(fn).parameters
            if "config" not in params:
                continue
            annotation = params["config"].annotation
            if annotation not in ACCEPTED_CONFIG_ANNOTATIONS:
                rejected.append((name, repr(annotation)))
        assert not rejected, (
            "Nodes whose config annotation langgraph will NOT inject "
            f"(config becomes None, stream queue lost): {rejected}"
        )

    def test_streaming_nodes_accept_config(self):
        """Content-emitting nodes must take config, or they lose the
        stream queue regardless of annotation."""
        for name in NODES_REQUIRING_CONFIG:
            fn = getattr(NodeFactory, name, None)
            assert fn is not None, f"graph node method missing: {name}"
            params = inspect.signature(fn).parameters
            assert "config" in params, f"{name} must accept config to reach the stream queue"
