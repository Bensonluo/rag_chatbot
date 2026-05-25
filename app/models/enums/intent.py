"""Business intent enums for customer service dialogue management."""
from enum import Enum


class Intent(str, Enum):
    """Business intent categories aligned with customer service operations."""

    # Task-oriented intents (→ Function Calling)
    REFUND = "refund"
    RETURN = "return"
    QUERY_ORDER = "query_order"
    TRACK_SHIPPING = "track_shipping"
    COMPLAINT = "complaint"

    # Knowledge intents (→ RAG)
    FAQ = "faq"
    POLICY = "policy"

    # Dialogue intents (→ Direct response)
    CHITCHAT = "chitchat"
    GREETING = "greeting"

    # Meta intents (→ State action)
    CONFIRM = "confirm"
    DENY = "deny"
    CANCEL = "cancel"
    UNKNOWN = "unknown"

    # Graph-related (→ GraphRAG, legacy support)
    RELATIONSHIP_QUERY = "relationship_query"
    GLOBAL_SUMMARY = "global_summary"
    ENTITY_LOOKUP = "entity_lookup"


# Routing groups
TASK_INTENTS = {"refund", "return", "query_order", "track_shipping", "complaint"}
RAG_INTENTS = {"faq", "policy"}
DIRECT_INTENTS = {"chitchat", "greeting"}
META_INTENTS = {"confirm", "deny", "cancel"}
GRAPH_INTENTS = {"relationship_query", "global_summary", "entity_lookup"}

# Display names for Chinese UI
INTENT_DISPLAY_NAMES = {
    "refund": "退款",
    "return": "退货",
    "query_order": "查订单",
    "track_shipping": "查物流",
    "complaint": "投诉",
    "faq": "常见问题",
    "policy": "政策查询",
}
