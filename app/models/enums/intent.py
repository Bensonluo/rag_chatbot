"""Intent-related enums"""
from enum import Enum


class Intent(str, Enum):
    """Intent categories for user queries"""
    QUESTION = "question"
    COMPARISON = "comparison"
    HOW_TO = "how_to"
    DEFINITION = "definition"
    RECOMMENDATION = "recommendation"
    SUMMARY = "summary"
    CODE_HELP = "code_help"
    CREATIVE = "creative"
    CHITCHAT = "chitchat"
    TASK = "task"
    UNKNOWN = "unknown"
    RELATIONSHIP_QUERY = "relationship_query"
    GLOBAL_SUMMARY = "global_summary"
    ENTITY_LOOKUP = "entity_lookup"
