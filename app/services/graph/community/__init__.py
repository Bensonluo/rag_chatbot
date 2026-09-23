"""Community detection and summarization services."""

from typing import Any

from app.services.graph.community.detection import CommunityDetectionService
from app.services.graph.community.global_search import GlobalSearchService

__all__ = ["CommunityDetectionService", "GlobalSearchService"]


def __getattr__(name: str) -> Any:
    if name == "CommunitySummarizationService":
        from app.services.graph.community.summarization import CommunitySummarizationService

        return CommunitySummarizationService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
