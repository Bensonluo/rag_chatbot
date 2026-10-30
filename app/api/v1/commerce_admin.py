"""Admin operations for commerce taxonomy and coverage governance."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import require_admin
from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.models.database.commerce_taxonomy import CommerceCoverageEvent, CommerceCoverageReview
from app.models.database.user import User
from app.models.schemas.commerce_taxonomy import (
    CoverageEventResponse,
    CoverageMetricsResponse,
    CoverageReviewCreateRequest,
    CoverageReviewDecisionRequest,
    CoverageReviewResponse,
    ExternalCategoryMappingRequest,
    RegistryTaxonomy,
    TaxonomyVersionCreateRequest,
    TaxonomyVersionResponse,
)
from app.services.commerce.taxonomy import CommerceTaxonomyRegistry

router = APIRouter(prefix="/commerce/admin", tags=["commerce-admin"])
logger = logging.getLogger(__name__)


def get_taxonomy_registry() -> CommerceTaxonomyRegistry:
    from app.api.database import async_session_maker

    return CommerceTaxonomyRegistry(async_session_maker)


def _taxonomy_error(exc: Exception) -> HTTPException:
    if isinstance(exc, NotFoundError):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(exc) or "Not found")
    if isinstance(exc, ConflictError):
        return HTTPException(status.HTTP_409_CONFLICT, str(exc) or "Conflict")
    if isinstance(exc, ValidationError):
        return HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    return HTTPException(
        status.HTTP_500_INTERNAL_SERVER_ERROR, "Commerce taxonomy operation failed"
    )


async def _refresh_commerce_projection(registry: CommerceTaxonomyRegistry) -> None:
    """Best-effort graph refresh after taxonomy or semantic changes."""

    try:
        from app.api.v1.chat import get_chat_service

        chat_service = get_chat_service()
        graph_service = getattr(chat_service, "commerce_graph_service", None)
        if graph_service is None:
            return
        taxonomy = await registry.load_active()
        semantic_data = await registry.load_semantics()
        await graph_service.refresh_projection(taxonomy, semantic_data)
    except Exception as exc:  # pragma: no cover - depends on optional GraphRAG startup
        logger.warning("Commerce graph projection refresh failed: %s", exc)


def _event_response(event: CommerceCoverageEvent) -> CoverageEventResponse:
    return CoverageEventResponse(
        id=event.id,
        query=event.query,
        normalized_query=event.normalized_query,
        owner_hash=event.owner_hash,
        matched_category_ids=list(event.matched_category_ids),
        nearest_category_ids=list(event.nearest_category_ids),
        result_count=event.result_count,
        status=event.status,
        created_at=event.created_at,
    )


def _review_response(review: CommerceCoverageReview) -> CoverageReviewResponse:
    return CoverageReviewResponse(
        id=review.id,
        event_id=review.event_id,
        proposed_type=review.proposed_type,
        proposal_payload=dict(review.proposal_payload),
        status=review.status,
        reviewer=review.reviewer,
        review_note=review.review_note,
    )


@router.get("/taxonomy/versions", response_model=list[TaxonomyVersionResponse])
async def list_taxonomy_versions(
    admin: Annotated[User, Depends(require_admin)],  # noqa: ARG001
    registry: Annotated[CommerceTaxonomyRegistry, Depends(get_taxonomy_registry)],
) -> list[TaxonomyVersionResponse]:
    versions = await registry.list_versions()
    summaries = [await registry.version_summary(version.id) for version in versions]
    return [TaxonomyVersionResponse.model_validate(summary) for summary in summaries]


@router.post("/taxonomy/versions", response_model=TaxonomyVersionResponse, status_code=201)
async def create_taxonomy_version(
    request: TaxonomyVersionCreateRequest,
    admin: Annotated[User, Depends(require_admin)],  # noqa: ARG001
    registry: Annotated[CommerceTaxonomyRegistry, Depends(get_taxonomy_registry)],
) -> TaxonomyVersionResponse:
    try:
        await registry.publish(
            version_id=request.version_id,
            taxonomy=request.taxonomy,
            activate=request.activate,
        )
        summary = await registry.version_summary(request.version_id)
        if request.activate:
            await _refresh_commerce_projection(registry)
        return TaxonomyVersionResponse.model_validate(summary)
    except Exception as exc:
        raise _taxonomy_error(exc) from exc


@router.post("/taxonomy/versions/{version_id}/activate", response_model=TaxonomyVersionResponse)
async def activate_taxonomy_version(
    version_id: str,
    admin: Annotated[User, Depends(require_admin)],  # noqa: ARG001
    registry: Annotated[CommerceTaxonomyRegistry, Depends(get_taxonomy_registry)],
) -> TaxonomyVersionResponse:
    try:
        await registry.activate(version_id=version_id)
        summary = await registry.version_summary(version_id)
        await _refresh_commerce_projection(registry)
        return TaxonomyVersionResponse.model_validate(summary)
    except Exception as exc:
        raise _taxonomy_error(exc) from exc


@router.get("/taxonomy/export", response_model=RegistryTaxonomy)
async def export_active_taxonomy(
    admin: Annotated[User, Depends(require_admin)],  # noqa: ARG001
    registry: Annotated[CommerceTaxonomyRegistry, Depends(get_taxonomy_registry)],
) -> RegistryTaxonomy:
    """Export the active registry snapshot for generated-provider startup."""

    return await registry.load_active()


@router.post("/taxonomy/external-mappings", status_code=201)
async def create_external_mapping(
    request: ExternalCategoryMappingRequest,
    admin: Annotated[User, Depends(require_admin)],  # noqa: ARG001
    registry: Annotated[CommerceTaxonomyRegistry, Depends(get_taxonomy_registry)],
) -> dict[str, str]:
    try:
        await registry.add_external_mapping(
            provider=request.provider,
            internal_category_key=request.internal_category_key,
            external_category_id=request.external_category_id,
            external_path=request.external_path,
            confidence=request.confidence,
            status=request.status,
        )
        return {"status": "created"}
    except Exception as exc:
        raise _taxonomy_error(exc) from exc


@router.get("/coverage/events", response_model=list[CoverageEventResponse])
async def list_coverage_events(
    admin: Annotated[User, Depends(require_admin)],  # noqa: ARG001
    registry: Annotated[CommerceTaxonomyRegistry, Depends(get_taxonomy_registry)],
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[CoverageEventResponse]:
    events = await registry.list_coverage_events(status=status_filter, limit=limit)
    return [_event_response(event) for event in events]


@router.post("/coverage/reviews", response_model=CoverageReviewResponse, status_code=201)
async def create_coverage_review(
    request: CoverageReviewCreateRequest,
    admin: Annotated[User, Depends(require_admin)],  # noqa: ARG001
    registry: Annotated[CommerceTaxonomyRegistry, Depends(get_taxonomy_registry)],
) -> CoverageReviewResponse:
    try:
        review = await registry.create_review(
            event_id=request.event_id,
            proposed_type=request.proposed_type,
            proposal_payload=request.proposal_payload,
        )
        return _review_response(review)
    except Exception as exc:
        raise _taxonomy_error(exc) from exc


@router.get("/coverage/reviews", response_model=list[CoverageReviewResponse])
async def list_coverage_reviews(
    admin: Annotated[User, Depends(require_admin)],  # noqa: ARG001
    registry: Annotated[CommerceTaxonomyRegistry, Depends(get_taxonomy_registry)],
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[CoverageReviewResponse]:
    reviews = await registry.list_reviews(status=status_filter, limit=limit)
    return [_review_response(review) for review in reviews]


@router.post("/coverage/reviews/{review_id}/approve", response_model=CoverageReviewResponse)
async def approve_coverage_review(
    review_id: int,
    request: CoverageReviewDecisionRequest,
    admin: Annotated[User, Depends(require_admin)],  # noqa: ARG001
    registry: Annotated[CommerceTaxonomyRegistry, Depends(get_taxonomy_registry)],
) -> CoverageReviewResponse:
    try:
        await registry.decide_review(
            review_id=review_id,
            approve=True,
            reviewer=request.reviewer,
            review_note=request.review_note,
        )
        await _refresh_commerce_projection(registry)
        reviews = await registry.list_reviews(limit=500)
        review = next((item for item in reviews if item.id == review_id), None)
        if review is None:
            raise NotFoundError("Coverage review")
        return _review_response(review)
    except Exception as exc:
        raise _taxonomy_error(exc) from exc


@router.post("/coverage/reviews/{review_id}/reject", response_model=CoverageReviewResponse)
async def reject_coverage_review(
    review_id: int,
    request: CoverageReviewDecisionRequest,
    admin: Annotated[User, Depends(require_admin)],  # noqa: ARG001
    registry: Annotated[CommerceTaxonomyRegistry, Depends(get_taxonomy_registry)],
) -> CoverageReviewResponse:
    try:
        await registry.decide_review(
            review_id=review_id,
            approve=False,
            reviewer=request.reviewer,
            review_note=request.review_note,
        )
        reviews = await registry.list_reviews(limit=500)
        review = next((item for item in reviews if item.id == review_id), None)
        if review is None:
            raise NotFoundError("Coverage review")
        return _review_response(review)
    except Exception as exc:
        raise _taxonomy_error(exc) from exc


@router.get("/coverage/metrics", response_model=CoverageMetricsResponse)
async def coverage_metrics(
    admin: Annotated[User, Depends(require_admin)],  # noqa: ARG001
    registry: Annotated[CommerceTaxonomyRegistry, Depends(get_taxonomy_registry)],
) -> CoverageMetricsResponse:
    metrics = await registry.coverage_metrics()
    return CoverageMetricsResponse.model_validate(metrics)
