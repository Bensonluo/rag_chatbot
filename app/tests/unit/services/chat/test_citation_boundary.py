"""Public references stay bounded, exact, and subject to output privacy checks."""

import json
from typing import Any
from unittest.mock import Mock

import pytest

from app.services.chat.citations import citations_from_documents, normalize_citations
from app.services.dialogue.nodes import NodeFactory
from app.services.guardrails.base import GuardrailService
from app.services.guardrails.output_guard import DefaultOutputGuardrail


@pytest.mark.parametrize("field", ["excerpt", "title", "section_path"])
def test_citation_is_omitted_if_any_display_text_needs_redaction(field: str) -> None:
    citation: dict[str, Any] = {
        "source_type": "document",
        "document_id": "policy",
        "version_id": "revision-1",
        "title": "Refund policy",
        "excerpt": "Refunds within seven days.",
        "section_path": ["Refunds"],
        "start_pos": 12,
        "end_pos": 37,
    }
    citation[field] = (
        ["Contact audit@example.invalid"]
        if field == "section_path"
        else "Contact audit@example.invalid"
    )
    nodes = NodeFactory(
        intent_detector=Mock(),
        slot_filler=None,
        tool_registry=Mock(),
        guardrail_service=GuardrailService(output_guard=DefaultOutputGuardrail()),
    )

    assert nodes._safe_citations([citation]) == []
    # The source remains untouched; redaction must not rewrite the snapshot.
    assert "audit@example.invalid" in json.dumps(citation)


def test_safe_citation_keeps_verbatim_excerpt_and_locations() -> None:
    citation = {
        "source_type": "document",
        "document_id": "policy",
        "version_id": "revision-1",
        "title": "Refund policy",
        "excerpt": "Refunds within seven days.",
        "section_path": ["Refunds"],
        "start_pos": 12,
        "end_pos": 37,
    }
    nodes = NodeFactory(
        intent_detector=Mock(),
        slot_filler=None,
        tool_registry=Mock(),
        guardrail_service=GuardrailService(output_guard=DefaultOutputGuardrail()),
    )

    safe = nodes._safe_citations([citation])

    assert len(safe) == 1
    for key, value in citation.items():
        assert safe[0][key] == value


@pytest.mark.parametrize("source_type", [[], {}, ["graph"], {"type": "graph"}])
def test_malformed_source_type_never_breaks_citation_generation(source_type: object) -> None:
    result = citations_from_documents(
        [
            {
                "document_id": "policy",
                "content": "Original source.",
                "metadata": {"source_type": source_type},
            }
        ]
    )

    assert all(citation["source_type"] != "graph" for citation in result)


def test_graph_provenance_sets_never_invent_document_version_pairs() -> None:
    result = citations_from_documents(
        [
            {
                "document_id": "graph_synthesized_digest",
                "content": "Synthesized graph evidence.",
                "metadata": {
                    "source_type": "community_summary",
                    "source_document_ids": ["document-a", "document-b"],
                    "source_version_ids": ["revision-b"],
                    "source_chunk_ids": ["chunk-b", "chunk-a"],
                    "internal_path": "/private/index",
                    "credentials": "must-not-escape",
                },
            }
        ]
    )

    assert len(result) == 1
    citation = result[0]
    assert citation["source_type"] == "graph"
    assert "document_id" not in citation
    assert "version_id" not in citation
    assert "chunk_id" not in citation
    assert citation["source_document_ids"] == ["document-a", "document-b"]
    assert citation["source_version_ids"] == ["revision-b"]
    assert citation["source_chunk_ids"] == ["chunk-b", "chunk-a"]
    assert "internal_path" not in citation
    assert "credentials" not in citation


@pytest.mark.parametrize(
    "invalid",
    [
        {"excerpt": "x" * 801},
        {"title": "x" * 501},
        {"section_path": ["x"] * 21},
        {"section_path": ["x" * 513]},
        {"source_document_ids": ["x"] * 51},
        {"source_version_ids": ["x" * 513]},
        {"source_chunk_ids": "not-an-array"},
        {"page_start": True},
        {"page_start": 3, "page_end": 2},
        {"start_pos": 9, "end_pos": 8},
    ],
)
def test_untrusted_cache_or_history_payload_cannot_bypass_bounds(invalid: dict[str, Any]) -> None:
    assert (
        normalize_citations([{"source_type": "document", "document_id": "policy", **invalid}]) == []
    )


def test_legacy_and_malformed_citation_payloads_add_no_evidence() -> None:
    legacy_values: list[object] = [
        None,
        "policy",
        {"document_id": "policy"},
        [None, "policy", {}, 42],
    ]
    for value in legacy_values:
        assert normalize_citations(value) == []
    bounded = normalize_citations(
        [{"source_type": "document", "document_id": f"doc-{index}"} for index in range(100)]
    )
    assert len(bounded) == 20
