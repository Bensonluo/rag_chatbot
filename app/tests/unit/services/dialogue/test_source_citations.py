"""References describe material actually used for the delivered answer."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.agent.service import AgentService
from app.services.chat.answer_cache import CachedAnswer
from app.services.chat.citations import citations_from_documents, faq_citations
from app.services.dialogue.nodes import NodeFactory
from app.services.dialogue.state import DialogueState, begin_turn
from app.services.dialogue.tools import create_default_tool_registry, create_knowledge_tool
from app.services.faq.store import FAQEntry
from app.services.guardrails.base import GuardrailResult
from app.services.llm.base import LLMResponse
from app.services.retrieval.vector_base import SearchResult


def evidence() -> dict[str, Any]:
    return {
        "document_id": "policy",
        "content": "Evidence about warranty coverage.",
        "metadata": {
            "version_id": "revision-1",
            "chunk_id": "chunk-3",
            "title": "Warranty",
            "page_start": 2,
            "page_end": 3,
            "start_pos": 100,
            "end_pos": 133,
            "section_path": ["Coverage"],
        },
    }


def factory(**options: Any) -> NodeFactory:
    return NodeFactory(
        intent_detector=Mock(),
        slot_filler=None,
        tool_registry=create_default_tool_registry(),
        **options,
    )


def state() -> DialogueState:
    return {"message": "What is covered?", "intent": "question", "retrieved_docs": [evidence()]}


async def test_rag_reference_keeps_the_exact_prompt_revision_and_location() -> None:
    llm = Mock(
        generate=AsyncMock(
            return_value=LLMResponse(content="Coverage is documented.", model="test")
        )
    )
    result = await factory(llm_service=llm).generate_response_node(state())
    assert evidence()["content"] in llm.generate.await_args.args[0][-1].content
    citation = result["citations"][0]
    assert citation["document_id"] == "policy"
    assert citation["version_id"] == "revision-1"
    assert citation["chunk_id"] == "chunk-3"
    assert (citation["page_start"], citation["page_end"]) == (2, 3)
    assert (citation["start_pos"], citation["end_pos"]) == (100, 133)
    assert citation["excerpt"] == evidence()["content"]


@pytest.mark.parametrize("mode", ["missing_llm", "failure", "no_evidence", "slot_prompt"])
async def test_unanswered_or_unused_material_does_not_gain_references(mode: str) -> None:
    llm = Mock(generate=AsyncMock(side_effect=RuntimeError("provider unavailable")))
    turn = state()
    if mode == "no_evidence":
        turn["retrieved_docs"] = []
        turn["retrieval_ran"] = True
    if mode == "slot_prompt":
        turn["slot_prompt"] = "Please provide your order number."
    result = await factory(
        llm_service=None if mode == "missing_llm" else llm
    ).generate_response_node(turn)
    assert result["citations"] == []


@pytest.mark.parametrize("partial", [False, True])
async def test_failed_stream_never_attaches_completed_evidence(partial: bool) -> None:
    async def stream(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        if partial:
            yield "Some preliminary text. "
        raise RuntimeError("provider failed")

    result = await factory(llm_service=Mock(generate_stream=stream)).generate_response_node(
        state(), {"configurable": {"stream_queue": asyncio.Queue()}}
    )
    assert result["citations"] == []


async def test_output_replacement_clears_original_references() -> None:
    guardrail = Mock(check_output=Mock(return_value=GuardrailResult(action="block")))
    llm = Mock(
        generate=AsyncMock(return_value=LLMResponse(content="Original answer.", model="test"))
    )
    result = await factory(llm_service=llm, guardrail_service=guardrail).generate_response_node(
        state()
    )
    assert result["citations"] == []
    assert result["response"] != "Original answer."


def test_turn_reset_prevents_old_evidence_from_following_smalltalk() -> None:
    old: DialogueState = {"citations": citations_from_documents([evidence()])}
    reset = begin_turn(old)
    assert reset["citations"] == []
    assert old["citations"]


def test_oversized_identities_are_unknown_instead_of_truncated() -> None:
    long_id = "x" * 513
    citation = citations_from_documents(
        [
            {
                "document_id": long_id,
                "content": "Source text",
                "metadata": {
                    "version_id": long_id,
                    "chunk_id": long_id,
                    "source_document_ids": [long_id, "real-doc"],
                    "source_version_ids": [long_id, "real-version"],
                    "source_chunk_ids": [long_id, "real-chunk"],
                },
            }
        ]
    )[0]
    assert "document_id" not in citation
    assert "version_id" not in citation
    assert "chunk_id" not in citation
    assert citation["source_document_ids"] == ["real-doc"]
    assert citation["source_version_ids"] == ["real-version"]
    assert citation["source_chunk_ids"] == ["real-chunk"]
    assert "faq_id" not in faq_citations(long_id, "Question", "Answer")[0]


async def test_faq_revision_changes_with_curated_text_and_has_no_fake_document_id() -> None:
    entry = FAQEntry(faq_id="coverage", question="What is covered?", answer="Covered parts.")
    faq = Mock(match=AsyncMock(return_value=entry))
    result = await factory(faq_service=faq).faq_lookup_node({"message": entry.question})
    first = result["citations"][0]
    assert first["source_type"] == "faq"
    assert first["faq_id"] == "coverage"
    assert first["version_id"].startswith("sha256:")
    assert "document_id" not in first
    assert (
        first["version_id"]
        != faq_citations(entry.faq_id, entry.question, "Updated parts.")[0]["version_id"]
    )


@pytest.mark.parametrize("cache_name", ["answer_cache", "semantic_cache"])
async def test_cache_nodes_restore_stored_references(cache_name: str) -> None:
    citations = citations_from_documents([evidence()])
    cached = CachedAnswer(
        response="A documented answer.", sources=["policy"], intent="chitchat", citations=citations
    )
    cache = Mock(get=AsyncMock(return_value=cached), capture_epoch=AsyncMock(return_value="epoch"))
    nodes = factory(**{cache_name: cache})
    result = await (
        nodes.answer_cache_lookup_node({"message": "Coverage?"})
        if cache_name == "answer_cache"
        else nodes.direct_response_node({"message": "Coverage?"})
    )
    assert result["citations"] == citations


@pytest.mark.parametrize("finish", ["answer", "stage", "exhaust"])
async def test_agent_keeps_evidence_outside_truncated_audit_summaries(finish: str) -> None:
    doc = evidence()
    doc["content"] = "Long source evidence. " * 30
    hit = SearchResult(
        document_id=doc["document_id"], content=doc["content"], score=0.9, metadata=doc["metadata"]
    )
    registry = create_default_tool_registry()
    registry.register(create_knowledge_tool(Mock(search=AsyncMock(return_value=[hit]))))
    first = LLMResponse(
        content="",
        model="test",
        tool_calls=[
            {
                "id": "call-1",
                "name": "search_knowledge_base",
                "arguments": json.dumps({"query": "coverage"}),
            }
        ],
    )
    if finish == "stage":
        second = LLMResponse(
            content="",
            model="test",
            tool_calls=[
                {
                    "id": "call-2",
                    "name": "process_refund",
                    "arguments": json.dumps({"order_id": "ORD1001", "reason": "faulty"}),
                }
            ],
        )
    elif finish == "exhaust":
        second = first
    else:
        second = LLMResponse(content="Here is the documented answer.", model="test")
    llm = Mock(generate_with_tools=AsyncMock(side_effect=[first, second]))
    agent = AgentService(
        llm_service=llm, tool_registry=registry, max_steps=2 if finish == "exhaust" else 3
    )
    result = await agent.run("Warranty coverage?", user_id=1)
    # IDs at the end of tool output cannot be recovered from its 200-char audit summary.
    assert "revision-1" not in result.tool_trace[0]["summary"]
    if finish == "answer":
        assert result.citations[0]["version_id"] == "revision-1"
        payload = json.loads(llm.generate_with_tools.await_args_list[1].args[0][-1].content)
        assert payload["citations"] == result.citations
        nodes = factory(agent_service=Mock(run=AsyncMock(return_value=result)))
        updates = await nodes.handle_agent_node({"message": "Warranty coverage?"})
        assert updates["citations"] == result.citations
        assert updates["sources"] == ["policy"]
    else:
        assert result.citations == []
