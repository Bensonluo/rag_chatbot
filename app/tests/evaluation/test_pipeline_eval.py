"""Real graph/SQLite/BM25 smoke gate and adversarial observation scoring."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import create_async_engine

from app.models.database.session import ChatSession
from app.models.database.ticket import HandoffTicket
from app.services.chat.citations import citations_from_documents
from app.services.dialogue import nodes
from app.services.evaluation import pipeline_eval
from app.services.evaluation.pipeline_eval import (
    PipelineEvalReport,
    RecordedPipelineLLM,
    load_pipeline_cases,
    run_pipeline_eval,
    score_pipeline_observation,
)
from app.services.handoff.service import HandoffService
from app.services.llm.base import LLMMessage, LLMResponse


@pytest.fixture
async def report() -> PipelineEvalReport:
    return await run_pipeline_eval()


async def test_golden_real_graph_database_and_bm25(report: PipelineEvalReport) -> None:
    assert report.passed, [
        (item.observation.case_id, item.routing_failures, item.artifact_failures)
        for item in report.results
        if not item.passed
    ]
    assert report.total_turns == 14
    assert report.passed_turns == 14
    assert report.model_mode == "recorded_fixture"
    assert report.semantic_quality == "not_evaluated"
    assert report.retrieval_mode == "local_production_bm25_no_embeddings"
    assert {item.observation.language for item in report.results} == {"zh", "en"}
    for result in report.results:
        obs = result.observation
        assert len(obs.persisted_messages) == 2 * obs.turn_index
        if obs.handoff_outcome == "queued":
            assert obs.ticket_ids == [1]
            assert obs.persisted_messages[-1]["metadata"]["handoff_outcome"] == "queued"
        elif obs.handoff_outcome == "failed":
            assert obs.ticket_ids == []
            assert obs.persisted_messages[-1]["metadata"]["handoff_outcome"] == "failed"
        elif obs.citations:
            assert obs.evidence[0]["content"] == obs.citations[0]["excerpt"]
            assert obs.evidence[0]["score"] > 0
            assert obs.citations[0]["page_start"] == 2
        if obs.category == "multi_turn" and obs.turn_index == 3:
            assert obs.handoff_outcome is None
            assert "handoff_outcome" not in obs.metadata
            assert obs.ticket_ids == [1]  # persisted ticket survives switching back to RAG
            assert obs.history_messages[1]["citations"] == obs.citations


@pytest.mark.parametrize(
    "mutation,expected_failure",
    [
        ("wrong_version", "citation_version_mismatch"),
        ("invented_citation", "citation_evidence_mismatch"),
        ("empty_excerpt", "empty_citation_evidence"),
        ("persisted_citation", "persisted_citation_mismatch"),
        ("lost_history", "history_roundtrip_mismatch"),
        ("lost_database_history", "persisted_history_mismatch"),
    ],
)
async def test_rejects_corrupted_citation_and_history_contracts(
    report: PipelineEvalReport, mutation: str, expected_failure: str
) -> None:
    obs = deepcopy(report.results[0].observation)
    if mutation == "wrong_version":
        obs.citations[0]["version_id"] = "old-v1"
        obs.evidence[0]["metadata"]["version_id"] = "old-v1"
    elif mutation == "invented_citation":
        obs.evidence = []
    elif mutation == "empty_excerpt":
        obs.citations[0]["excerpt"] = ""
        obs.evidence[0]["content"] = ""
    elif mutation == "persisted_citation":
        obs.persisted_messages[-1]["metadata"]["citations"] = []
    elif mutation == "lost_history":
        obs.history_messages = []
    else:
        obs.persisted_messages[0]["content"] = "another customer's message"
    result = score_pipeline_observation(obs, load_pipeline_cases()[0].turns[0])
    assert not result.passed
    assert expected_failure in result.artifact_failures


@pytest.mark.parametrize(
    "language,promise",
    [
        ("zh", "已为您转接人工客服，工单号 #99。"),
        ("en", "You have been transferred and an agent will respond immediately."),
    ],
)
async def test_failed_ticket_cannot_claim_success_even_after_disclaimer(
    report: PipelineEvalReport, language: str, promise: str
) -> None:
    case = next(c for c in load_pipeline_cases() if c.id == f"handoff-failure-{language}")
    obs = deepcopy(next(r.observation for r in report.results if r.observation.case_id == case.id))
    obs.answer += promise
    result = score_pipeline_observation(obs, case.turns[0])
    assert "failed_handoff_false_promise" in result.artifact_failures
    assert not result.passed


async def test_cannot_claim_queued_without_a_database_ticket(report: PipelineEvalReport) -> None:
    case = next(c for c in load_pipeline_cases() if c.id == "handoff-success-en")
    obs = deepcopy(next(r.observation for r in report.results if r.observation.case_id == case.id))
    obs.ticket_ids = []
    assert (
        "queued_handoff_missing_ticket"
        in score_pipeline_observation(obs, case.turns[0]).artifact_failures
    )


async def test_injected_model_is_called_but_semantic_correctness_is_not_faked() -> None:
    class BadAnswerLLM(RecordedPipelineLLM):
        async def generate(
            self,
            messages: list[LLMMessage],
            max_tokens: int | None = None,
            temperature: float | None = None,
            **kwargs: Any,
        ) -> LLMResponse:
            await super().generate(messages, max_tokens, temperature, **kwargs)
            return LLMResponse(content="The moon is made of cheese.", model=self.model)

    llm = BadAnswerLLM()
    result = await run_pipeline_eval(llm, [load_pipeline_cases()[0]])
    assert llm.prompts
    assert result.results[0].observation.answer == "The moon is made of cheese."
    assert result.results[0].observation.required_points
    assert result.semantic_quality == "not_evaluated"
    # This is deliberately an artifact gate: an independent judge must reject meaning.
    assert result.passed


async def test_only_temporary_sqlite_is_used_and_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []

    def engine_factory(url: str) -> Any:
        urls.append(url)
        assert url.startswith("sqlite+aiosqlite:///")
        assert "rag-pipeline-eval-" in url
        return create_async_engine(url)

    monkeypatch.setattr(pipeline_eval, "create_async_engine", engine_factory)
    await run_pipeline_eval(cases=load_pipeline_cases()[:2])
    assert len(set(urls)) == 2
    assert all(not Path(url.removeprefix("sqlite+aiosqlite:///")).exists() for url in urls)


async def test_empty_run_is_not_green() -> None:
    with pytest.raises(ValueError, match="at least one"):
        await run_pipeline_eval(cases=[])


def test_loader_rejects_empty_golden(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text('{"schema_version":1,"scope":"test","cases": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty"):
        load_pipeline_cases(path)


async def test_no_evidence_requires_abstention_without_fabricated_references(
    report: PipelineEvalReport,
) -> None:
    case = next(c for c in load_pipeline_cases() if c.id == "empty-en")
    obs = next(r.observation for r in report.results if r.observation.case_id == case.id)
    bad = replace(obs, answer="We guarantee lunar delivery tomorrow.")
    result = score_pipeline_observation(bad, case.turns[0])
    assert "no_evidence_answer_contract" in result.artifact_failures


async def test_multiturn_generation_receives_actual_persisted_history() -> None:
    llm = RecordedPipelineLLM()
    case = next(c for c in load_pipeline_cases() if c.id == "switch-en")
    report = await run_pipeline_eval(llm, [case])
    assert report.passed
    last_prompt = llm.prompts[-1]
    prior_answer = report.results[0].observation.answer
    prior_handoff = report.results[1].observation.answer
    assert any(
        message.role == "assistant" and message.content == prior_answer for message in last_prompt
    )
    assert any(
        message.role == "assistant" and message.content == prior_handoff for message in last_prompt
    )
    assert report.results[-1].observation.evidence


async def test_temporary_database_is_removed_when_graph_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urls: list[str] = []

    def engine_factory(url: str) -> Any:
        urls.append(url)
        return create_async_engine(url)

    def graph_failure(**_kwargs: Any) -> None:
        raise RuntimeError("evaluation graph setup failure")

    monkeypatch.setattr(pipeline_eval, "create_async_engine", engine_factory)
    monkeypatch.setattr(pipeline_eval, "build_dialogue_graph", graph_failure)
    with pytest.raises(RuntimeError, match="evaluation graph setup failure"):
        await run_pipeline_eval(cases=load_pipeline_cases()[:1])
    assert urls
    assert all(not Path(url.removeprefix("sqlite+aiosqlite:///")).exists() for url in urls)


@pytest.mark.parametrize("corruption", ["resolved", "wrong_session", "wrong_user"])
async def test_real_ticket_mutation_cannot_pass_as_queued(
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    class CorruptingHandoff(HandoffService):
        async def create_ticket_for_session(
            self,
            session_id: int,
            user_id: int | None,
            reason: str,
            context: dict[str, Any],
        ) -> dict[str, Any]:
            result = await super().create_ticket_for_session(session_id, user_id, reason, context)
            async with self._session_maker() as session:
                values: dict[str, Any]
                if corruption == "wrong_session":
                    session.add(ChatSession(id=2, user_id=1))
                    await session.flush()
                    values = {"session_id": 2}
                elif corruption == "wrong_user":
                    values = {"user_id": 2}
                else:
                    values = {"status": "resolved"}
                await session.execute(update(HandoffTicket).values(**values))
                await session.commit()
            return result

    monkeypatch.setattr(pipeline_eval, "HandoffService", CorruptingHandoff)
    case = next(c for c in load_pipeline_cases() if c.id == "handoff-success-en")
    report = await run_pipeline_eval(cases=[case])
    assert not report.passed
    result = report.results[0]
    assert result.observation.ticket_ids == [1]
    assert result.observation.ticket_records
    assert "queued_handoff_invalid_ticket" in result.artifact_failures


@pytest.mark.parametrize(
    "key,value",
    [
        ("document_id", "another-document"),
        ("chunk_id", "wrong-chunk"),
        ("excerpt", "invented evidence"),
        ("page_start", 1),
    ],
)
async def test_shared_citation_converter_regression_cannot_corrupt_scoring_oracle(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: object,
) -> None:
    original = citations_from_documents

    def corrupt_converter(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        citations = original(documents)
        for citation in citations:
            citation[key] = value
        return citations

    monkeypatch.setattr(nodes, "citations_from_documents", corrupt_converter)
    monkeypatch.setattr(pipeline_eval, "citations_from_documents", corrupt_converter)
    report = await run_pipeline_eval(cases=load_pipeline_cases()[:1])
    assert not report.passed
    assert "citation_source_identity_mismatch" in report.results[0].artifact_failures


@pytest.mark.parametrize(
    "mutation",
    [
        "string_boolean",
        "integer_boolean",
        "version_string",
        "string_point",
        "null_documents",
        "wrong_document_content",
        "wrong_page",
        "duplicate_chunk",
        "unknown_root",
        "unknown_case",
        "unknown_turn",
        "unknown_metadata",
        "duplicate_case",
    ],
)
def test_loader_rejects_malformed_goldens(tmp_path: Path, mutation: str) -> None:
    raw = json.loads(pipeline_eval.PIPELINE_GOLDEN_SET_FILE.read_text(encoding="utf-8"))
    case = raw["cases"][0]
    turn = case["turns"][0]
    if mutation == "string_boolean":
        case["handoff_failure"] = "false"
    elif mutation == "integer_boolean":
        turn["abstention"] = 1
    elif mutation == "version_string":
        turn["expected_version_ids"] = "v2"
    elif mutation == "string_point":
        turn["required_points"] = [42]
    elif mutation == "null_documents":
        case["documents"] = None
    elif mutation == "wrong_document_content":
        case["documents"][0]["content"] = 42
    elif mutation == "wrong_page":
        case["documents"][0]["metadata"]["page_start"] = True
    elif mutation == "duplicate_chunk":
        case["documents"].append(deepcopy(case["documents"][0]))
    elif mutation == "unknown_root":
        raw["unexpected"] = True
    elif mutation == "unknown_case":
        case["unexpected"] = True
    elif mutation == "unknown_turn":
        turn["unexpected"] = True
    elif mutation == "unknown_metadata":
        case["documents"][0]["metadata"]["unexpected"] = True
    else:
        raw["cases"].append(deepcopy(case))
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_pipeline_cases(path)


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":1,"scope":"test","cases":[],"cases":[]}',
        '{"schema_version":1,"scope":"test","cases":[{"handoff_failure":false,"handoff_failure":true}]}',
    ],
)
def test_loader_rejects_duplicate_json_keys(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate pipeline JSON key"):
        load_pipeline_cases(path)
