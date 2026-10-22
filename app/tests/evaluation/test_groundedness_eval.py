"""Contract and failure tests; these do not establish a model judge's accuracy."""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.services.evaluation.groundedness_eval import (
    GroundednessInput,
    judge_groundedness,
    parse_groundedness_verdict,
)
from app.services.llm.base import LLMResponse, LLMServiceBase


def verdict(**changes: Any) -> str:
    return json.dumps(
        {
            "grounded": True,
            "action_claims_supported": True,
            "abstention_correct": True,
            "promises_supported": True,
            "points": [True, True],
            **changes,
        }
    )


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "null",
        "{}",
        "Here is my verdict: " + verdict(),
        verdict(points=[True]),
        verdict(points=[1, True]),
        verdict(grounded="true"),
        verdict(extra=True),
        verdict()[:-1] + ',"grounded":false}',
    ],
)
def test_rejects_ambiguous_or_incomplete_verdict(raw):
    with pytest.raises(ValueError):
        parse_groundedness_verdict(raw, 2)


@pytest.mark.parametrize(
    "flag", ["grounded", "action_claims_supported", "abstention_correct", "promises_supported"]
)
def test_coverage_alone_cannot_pass(flag):
    result = parse_groundedness_verdict(verdict(**{flag: False}), 2)
    assert result.points == [True, True]
    assert not result.passed


def test_completed_think_is_removed_without_accepting_partial_reasoning():
    assert parse_groundedness_verdict("<think>private</think>" + verdict(), 2).passed
    with pytest.raises(ValueError):
        parse_groundedness_verdict("<think>" + verdict(), 2)


@pytest.fixture
def sample():
    return GroundednessInput(
        "What is the policy?",
        "Seven days; I created a ticket.",
        [{"content": "Returns within seven days.", "version": "v2"}],
        {"handoff_outcome": "failed", "ticket_ids": []},
        ["seven days", "ticket failed"],
        ["compensation"],
    )


@pytest.mark.asyncio
async def test_judge_receives_actual_evidence_and_failed_actions(sample):
    judge = AsyncMock(spec=LLMServiceBase)
    judge.generate.return_value = LLMResponse(verdict(action_claims_supported=False), "judge")
    result = await judge_groundedness(sample, judge)
    messages = judge.generate.call_args.args[0]
    payload = json.loads(messages[1].content)
    assert payload["observed_evidence"] == sample.evidence
    assert payload["observed_actions"] == {"handoff_outcome": "failed", "ticket_ids": []}
    assert "NOT source evidence" in messages[0].content
    assert not result.passed
    assert result.model == "judge"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("secret-url"), TimeoutError("secret")])
async def test_provider_failure_preserves_denominator_without_leaking_error(sample, failure):
    judge = AsyncMock(spec=LLMServiceBase)
    judge.generate.side_effect = failure
    result = await judge_groundedness(sample, judge)
    assert result.points == [False, False]
    assert result.error == type(failure).__name__
    assert not result.passed


@pytest.mark.asyncio
async def test_invalid_verdict_fails_all_points(sample):
    judge = AsyncMock(spec=LLMServiceBase)
    judge.generate.return_value = LLMResponse("{}", "judge")
    result = await judge_groundedness(sample, judge)
    assert result.points == [False, False]
    assert result.error == "invalid_verdict"


@pytest.mark.asyncio
async def test_cancellation_propagates(sample):
    judge = AsyncMock(spec=LLMServiceBase)
    judge.generate.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await judge_groundedness(sample, judge)
