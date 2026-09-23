"""Answer-quality eval: loading, think-stripping, judge parsing, scoring."""

import json
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.evaluation.answer_eval import (
    ANSWER_GOLDEN_SET_FILE,
    AnswerCase,
    build_judge_messages,
    load_answer_cases,
    parse_judge_json,
    run_answer_eval,
    strip_think,
)


class TestStripThink:
    """strip_think removes provider-inlined reasoning blocks."""

    def test_removes_single_block(self):
        assert strip_think("<think>推理过程</think>最终答案") == "最终答案"

    def test_removes_multiline_block(self):
        text = "<think>\nstep 1\nstep 2\n</think>\n答案文本"
        assert strip_think(text) == "答案文本"

    def test_removes_multiple_blocks(self):
        assert strip_think("<think>a</think>中段<think>b</think>尾") == "中段尾"

    def test_no_block_is_noop(self):
        assert strip_think("  普通回答  ") == "普通回答"

    def test_unclosed_block_left_intact(self):
        # An unterminated <think> is left as-is: stripping it would risk
        # discarding a legitimate answer that merely quotes the tag.
        assert strip_think("<think>只有开头") == "<think>只有开头"


class TestParseJudgeJson:
    """parse_judge_json recovers per-point verdicts, or gives up loudly."""

    def test_plain_json(self):
        assert parse_judge_json('{"0": true, "1": false}', 2) == [True, False]

    def test_json_with_surrounding_prose(self):
        text = '好的，结论如下：\n{"0": true, "1": true}\n以上。'
        assert parse_judge_json(text, 2) == [True, True]

    def test_json_with_think_block(self):
        text = '<think>逐条核对</think>{"0": false, "1": true}'
        assert parse_judge_json(text, 2) == [False, True]

    def test_unquoted_keys_are_invalid_json(self):
        # Strict JSON only: unquoted keys are rejected, not guessed at.
        assert parse_judge_json("{0: true, 1: false}", 2) is None

    def test_missing_verdict_returns_none(self):
        assert parse_judge_json('{"0": true}', 2) is None

    def test_non_bool_verdict_returns_none(self):
        assert parse_judge_json('{"0": "yes", "1": true}', 2) is None

    def test_garbage_returns_none(self):
        assert parse_judge_json("抱歉我不能判断", 2) is None


class TestBuildJudgeMessages:
    def test_prompt_contains_case_and_verdict_contract(self):
        case = AnswerCase(
            id="t1",
            query="怎么退货？",
            category="return",
            required_points=["7天无理由", "运费规则"],
        )
        messages = build_judge_messages(case, "可以退货")
        assert len(messages) == 1
        content = messages[0].content
        assert "怎么退货？" in content
        assert "[0] 7天无理由" in content
        assert "[1] 运费规则" in content
        assert "可以退货" in content
        assert "JSON" in content


def _judge_responding(text: str) -> Mock:
    """LLM mock whose generate returns a response object with .content."""
    llm = Mock()
    llm.generate = AsyncMock(return_value=Mock(content=text))
    return llm


def _case(**overrides: Any) -> AnswerCase:
    base: dict[str, Any] = {
        "id": "c1",
        "query": "q",
        "category": "cat",
        "required_points": ["要点甲", "要点乙"],
    }
    base.update(overrides)
    return AnswerCase(**base)


class TestRunAnswerEval:
    @pytest.mark.asyncio
    async def test_all_points_pass(self):
        judge = _judge_responding('{"0": true, "1": true}')
        answers = iter(["答案A", "答案B"])
        results = []

        async def answer_fn(case):
            results.append(next(answers))
            return results[-1]

        report = await run_answer_eval(answer_fn, judge, cases=[_case(), _case(id="c2")])

        assert report.total == 2
        assert report.passed == 2
        assert report.point_total == 4
        assert report.point_passed == 4
        assert report.failures == []
        assert report.pass_rate == 1.0

    @pytest.mark.asyncio
    async def test_missed_point_fails_case(self):
        judge = _judge_responding('{"0": true, "1": false}')

        async def answer_fn(case):
            return "只覆盖了一个要点"

        report = await run_answer_eval(answer_fn, judge, cases=[_case()])

        assert report.passed == 0
        assert report.point_passed == 1
        assert report.point_coverage == 0.5
        assert report.failures[0].points_missed == ["要点乙"]

    @pytest.mark.asyncio
    async def test_forbidden_hit_fails_even_with_all_points(self):
        judge = _judge_responding('{"0": true, "1": true}')

        async def answer_fn(case):
            return "完美的答案，但提到了根据训练数据"

        report = await run_answer_eval(
            answer_fn, judge, cases=[_case(forbidden_patterns=["根据训练数据"])]
        )

        assert report.passed == 0
        assert report.failures[0].forbidden_hits == ["根据训练数据"]
        # Points still count toward coverage even on a failed case.
        assert report.point_passed == 2

    @pytest.mark.asyncio
    async def test_unparseable_judge_fails_closed(self):
        judge = _judge_responding("我拒绝输出JSON")

        async def answer_fn(case):
            return "答案"

        report = await run_answer_eval(answer_fn, judge, cases=[_case()])

        assert report.judge_errors == 1
        assert report.passed == 0
        assert report.failures[0].judge_error is True
        # A judge error yields no point verdicts, so coverage is 0/0.
        assert report.point_total == 0
        assert report.point_coverage == 0.0

    @pytest.mark.asyncio
    async def test_think_block_in_answer_is_stripped_before_scoring(self):
        judge = _judge_responding('{"0": true, "1": true}')

        async def answer_fn(case):
            return "<think>内心独白</think>正式回答，覆盖要点甲和要点乙"

        report = await run_answer_eval(answer_fn, judge, cases=[_case()])

        assert report.passed == 1
        # The judge saw the stripped answer, not the raw one.
        sent = judge.generate.call_args.kwargs["messages"][0].content
        assert "内心独白" not in sent
        assert "正式回答" in sent


class TestLoadAnswerCases:
    def test_bundled_golden_set_loads(self):
        cases = load_answer_cases()
        assert len(cases) >= 20
        assert all(c.required_points for c in cases)
        assert all(c.category for c in cases)

    def test_roundtrip_from_tmp_file(self, tmp_path):
        payload = [
            {
                "id": "x1",
                "query": "q",
                "category": "c",
                "required_points": ["p1"],
                "forbidden_patterns": ["f1"],
                "faq_id": "returns_policy",
            }
        ]
        path = tmp_path / "cases.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        cases = load_answer_cases(path)

        assert len(cases) == 1
        assert cases[0].faq_ids == ["returns_policy"]
        assert cases[0].forbidden_patterns == ["f1"]

    def test_malformed_entries_skipped(self, tmp_path):
        payload = [
            {"id": "ok", "query": "q", "category": "c", "required_points": ["p"]},
            {"id": "no-points", "query": "q", "category": "c", "required_points": []},
            {"id": "missing-query", "category": "c", "required_points": ["p"]},
        ]
        path = tmp_path / "cases.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        cases = load_answer_cases(path)

        assert [c.id for c in cases] == ["ok"]

    def test_missing_file_degrades_to_empty(self, tmp_path):
        assert load_answer_cases(tmp_path / "nope.json") == []

    def test_bundled_set_uses_known_file(self):
        assert ANSWER_GOLDEN_SET_FILE.exists()
