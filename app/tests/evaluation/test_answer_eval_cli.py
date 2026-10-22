"""CLI gates use offline model substitutes and reject invalid inputs before calls."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.llm.factory import LLMFactory
from scripts import run_answer_eval as cli


def _cases(path: Path, **overrides: object) -> Path:
    case: dict[str, object] = {
        "id": "fixture",
        "query": "q",
        "category": "test",
        "required_points": ["point"],
    }
    case.update(overrides)
    path.write_text(json.dumps([case]))
    return path


@pytest.fixture(autouse=True)
def no_real_models(monkeypatch):
    factory = Mock(side_effect=AssertionError("Real model access is forbidden"))
    monkeypatch.setattr(LLMFactory, "create", factory)
    return factory


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", ["nan", "inf", "-inf", "-0.1", "1.01"])
async def test_invalid_threshold_fails_before_loading_models(
    monkeypatch, no_real_models, threshold
):
    monkeypatch.setattr("sys.argv", ["run_answer_eval.py", f"--threshold={threshold}"])
    with pytest.raises(SystemExit) as exc:
        await cli.main()
    assert exc.value.code == 2
    no_real_models.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [[], {}, [{"id": "broken"}]])
async def test_invalid_cases_are_configuration_failure(
    tmp_path, monkeypatch, no_real_models, payload
):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr("sys.argv", ["run_answer_eval.py", "--cases", str(path)])
    assert await cli.main() == 2
    no_real_models.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_faq_reference_is_rejected_before_model_creation(
    tmp_path, monkeypatch, no_real_models
):
    path = _cases(tmp_path / "case.json", faq_ids=["nonexistent-fixture-faq"])
    monkeypatch.setattr("sys.argv", ["run_answer_eval.py", "--cases", str(path)])
    assert await cli.main() == 2
    no_real_models.assert_not_called()


@pytest.mark.asyncio
async def test_model_errors_are_reported_and_cannot_green_even_zero_threshold(
    tmp_path, monkeypatch, capsys
):
    path = _cases(tmp_path / "case.json")
    llm = Mock()
    llm.generate = AsyncMock(side_effect=RuntimeError("secret-api-token"))
    monkeypatch.setattr(LLMFactory, "create", Mock(return_value=llm))
    monkeypatch.setattr("sys.argv", ["run_answer_eval.py", "--cases", str(path), "--threshold=0"])
    assert await cli.main() == 1
    output = capsys.readouterr()
    assert "generation_errors=1" in output.out
    assert "secret-api-token" not in output.out + output.err
    assert "\nPASS" not in output.out


@pytest.mark.asyncio
async def test_valid_given_context_run_retains_legacy_faq_id(tmp_path, monkeypatch, capsys):
    path = _cases(tmp_path / "case.json", faq_id="returns_policy")
    llm = Mock()
    llm.generate = AsyncMock(side_effect=[Mock(content="answer"), Mock(content='{"0":true}')])
    monkeypatch.setattr(LLMFactory, "create", Mock(return_value=llm))
    monkeypatch.setattr("sys.argv", ["run_answer_eval.py", "--cases", str(path), "--threshold=1"])
    assert await cli.main() == 0
    output = capsys.readouterr().out
    assert "Given-context" in output and "point_coverage=100.0%" in output
    assert "faithfulness" not in output
    assert llm.generate.await_count == 2


@pytest.mark.asyncio
async def test_provider_initialization_failure_does_not_expose_credentials(
    tmp_path, monkeypatch, caplog
):
    path = _cases(tmp_path / "case.json")
    monkeypatch.setattr(LLMFactory, "create", Mock(side_effect=RuntimeError("secret-provider-key")))
    monkeypatch.setattr("sys.argv", ["run_answer_eval.py", "--cases", str(path)])
    assert await cli.main() == 2
    assert "secret-provider-key" not in caplog.text


@pytest.mark.asyncio
async def test_bad_judge_verdict_is_not_allowed_by_zero_threshold(tmp_path, monkeypatch, capsys):
    path = _cases(tmp_path / "case.json")
    llm = Mock()
    llm.generate = AsyncMock(side_effect=[Mock(content="private-answer"), Mock(content="bad JSON")])
    monkeypatch.setattr(LLMFactory, "create", Mock(return_value=llm))
    monkeypatch.setattr("sys.argv", ["run_answer_eval.py", "--cases", str(path), "--threshold=0"])
    assert await cli.main() == 1
    output = capsys.readouterr()
    assert "judge_errors=1" in output.out
    assert "private-answer" not in output.out + output.err
