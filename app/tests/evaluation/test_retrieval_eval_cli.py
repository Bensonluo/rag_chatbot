"""Retrieval CLI must never accept nonfinite gates or empty/failed runs."""

from unittest.mock import AsyncMock, Mock

import pytest

from app.services.evaluation.retrieval_eval import run_retrieval_baseline
from scripts import run_retrieval_eval as cli


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", ["nan", "inf", "-inf", "-0.01", "1.01"])
async def test_invalid_threshold_rejected_before_running_baseline(monkeypatch, threshold):
    baseline = AsyncMock()
    monkeypatch.setattr(cli, "run_retrieval_baseline", baseline)
    monkeypatch.setattr("sys.argv", ["run_retrieval_eval.py", f"--threshold={threshold}"])
    with pytest.raises(SystemExit) as exc:
        await cli.main()
    assert exc.value.code == 2
    baseline.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_default_cases_cannot_report_success(monkeypatch):
    baseline = AsyncMock(return_value={"cases": 0, "ks": [], "arms": {}})
    monkeypatch.setattr(cli, "load_retrieval_cases", Mock(return_value=[]))
    monkeypatch.setattr(cli, "run_retrieval_baseline", baseline)
    monkeypatch.setattr("sys.argv", ["run_retrieval_eval.py"])
    assert await cli.main() == 2
    baseline.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_input_has_safe_nonzero_diagnostic(monkeypatch, caplog):
    monkeypatch.setattr(cli, "load_retrieval_cases", Mock(side_effect=ValueError("private-value")))
    monkeypatch.setattr("sys.argv", ["run_retrieval_eval.py", "--cases", "invalid.json"])
    assert await cli.main() == 2
    assert "private-value" not in caplog.text


@pytest.mark.asyncio
async def test_runtime_failure_is_nonzero_without_exception_text(monkeypatch, caplog):
    monkeypatch.setattr(
        cli, "run_retrieval_baseline", AsyncMock(side_effect=RuntimeError("secret"))
    )
    monkeypatch.setattr("sys.argv", ["run_retrieval_eval.py"])
    assert await cli.main() == 1
    assert "secret" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("metric", [float("nan"), float("inf"), -0.1, 1.1])
async def test_invalid_metric_cannot_green(monkeypatch, metric):
    report = await run_retrieval_baseline()
    report["arms"]["hybrid_rrf_lexical_vector"]["overall"]["recall@3"] = metric
    monkeypatch.setattr(cli, "run_retrieval_baseline", AsyncMock(return_value=report))
    monkeypatch.setattr("sys.argv", ["run_retrieval_eval.py", "--threshold=0"])
    assert await cli.main() == 1


@pytest.mark.asyncio
async def test_real_offline_baseline_still_runs(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run_retrieval_eval.py", "--threshold=0"])
    assert await cli.main() == 0
    assert "retrieval baseline" in capsys.readouterr().out
