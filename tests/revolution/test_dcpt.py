import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from revolution.dcpt import (
    DCPTError,
    DCPTFlow,
    load_benchmark_manifest,
    load_problem_point,
)
from revolution.dcpt_evaluation import DCPTEvaluator


def _write_benchmark(tmp_path: Path) -> Path:
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "benchmark.json").write_text(
        json.dumps({"schema_version": 1, "optimization_clock_ns": 1.0})
    )
    (bench / "prob_dcpt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "points": [
                    {"clock_ns": 1.0, "sdc": "create_clock -period 1 vclk"},
                    {"clock_ns": 3.0, "sdc": "create_clock -period 3 vclk"},
                ],
            }
        )
    )
    return bench


def test_load_benchmark_clock_point(tmp_path):
    bench = _write_benchmark(tmp_path)
    assert load_benchmark_manifest(bench)["optimization_clock_ns"] == 1.0
    assert "period 1" in load_problem_point(bench, "prob", 1.0)["sdc"]
    with pytest.raises(DCPTError, match="No 5 ns point"):
        load_problem_point(bench, "prob", 5.0)


def test_preflight_rejects_excess_workers(tmp_path):
    config = SimpleNamespace(max_workers=4)
    core = SimpleNamespace(default_config=lambda: config)
    flow = DCPTFlow(tmp_path, core, SimpleNamespace())
    with pytest.raises(DCPTError, match="between 1 and 4"):
        flow.preflight(tmp_path, 5)


def test_dcpt_failure_contract():
    result = DCPTEvaluator._failure(True, False, "gate simulation failed")
    assert result == {
        "synthesis_success": True,
        "synthesis_functionality_success": False,
        "ppa_success": False,
        "synthesis_log": "gate simulation failed",
        "ppa_metrics": None,
    }


def test_dcpt_timing_annotation_uses_conventional_signed_slack():
    evaluator = object.__new__(DCPTEvaluator)
    evaluator.constraint_tolerance_ns = 1e-6

    met = evaluator._annotate_timing({"eff_clk_period": 4.9}, 5.0)
    violated = evaluator._annotate_timing({"eff_clk_period": 5.2}, 5.0)

    assert met["timing_slack_ns"] == pytest.approx(0.1)
    assert met["constraint_met"] is True
    assert violated["timing_slack_ns"] == pytest.approx(-0.2)
    assert violated["constraint_met"] is False


@pytest.mark.skipif(
    not (os.environ.get("RTL_PPA_EDA_FLOW_ROOT") and os.environ.get("RTL_PPA_EDA_BENCHMARK")),
    reason="Requires an external synthesis flow and benchmark supplied by the caller.",
)
def test_current_benchmark_stages_18_by_6():
    from build_dc_opt_benchmark import SOURCE_BENCH, make_rows, source_tasks

    flow = DCPTFlow.load(os.environ["RTL_PPA_EDA_FLOW_ROOT"])
    tasks = source_tasks(Path(os.environ["RTL_PPA_EDA_BENCHMARK"]), flow)
    rows = make_rows(tasks, flow)
    assert len(tasks) == 18
    assert len(rows) == 108
    assert {row["clock"] for row in rows} == {1.0, 3.0, 5.0, 8.0, 10.0, 15.0}
