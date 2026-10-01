#!/usr/bin/env python3
"""CVDP benchmark adapter for the ICRTL EvolVE search engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from . import evolve
from .evolve import EvalResult, ExperimentError, SCORE_SCALE, Task


from . import settings
REVOLUTION_ROOT = settings.ROOT

from revolution.cvdp import CVDPError, CVDPTask, parse_dotenv, project_digest  # noqa: E402
from revolution.cvdp_dcpt import CVDPProjectDCPTEvaluator  # noqa: E402
from revolution.dcpt import DCPTError


DEFAULT_BENCHMARK = settings.path('cvdp_benchmark')
DEFAULT_HEAVY_ROOT = settings.path('cvdp_heavy')
DEFAULT_TOOLS_ROOT = settings.path('cvdp_tools')
DEFAULT_PERIODS = "1,3,5,8,10,15"


def _option(argv: list[str], name: str, default: str | None = None) -> str | None:
    if name not in argv:
        return default
    index = argv.index(name)
    if index + 1 >= len(argv):
        raise ExperimentError(f"{name} requires a value")
    return argv[index + 1]


def _remove_option(argv: list[str], name: str) -> str | None:
    if name not in argv:
        return None
    index = argv.index(name)
    if index + 1 >= len(argv):
        raise ExperimentError(f"{name} requires a value")
    value = argv[index + 1]
    del argv[index : index + 2]
    return value


def _combined_seed(task: CVDPTask) -> str:
    return "\n".join(
        f"// CVDP target: {relative}\n{task.seed[relative].rstrip()}\n"
        for relative in task.targets
    )


def _materialize_input(task: CVDPTask, run_dir: Path) -> Task:
    inputs = run_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    baseline = inputs / "initial.sv"
    spec = inputs / "SPEC.md"
    testbench = inputs / "native_cocotb"
    testbench.mkdir(exist_ok=True)
    evolve.atomic_write_text(baseline, _combined_seed(task))
    evolve.atomic_write_text(
        spec,
        task.prompt.rstrip()
        + "\n\nOptimize the complete target RTL while preserving every public module "
        "interface and all native CVDP harness behavior.\n",
    )
    top = parse_dotenv(task.harness.get("src/.env", "")).get("TOPLEVEL", "").strip()
    if not top:
        raise ExperimentError(f"CVDP task {task.id} has no TOPLEVEL in src/.env")
    return Task(
        name=task.id,
        spec=spec,
        baseline=baseline,
        testbench=testbench,
        vcd_strip_path="",
        top=top,
        io_delay_fraction=0.0,
        native=task,
    )


class CVDPCommercialEvaluator:
    """Expose native Cocotb plus DC/PT results through EvolVE's EvalResult."""

    heavy_root = DEFAULT_HEAVY_ROOT

    def __init__(self, task: Task, env: dict[str, str], cache_root: Path):
        if not isinstance(task.native, CVDPTask):
            raise ExperimentError("CVDP evaluator received a non-CVDP task")
        self.task = task
        self.native = task.native
        self.env = env
        self.cache_root = cache_root.resolve()
        self.flow = CVDPProjectDCPTEvaluator

    def _work_dir(self, candidate: Path, period: float, mode: str) -> Path:
        return self.cache_root / evolve.sha256_file(candidate) / f"{mode}_{evolve.period_tag(period)}ns"

    def _materialize(self, candidate: Path, workspace: Path) -> None:
        self.native.materialize(workspace, self.heavy_root)
        first = workspace / "code" / self.native.targets[0]
        first.write_text(candidate.read_text(encoding="utf-8"), encoding="utf-8")
        for relative in self.native.targets[1:]:
            (workspace / "code" / relative).write_text("", encoding="utf-8")

    def evaluate(self, candidate: Path, period: float, full: bool) -> EvalResult:
        mode = "full" if full else "proxy"
        candidate_hash = evolve.sha256_file(candidate)
        work = self._work_dir(candidate, period, mode)
        result_path = work / "result.json"
        if result_path.is_file():
            cached = EvalResult.from_dict(json.loads(result_path.read_text()))
            if cached.candidate_sha256 == candidate_hash:
                return cached

        work.mkdir(parents=True, exist_ok=True)
        workspace = work / "workspace"
        result = EvalResult(candidate_hash, period, mode)
        start = time.monotonic()
        try:
            self._materialize(candidate, workspace)
            evaluator = self.flow(
                self.cache_root / ".dcpt_cache",
                flow_root=settings.path('synthesis_flow'),
                config_path=settings.path('synthesis_config'),
                cell_library=settings.path('cell_verilog'),
                clocks=[period],
                harness_timeout_seconds=900,
                sandbox_harness=True,
            )
            native_result = evaluator.evaluate(
                self.native.id,
                workspace,
                project_digest(workspace, self.native.targets),
                run_rtl=True,
            ).to_dict()
        except (CVDPError, DCPTError, OSError, ValueError) as exc:
            result.failure_stage = "evaluation"
            result.feedback = str(exc)
            result.runtimes_s["cvdp_evaluation"] = time.monotonic() - start
            evolve.atomic_write_json(result_path, result.to_dict())
            return result

        result.runtimes_s["cvdp_evaluation"] = time.monotonic() - start
        result.rtl_pass = bool(native_result.get("rtl_passed"))
        point = next(
            (
                item
                for item in native_result.get("clock_results", [])
                if float(item.get("clock_ns", -1)) == float(period)
            ),
            None,
        )
        if not result.rtl_pass:
            result.failure_stage = "rtl_simulation"
            result.feedback = json.dumps(native_result.get("clock_results", []), indent=2)[-8000:]
        elif not point or not point.get("success"):
            result.gate_pass = bool(point and point.get("gate_passed"))
            result.failure_stage = str(point.get("stage", "synthesis")) if point else "synthesis"
            result.feedback = str(point.get("error", "No clock result")) if point else "No clock result"
        else:
            metrics = point["metrics"]
            result.gate_pass = bool(point.get("gate_passed"))
            result.timing_clean = bool(point.get("constraint_met"))
            result.area = float(metrics["area"])
            result.slack_ns = float(metrics["wns"])
            result.power_mw = float(metrics["power"])
            # CVDP does not define a transaction-latency score. Holding the
            # requested clock constant makes EvolVE's proxy rank by area.
            result.latency_ns = float(period)
            result.at = result.area * result.latency_ns
            result.ppa = result.at * result.power_mw
            result.score = -result.at / SCORE_SCALE
            if not result.timing_clean:
                result.failure_stage = "timing"
            elif not result.gate_pass:
                result.failure_stage = "gate_simulation"
            else:
                result.failure_stage = None
            result.feedback = (
                f"RTL PASS; gate={result.gate_pass}; period={period:g}ns; "
                f"area={result.area:.3f}; slack={result.slack_ns:.3f}ns; "
                f"power={result.power_mw:.6f}mW; area*period={result.at:.3f}; "
                f"PPA={result.ppa:.3f}"
            )
        evolve.atomic_write_json(result_path, result.to_dict())
        return result


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(argv if argv is not None else sys.argv[1:])
    if '--help' in raw or '-h' in raw:
        parser = argparse.ArgumentParser(description=__doc__, epilog='Additional IGR options are forwarded to the shared EvolVE engine.')
        parser.add_argument('--cvdp-task', required=True)
        parser.add_argument('--cvdp-benchmark', type=Path)
        parser.add_argument('--cvdp-heavy-root', type=Path)
        parser.add_argument('--cvdp-tools-root', type=Path)
        parser.add_argument('--run-dir', required=True, type=Path)
        parser.print_help()
        return 0
    task_id = _remove_option(raw, "--cvdp-task")
    if not task_id:
        raise ExperimentError("--cvdp-task is required")
    benchmark = Path(
        _remove_option(raw, "--cvdp-benchmark") or DEFAULT_BENCHMARK
    ).expanduser().resolve()
    heavy_root = Path(
        _remove_option(raw, "--cvdp-heavy-root") or DEFAULT_HEAVY_ROOT
    ).expanduser().resolve()
    tools_root = Path(
        _remove_option(raw, "--cvdp-tools-root") or DEFAULT_TOOLS_ROOT
    ).expanduser().resolve()
    if not (tools_root / "bin" / "iverilog").is_file():
        raise ExperimentError(f"CVDP host Icarus is missing under {tools_root / 'bin'}")
    os.environ["PATH"] = str(tools_root / "bin") + os.pathsep + os.environ.get("PATH", "")

    run_dir_value = _option(raw, "--run-dir")
    if run_dir_value is None:
        raise ExperimentError("Provide an explicit --run-dir.")
    run_dir = Path(run_dir_value).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        native = CVDPTask.load(benchmark, task_id)
    except (CVDPError, OSError, ValueError) as exc:
        raise ExperimentError(str(exc)) from exc
    task = _materialize_input(native, run_dir)
    CVDPCommercialEvaluator.heavy_root = heavy_root
    evolve.TASKS = {task_id: task}
    evolve.CommercialEvaluator = CVDPCommercialEvaluator
    raw.extend(["--task", task_id])
    if "--periods" not in raw:
        raw.extend(["--periods", DEFAULT_PERIODS])
    if run_dir_value is None:
        raw.extend(["--run-dir", str(run_dir)])
    return evolve.main(raw)


if __name__ == "__main__":
    raise SystemExit(main())
