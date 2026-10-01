#!/usr/bin/env python3
"""Run the evolve four-pipeline search on VerilogEval-Spec-to-RTL-DC-opt.

The ICRTL runner predates the VerilogEval-DC-opt benchmark and assumes a
``TOP`` module, an ``All tests PASS!`` VCS marker, and an ICRTL-style testbench.
This adapter converts each VerilogEval problem into that small interface:

* ``RefModule`` in the reference is renamed to ``TOP``;
* the testbench's ``TopModule`` DUT instance is renamed to ``TOP``;
* the prompt explicitly requires a complete ``TOP`` module; and
* VCS results are parsed from VerilogEval's ``Mismatches: 0`` report.

Each chain starts from the converted reference RTL by default.  Codex still
generates one idea per chain and then refines the reference at subsequent
nodes.  All Codex, VCS, Yosys/OpenROAD, model, and DC/PT durations are stored
in the normal ``timings.csv``/``state_v2.sqlite3`` files under one run
directory per problem.

This file intentionally reuses the resumable state and pipeline machinery in
``four_pipeline.py``.  The adapter only supplies the benchmark-specific task,
functional parser, SDC, and DC/PT backend.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from types import SimpleNamespace
from typing import Any, Sequence


# The public dispatcher supplies a clean loader environment.
from . import settings

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
REVOLUTION_ROOT = settings.ROOT
DEFAULT_BENCHMARK = settings.path('verilogeval_benchmark')

# Import only after the loader-path restart above.
from . import four_pipeline as fp  # noqa: E402
from .evolve import (  # noqa: E402
    CodexRunner,
    ExperimentError,
    atomic_write_json,
    initialize_chain_workspace,
    parse_thread_id,
    sha256_file,
)


@dataclasses.dataclass(frozen=True)
class ConvertedTask:
    name: str
    spec: Path
    baseline: Path
    reference: Path
    testbench: Path
    vcd_strip_path: str = "tb"
    top: str = "TOP"
    io_delay_fraction: float = 0.5
    native: Any | None = None
    problem: str = ""
    benchmark_dir: Path | None = None
    sidecar: Path | None = None
    clock_port: str | None = None
    input_ports: tuple[str, ...] = ()
    output_ports: tuple[str, ...] = ()


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ExperimentError(f"expected a JSON object in {path}")
    return data


def ansi_ports(header: str) -> dict[str, list[str]]:
    """Extract simple ANSI ports while carrying directions across commas."""
    ports: dict[str, list[str]] = {"input": [], "output": [], "inout": []}
    direction: str | None = None
    for raw_part in header.split(","):
        part = re.sub(r"//.*|/\*.*?\*/", "", raw_part, flags=re.S).strip()
        match = re.search(r"\b(input|output|inout)\b", part)
        if match:
            direction = match.group(1)
            part = part[match.end() :]
        if direction is None:
            continue
        part = re.sub(r"\[[^]]+\]", " ", part)
        identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_$]*", part)
        identifiers = [
            item
            for item in identifiers
            if item not in {"wire", "reg", "logic", "signed", "unsigned", "var"}
        ]
        if identifiers:
            ports[direction].append(identifiers[-1])
    return ports


def parse_reference_ports(reference: str, path: Path) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    header = re.search(r"(?ms)^\s*module\s+RefModule\s*\((.*?)\)\s*;", reference)
    if header is None:
        raise ExperimentError(f"cannot parse RefModule port list: {path}")
    ports = ansi_ports(header.group(1))
    inputs = tuple(ports["input"])
    outputs = tuple(ports["output"])
    clock = next(
        (name for name in inputs if name.lower() in {"clk", "clock"}),
        None,
    )
    return clock, inputs, outputs


def convert_problem(benchmark_dir: Path, problem: str, run_dir: Path) -> ConvertedTask:
    """Create an immutable, TOP-based view of one VerilogEval problem."""
    benchmark_dir = benchmark_dir.expanduser().resolve()
    source_ref = benchmark_dir / f"{problem}_ref.sv"
    source_prompt = benchmark_dir / f"{problem}_prompt.txt"
    source_tb = benchmark_dir / f"{problem}_test.sv"
    source_sidecar = benchmark_dir / f"{problem}_dcpt.json"
    missing = [str(path) for path in (source_ref, source_prompt, source_tb, source_sidecar) if not path.is_file()]
    if missing:
        raise ExperimentError(f"{problem}: missing benchmark files:\n" + "\n".join(missing))

    conversion = run_dir / "converted"
    conversion.mkdir(parents=True, exist_ok=True)
    baseline = conversion / "initial.sv"
    reference_copy = conversion / "reference.sv"
    spec = conversion / "SPEC.md"
    testbench = conversion / "00_TB"
    reference = source_ref.read_text(encoding="utf-8", errors="replace")
    prompt = source_prompt.read_text(encoding="utf-8", errors="replace")
    test_source = source_tb.read_text(encoding="utf-8", errors="replace")
    updated_reference, ref_count = re.subn(
        r"(?m)^(\s*module\s+)RefModule\b", r"\1TOP", reference, count=1
    )
    if ref_count != 1:
        raise ExperimentError(f"{problem}: reference does not contain one RefModule declaration")
    # VerilogEval testbenches instantiate the candidate as TopModule.  The
    # golden RefModule instance remains unchanged.
    updated_test, dut_count = re.subn(r"\bTopModule\b", "TOP", test_source)
    if dut_count < 1:
        raise ExperimentError(f"{problem}: testbench has no TopModule DUT instance")
    clock_port, input_ports, output_ports = parse_reference_ports(reference, source_ref)

    adapted_prompt = (
        prompt.rstrip()
        + "\n\nEVOLVE ADAPTER CONTRACT:\n"
        + "The candidate module must be named TOP (not RefModule or TopModule). "
        + "Preserve this exact ANSI interface and all externally observable behavior. "
        + "Return only a complete synthesizable SystemVerilog TOP module when asked for RTL.\n"
    )
    if not baseline.is_file() or baseline.read_text(encoding="utf-8", errors="replace") != updated_reference:
        baseline.write_text(updated_reference.rstrip() + "\n", encoding="utf-8")
    if not reference_copy.is_file() or reference_copy.read_text(encoding="utf-8", errors="replace") != reference:
        # VerilogEval's testbench instantiates RefModule as the golden DUT.
        # Keep the original reference alongside the converted TOP candidate.
        reference_copy.write_text(reference.rstrip() + "\n", encoding="utf-8")
    if not spec.is_file() or spec.read_text(encoding="utf-8", errors="replace") != adapted_prompt:
        spec.write_text(adapted_prompt, encoding="utf-8")
    testbench.mkdir(parents=True, exist_ok=True)
    converted_tb = testbench / "test.sv"
    if not converted_tb.is_file() or converted_tb.read_text(encoding="utf-8", errors="replace") != updated_test:
        converted_tb.write_text(updated_test, encoding="utf-8")

    metadata = {
        "problem": problem,
        "source_reference": str(source_ref),
        "source_prompt": str(source_prompt),
        "source_testbench": str(source_tb),
        "source_dcpt": str(source_sidecar),
        "reference_sha256": sha256_file(source_ref),
        "converted_reference_sha256": sha256_file(baseline),
        "top": "TOP",
        "clock_port": clock_port,
        "input_ports": list(input_ports),
        "output_ports": list(output_ports),
    }
    atomic_write_json(conversion / "conversion.json", metadata)
    return ConvertedTask(
        name=problem,
        spec=spec,
        baseline=baseline,
        reference=reference_copy,
        testbench=testbench,
        problem=problem,
        benchmark_dir=benchmark_dir,
        sidecar=source_sidecar,
        clock_port=clock_port,
        input_ports=input_ports,
        output_ports=output_ports,
    )


def render_verilogeval_sdc(task: ConvertedTask, period: float) -> str:
    """Render the Revolution OpenROAD constraints for the proxy pipelines.

    The DC/PT sidecars contain several Synopsys-only commands (for example
    ``set_dont_touch_network`` and ``set_fix_hold``).  They must not be fed to
    OpenROAD: its SDC reader aborts on the first unknown command, which used to
    make every Yosys/OpenROAD point look like a timing failure.  Revolution's
    OpenROAD flow uses the portable clock and IO-delay constraints below; the
    implementation-specific optimization commands remain in the DC/PT path.
    """
    # This is the portable subset used by Revolution's
    # ``scripts/util/flow_helpers.tcl`` (clock plus input/output delays).
    delay = 0.6 if period == 1.0 else 1.0
    if task.clock_port:
        clock = f"create_clock -name clk -period {period:g} [get_ports {task.clock_port}]"
        clock_name = "clk"
    else:
        clock = f"create_clock -name vclk -period {period:g}\n"
        clock_name = "vclk"
    data_inputs = [name for name in task.input_ports if name != task.clock_port]
    lines = [f"current_design {task.top}", clock.rstrip()]
    if data_inputs:
        lines.append(
            f"set_input_delay {delay:g} -clock {clock_name} [get_ports {{{' '.join(data_inputs)}}}]"
        )
    if task.output_ports:
        lines.append(f"set_output_delay {delay:g} -clock {clock_name} [all_outputs]")
    return "\n".join(lines) + "\n"


def parse_verilogeval_vcs(output: str, period: float) -> tuple[bool, int | None, float | None]:
    mismatch = re.findall(r"Mismatches:\s*(\d+)\s+in\s+(\d+)\s+samples", output, re.I)
    passed = bool(re.search(r"All tests PASS!?", output, re.I))
    samples: int | None = None
    if mismatch:
        errors, samples_text = mismatch[-1]
        samples = int(samples_text)
        passed = passed or int(errors) == 0
    cycle_match = re.findall(r"total time:\s*(\d+)\s*cycles", output, re.I)
    cycles = int(cycle_match[-1]) if cycle_match else samples
    latency = cycles * period if cycles is not None else None
    return passed, cycles, latency


class VerilogEvalVCS:
    """VCS functional evaluator with a candidate-only cache.

    VerilogEval testbenches do not contain the ICRTL ``CYCLE`` macro and use a
    fixed 1-ps simulation clock.  Functional correctness is therefore identical
    at every requested synthesis period; compile/simulation runs once per
    candidate and subsequent periods receive explicit cache-hit timing events.
    """

    def __init__(self, state: fp.State, run_dir: Path, env: dict[str, str], periods: list[float]):
        self.state = state
        self.root = run_dir / "functional"
        self.env = env
        self.periods = periods
        self.task: ConvertedTask = fp.TASKS["Q3_HC"]  # type: ignore[assignment]

    def evaluate(
        self, candidate: Path, pipeline: str, chain: int, node: int
    ) -> dict[float, tuple[bool, int | None, float | None]]:
        digest = sha256_file(candidate)
        work = self.root / digest
        result_path = work / "result.json"
        if result_path.is_file():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            results = {
                period: (bool(data["passed"]), data.get("cycles"), data.get("latency_ns"))
                for period in self.periods
            }
            for period in self.periods:
                with self.state.event(
                    "vcs_functional", pipeline=pipeline, chain=chain, node=node,
                    candidate=digest, period=period, cache_hit=True,
                ):
                    pass
            return results

        rtl_dir = work / "01_RTL"
        tb_dir = work / "00_TB"
        rtl_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, rtl_dir / "initial.sv")
        shutil.copy2(self.task.reference, rtl_dir / "reference.sv")
        shutil.copytree(self.task.testbench, tb_dir, dirs_exist_ok=True)
        # VerilogEval testbenches carry a 1-ps `timescale while the converted
        # reference may not.  Supply a default timescale for the candidate so
        # VCS does not reject the mixed-timescale compilation unit.
        compile_cmd = [
            "vcs", "-timescale=1ns/1ps", "-full64", "-sverilog", "-o", "simv",
            "initial.sv", "reference.sv", "../00_TB/test.sv",
        ]
        rc, output = self.state.run(
            compile_cmd, rtl_dir, rtl_dir / "compile.log", 600, self.env,
            stage="vcs_compile", pipeline=pipeline, chain=chain, node=node,
            candidate=digest, period=self.periods[0],
        )
        passed = False
        cycles: int | None = None
        latency: float | None = None
        if rc == 0:
            rc, output = self.state.run(
                [str(rtl_dir / "simv")], rtl_dir, rtl_dir / "simulation.log", 60,
                self.env, stage="vcs_simulation", pipeline=pipeline, chain=chain,
                node=node, candidate=digest, period=self.periods[0],
            )
            passed, cycles, latency = parse_verilogeval_vcs(output, self.periods[0])
            if rc:
                passed = False
        data = {"passed": passed, "cycles": cycles, "latency_ns": latency}
        atomic_write_json(result_path, data)
        results = {
            period: (passed, cycles, cycles * period if cycles is not None else None)
            for period in self.periods
        }
        # The first period already has real compile/simulation events.  Keep the
        # other requested periods visible as cache hits for accounting.
        for period in self.periods[1:]:
            with self.state.event(
                "vcs_functional", pipeline=pipeline, chain=chain, node=node,
                candidate=digest, period=period, cache_hit=True,
            ):
                pass
        return results


class VerilogEvalPipelineEvaluator(fp.PipelineEvaluator):
    """Existing model/Yosys evaluator with VerilogEval DC/PT for the golden path."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.task: ConvertedTask = fp.TASKS["Q3_HC"]  # type: ignore[assignment]
        self._dcpt_flow: Any | None = None
        self._dcpt_runner: Any | None = None

    def _ensure_dcpt(self) -> tuple[Any, Any]:
        if self._dcpt_flow is not None and self._dcpt_runner is not None:
            return self._dcpt_flow, self._dcpt_runner
        from ..dcpt import DCPTFlow, DCPTJobRunner

        config_path = settings.path('synthesis_config')
        if not config_path.is_file():
            raise ExperimentError("Configure paths.synthesis_config as an external tool-config JSON file.")
        flow = DCPTFlow.load(settings.path('synthesis_flow'))
        runner = DCPTJobRunner(flow, self.root.parent / "dcpt_cache", config_path)
        self._dcpt_flow = flow
        self._dcpt_runner = runner
        return flow, runner

    def _sidecar_point(self, period: float) -> dict[str, Any]:
        if self.task.sidecar is None:
            raise ExperimentError(f"{self.task.name}: missing DC/PT sidecar")
        payload = read_json(self.task.sidecar)
        for point in payload.get("points", []):
            if float(point.get("clock_ns", -1)) == float(period):
                return point
        raise ExperimentError(f"{self.task.name}: no {period:g}ns point in {self.task.sidecar}")

    def _dc_pt(
        self, candidate: Path, digest: str, period: float, work: Path, args: dict[str, Any]
    ) -> fp.Point:
        del work  # The content-addressed syn_tool cache is the source of truth.
        try:
            flow, runner = self._ensure_dcpt()
            sidecar = self._sidecar_point(period)
            rtl = candidate.read_text(encoding="utf-8", errors="replace")
            sdc = str(sidecar["sdc"])
            core = flow.core
            job_id = core.job_identity(rtl, sdc, self.task.top)
            directory = core.job_dir(runner.work_root, job_id)
            dc_outputs = core.required_dc_outputs(directory, self.task.top)
            dc_cached = all(Path(path).is_file() for path in dc_outputs)
            with self.state.event(
                "dc", **args, cache_hit=dc_cached,
                details={"backend": "external_synthesis_flow", "job_id": job_id},
            ):
                if not dc_cached:
                    runner.run(rtl, sdc, self.task.problem, self.task.top, False)
            pt_outputs = core.required_pt_outputs(directory, self.task.top)
            pt_cached = all(Path(path).is_file() for path in pt_outputs)
            with self.state.event(
                "pt_fixed", **args, cache_hit=pt_cached,
                details={"backend": "external_synthesis_flow", "job_id": job_id},
            ):
                if not pt_cached:
                    runner.run(rtl, sdc, self.task.problem, self.task.top, True)
            metrics = runner.metrics(directory, self.task.top)
            area = float(metrics["area"])
            power = float(metrics["power"])
            delay = float(metrics["eff_clk_period"])
            slack = period - delay
            clean = slack >= -fp.TIMING_EPSILON_NS
            return fp.Point(
                self.name,
                period,
                status="success" if clean and area > 0 and power > 0 and delay > 0 else "timing_failed",
                area_um2=area,
                delay_ns=delay,
                slack_ns=slack,
                power_mw=power,
                feedback=(
                    f"DC/PT {self.task.problem}; period={period:g}ns; "
                    f"delay={delay:.6f}ns; slack={slack:.6f}ns; "
                    f"area={area:.6f}; power={power:.6f}mW"
                ),
                extra={"dcpt_job_id": job_id, "dcpt_directory": str(directory), "sdc": sdc},
            )
        except Exception as exc:  # noqa: BLE001 - convert tool errors to a scored point
            return fp.Point(self.name, period, feedback=f"DC/PT evaluation failed: {type(exc).__name__}: {exc}")


def score_with_synthesis_delay(point: fp.Point) -> fp.Point:
    """Use synthesized/model delay for PPA instead of testbench sample count."""
    if point.delay_ns is not None and math.isfinite(float(point.delay_ns)):
        point.latency_ns = max(float(point.delay_ns), 1e-6)
    elif point.latency_ns is None:
        point.latency_ns = point.period_ns
    # The stock score function enforces positive area/delay/power and writes the
    # unscaled negative product used by the four-pipeline runner.
    return fp._verilogeval_original_score(point)  # type: ignore[attr-defined]


def create_reference_roots(
    state: fp.State,
    run_dir: Path,
    task: ConvertedTask,
    codex: CodexRunner,
    ideas_count: int,
) -> None:
    """Generate ideas, but seed every chain's root candidate with the reference."""
    ideas: list[dict[str, str]] = []
    spec = task.spec.read_text(encoding="utf-8", errors="replace")
    baseline = task.baseline.read_text(encoding="utf-8", errors="replace")
    del spec, baseline
    for chain in range(ideas_count):
        row = state.db.execute("SELECT * FROM roots WHERE chain_id=?", (chain,)).fetchone()
        if row and row["status"] == "generated":
            ideas.append(json.loads(row["idea_json"]))
            continue
        idea_path = run_dir / "ideas" / f"idea_{chain:03d}.json"
        if idea_path.is_file():
            idea = read_json(idea_path)
        else:
            print(f"[idea {chain + 1}/{ideas_count}] generating", flush=True)
            with state.event("idea_generation", chain=chain):
                idea = codex.generate_idea(task, chain, ideas, run_dir / "ideas")
            atomic_write_json(idea_path, idea)
        root_dir = run_dir / "shared_roots" / f"chain_{chain:03d}"
        workspace = root_dir / "workspace"
        initialize_chain_workspace(workspace, task)
        candidate = root_dir / "initial.sv"
        if not candidate.is_file():
            shutil.copy2(task.baseline, candidate)
        state.db.execute(
            """INSERT OR REPLACE INTO roots(
                 chain_id,idea_json,candidate_path,candidate_sha256,thread_id,status)
               VALUES(?,?,?,?,?,'generated')""",
            (chain, json.dumps(idea, sort_keys=True), str(candidate), sha256_file(candidate), None),
        )
        state.db.commit()
        ideas.append(idea)


def write_finalist_validation(
    state: fp.State,
    run_dir: Path,
    env: dict[str, str],
    periods: list[float],
    vcs: VerilogEvalVCS,
    finalists: dict[str, list[Any]],
) -> None:
    """Validate proxy finalists with the DC/PT evaluator and export joins."""
    del env
    task: ConvertedTask = fp.TASKS["Q3_HC"]  # type: ignore[assignment]
    config = state.metadata("config") or {}
    device = str(config.get("device", "cuda:0"))
    raw_checkpoint = Path(config.get("raw_checkpoint", str(fp.RAW_CHECKPOINT)))
    yosys_checkpoint = Path(config.get("yosys_checkpoint", str(fp.YOSYS_CHECKPOINT)))
    golden = VerilogEvalPipelineEvaluator(
        "dc_pt", state, run_dir, vcs.env, periods, vcs, device,
        raw_checkpoint, yosys_checkpoint,
    )
    rows: list[dict[str, Any]] = []
    baseline_digest = sha256_file(task.baseline)
    baseline_points = {
        float(row["period_ns"]): row
        for row in state.db.execute(
            "SELECT * FROM point_results WHERE pipeline='dc_pt' AND candidate_sha256=?",
            (baseline_digest,),
        )
    }
    baseline_rows = [
        {
            "period_ns": row["period_ns"],
            "status": row["status"],
            "score": row["score"],
            "area_um2": row["area_um2"],
            "delay_ns": row["delay_ns"],
            "slack_ns": row["slack_ns"],
            "power_mw": row["power_mw"],
            "result_json": row["result_json"],
        }
        for row in baseline_points.values()
    ]
    atomic_write_json(run_dir / "baseline_golden.json", baseline_rows)

    try:
        for source_pipeline, selected in finalists.items():
            for rank, node in enumerate(selected, 1):
                candidate = Path(node["candidate_path"])
                digest = node["candidate_sha256"]
                print(
                    f"[golden] {source_pipeline} finalist {rank}: DC/PT {task.problem}",
                    flush=True,
                )
                golden.evaluate(candidate, -1, rank)
                for period in periods:
                    golden_row = state.db.execute(
                        """SELECT * FROM point_results
                           WHERE pipeline='dc_pt' AND candidate_sha256=? AND period_ns=?""",
                        (digest, period),
                    ).fetchone()
                    proxy_row = state.db.execute(
                        """SELECT * FROM point_results
                           WHERE pipeline=? AND candidate_sha256=? AND period_ns=?""",
                        (source_pipeline, digest, period),
                    ).fetchone()
                    if golden_row is None:
                        continue
                    baseline_row = baseline_points.get(float(period))
                    baseline_score = baseline_row["score"] if baseline_row is not None else None
                    candidate_score = golden_row["score"]
                    ratio = (
                        baseline_score / candidate_score
                        if baseline_score is not None and candidate_score not in (None, 0)
                        else None
                    )
                    golden_payload = json.loads(golden_row["result_json"])
                    proxy_payload = json.loads(proxy_row["result_json"]) if proxy_row else {}
                    rows.append(
                        {
                            "source_pipeline": source_pipeline,
                            "proxy_rank": rank,
                            "candidate_sha256": digest,
                            "candidate_path": str(candidate),
                            "period_ns": period,
                            "golden_status": golden_row["status"],
                            "golden_score": golden_row["score"],
                            "area_um2": golden_row["area_um2"],
                            "delay_ns": golden_row["delay_ns"],
                            "slack_ns": golden_row["slack_ns"],
                            "fixed_power_mw": golden_row["power_mw"],
                            "vcd_power_mw": None,
                            "vcd_status": "not_run",
                            "vcd_annotation_percent": None,
                            "baseline_golden_score": baseline_score,
                            "baseline_over_candidate_ppa": ratio,
                            "proxy_score": proxy_row["score"] if proxy_row else None,
                            "proxy_area_um2": proxy_row["area_um2"] if proxy_row else None,
                            "proxy_delay_ns": proxy_row["delay_ns"] if proxy_row else None,
                            "proxy_power_mw": proxy_row["power_mw"] if proxy_row else None,
                            "golden_result_json": json.dumps(golden_payload, sort_keys=True),
                            "proxy_result_json": json.dumps(proxy_payload, sort_keys=True),
                        }
                    )
    finally:
        golden.close()
    if rows:
        with (run_dir / "finalists.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        (run_dir / "finalists.csv").write_text("\n", encoding="utf-8")
    atomic_write_json(run_dir / "finalists.json", rows)


def load_problems(path: Path, explicit: str | None, limit: int | None) -> list[str]:
    if explicit:
        problems = [item.strip() for item in explicit.replace(",", " ").split() if item.strip()]
    else:
        if not path.is_file():
            raise ExperimentError(f"problem list not found: {path}")
        problems = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if limit is not None:
        problems = problems[:limit]
    if not problems:
        raise ExperimentError("no VerilogEval problems selected")
    return problems


def install_task_hooks(task: ConvertedTask) -> None:
    """Install task-specific globals consumed by the existing runner."""
    fp.TASKS["Q3_HC"] = task  # type: ignore[assignment]
    if not hasattr(fp, "_verilogeval_original_score"):
        fp._verilogeval_original_score = fp.score_point  # type: ignore[attr-defined]
    fp.score_point = score_with_synthesis_delay
    fp.FunctionalVCS = VerilogEvalVCS
    fp.PipelineEvaluator = VerilogEvalPipelineEvaluator
    fp.render_sdc = lambda period: render_verilogeval_sdc(task, period)


def run_one_problem(args: argparse.Namespace, problem: str) -> int:
    run_dir = (args.output_root / problem).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    task = convert_problem(args.benchmark_dir, problem, run_dir)
    if args.convert_only:
        print(f"[converted] {problem}: {run_dir / 'converted'}", flush=True)
        return 0
    summary_path = run_dir / "summary.json"
    if summary_path.is_file() and not args.force:
        try:
            summary = read_json(summary_path)
            sessions = summary.get("sessions", [])
            if sessions and sessions[-1].get("status") == "completed":
                print(f"[skip] {problem}: completed; use --force to re-enter", flush=True)
                return 0
        except ExperimentError:
            pass

    install_task_hooks(task)
    fp.create_ideas_and_roots = create_reference_roots
    fp.validate_golden = write_finalist_validation
    argv = [
        "--task", "Q3_HC",
        "--pipelines", args.pipelines,
        "--periods", args.periods,
        "--ideas", str(args.ideas),
        "--nodes-per-chain", str(args.nodes_per_chain),
        "--finalists", str(args.finalists),
        "--score", "ppa",
        "--golden-power", "both",
        "--shared-roots",
        "--vcs-period-policy", "all",
        "--model", args.model,
        "--reasoning-effort", args.reasoning_effort,
        "--raw-checkpoint", str(args.raw_checkpoint),
        "--yosys-checkpoint", str(args.yosys_checkpoint),
        "--device", args.device,
        "--run-dir", str(run_dir),
    ]
    if args.preflight_only:
        argv.append("--preflight-only")
    print(f"[start] {problem} -> {run_dir}", flush=True)
    result = int(fp.main(argv))
    refresh_summary(run_dir)
    return result


def refresh_summary(run_dir: Path) -> None:
    """Refresh session status after four_pipeline closes its SQLite connection.

    The shared runner exports ``summary.json`` just before its ``finally`` block
    marks the current session completed.  Re-export the session rows here so a
    completed adapter run is not displayed as ``running`` after a restart.
    """
    summary_path = run_dir / "summary.json"
    db_path = run_dir / "state_v2.sqlite3"
    if not summary_path.is_file() or not db_path.is_file():
        return
    try:
        summary = read_json(summary_path)
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row
        sessions = [dict(row) for row in db.execute("SELECT * FROM sessions ORDER BY session_id")]
        db.close()
        report_time = dt.datetime.now(dt.timezone.utc)
        active = 0.0
        for row in sessions:
            started = dt.datetime.fromisoformat(row["started_at"])
            completed = (
                dt.datetime.fromisoformat(row["completed_at"])
                if row["completed_at"]
                else report_time
            )
            active += max(0.0, (completed - started).total_seconds())
        summary["sessions"] = sessions
        summary["active_wall_seconds"] = active
        atomic_write_json(summary_path, summary)
    except (OSError, sqlite3.Error, ValueError, TypeError):
        # The detailed timings.csv remains authoritative if a report refresh
        # is interrupted during a server restart.
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--problems-file", type=Path, help="External problem list; default: benchmark directory / problems.txt")
    parser.add_argument("--problems", help="space- or comma-separated problem IDs")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pipelines", default=",".join(fp.DEFAULT_PIPELINES))
    parser.add_argument("--periods", default=",".join(str(x) for x in fp.PERIODS))
    parser.add_argument("--ideas", type=int, default=12)
    parser.add_argument("--nodes-per-chain", type=int, default=5)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--reasoning-effort", default="max")
    parser.add_argument("--raw-checkpoint", type=Path, default=fp.RAW_CHECKPOINT)
    parser.add_argument("--yosys-checkpoint", type=Path, default=fp.YOSYS_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true", help="re-enter completed problem run directories")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--convert-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.benchmark_dir = args.benchmark_dir.expanduser().resolve()
    args.problems_file = (args.problems_file or args.benchmark_dir / 'problems.txt').expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.raw_checkpoint = args.raw_checkpoint.expanduser().resolve()
    args.yosys_checkpoint = args.yosys_checkpoint.expanduser().resolve()
    if not args.convert_only:
        config_path = settings.path('synthesis_config')
        flow_root = settings.path('synthesis_flow')
        required = [config_path, flow_root / 'resynthesize_dataset.py',
                    flow_root / 'synthesize_verilogeval_spec2rtl.py']
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ExperimentError('Configure external synthesis_flow and synthesis_config; missing files:\n' + '\n'.join(missing))
    problems = load_problems(args.problems_file, args.problems, args.limit)
    args.output_root.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for problem in problems:
        try:
            rc = run_one_problem(args, problem)
            if rc:
                failures.append(problem)
        except (ExperimentError, OSError, ValueError) as exc:
            failures.append(problem)
            print(f"[ERROR] {problem}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if args.force and os.environ.get("EVOLVE_FAIL_FAST") == "1":
                raise
    if failures:
        print(f"[done] failures: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"[done] completed/converted {len(problems)} problem(s) under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
