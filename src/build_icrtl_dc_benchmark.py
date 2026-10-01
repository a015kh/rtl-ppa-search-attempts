#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from revolution.dcpt import DCPTError, DCPTFlow


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "bench" / "ICRTL"
WORK_ROOT = REPO_ROOT / "output" / "dcpt_ground_truth" / "ICRTL"
CLOCKS = (1.0, 3.0, 5.0, 8.0, 10.0, 15.0)
OPTIMIZATION_CLOCK = 1.0
PROBLEMS = ("Q1_LBP", "Q2_GEMM", "Q3_CONV", "Q4_HC", "Q5_JAM", "Q6_DT")
IMPLEMENTATION_SOURCE = {
    "Q1_LBP": "Q1_LBP",
    "Q2_GEMM": "Q2_GEMM",
    "Q3_CONV": "Q3_CONV",
    # The compiled checkout has these two implementation/testbench directories
    # swapped relative to their specifications.
    "Q4_HC": "Q5_JAM",
    "Q5_JAM": "Q4_HC",
    "Q6_DT": "Q6_DT",
}
DATA_INPUTS = {
    "Q1_LBP": ("rst", "gray_ready", "gray_data"),
    "Q2_GEMM": (
        "rst",
        "tile_en",
        "tileA_data_0",
        "tileA_data_1",
        "tileA_data_2",
        "tileA_data_3",
        "tileB_data_0",
        "tileB_data_1",
        "tileB_data_2",
        "tileB_data_3",
    ),
    "Q3_CONV": ("rst", "ready", "idata", "cdata_rd"),
    "Q4_HC": ("rst", "gray_valid", "gray_data"),
    "Q5_JAM": ("rst", "Cost"),
    "Q6_DT": ("rst", "sti_di", "res_di"),
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def render_sdc(problem: str, period: float) -> str:
    delay = 0.6 if period == 1.0 else 1.0
    data_inputs = " ".join(DATA_INPUTS[problem])
    return f"""
# ICRTL designs use clk as their sequential clock.
set cycle_clk {period:.1f}
create_clock -name clk -period $cycle_clk [get_ports clk]
set_ideal_network [get_ports clk]
set_input_delay {delay:.1f} -clock clk [get_ports {{{data_inputs}}}]
set_output_delay {delay:.1f} -clock clk [all_outputs]

set_dont_touch_network [all_clocks]
set_fix_hold [all_clocks]
set_clock_uncertainty 0.1 [all_clocks]
set_clock_latency 0.5 [all_clocks]
set_operating_conditions -max_library slow -max slow
set_wire_load_model -name tsmc13_wl10 -library slow
set_load 1 [all_outputs]
set_drive 0.1 [get_ports {{{data_inputs}}}]
set_max_fanout 20 [get_ports {{{data_inputs}}}]
"""


def source_tasks(
    benchmark: Path, problems: Sequence[str] = PROBLEMS
) -> list[dict[str, Any]]:
    tasks = []
    for problem in problems:
        spec_root = benchmark / problem
        implementation_root = benchmark / IMPLEMENTATION_SOURCE[problem]
        rtl_path = implementation_root / "ref_solution" / "initial.sv"
        test_path = implementation_root / "00_TB" / "test.sv"
        spec_path = spec_root / "referenced_spec" / "human.md"
        for path in (rtl_path, test_path, spec_path):
            if not path.is_file():
                raise DCPTError(f"Missing compiled ICRTL input: {path}")
        rtl = rtl_path.read_text(encoding="utf-8")
        if re.search(r"\bmodule\s+TOP\b", rtl) is None:
            raise DCPTError(f"{rtl_path} does not define module TOP")
        required_ports = DATA_INPUTS[problem]
        missing_ports = [
            port
            for port in required_ports
            if re.search(rf"\b{re.escape(port)}\b", rtl) is None
        ]
        if missing_ports:
            raise DCPTError(
                f"{problem}: implementation {rtl_path} is missing ports "
                f"{missing_ports}"
            )
        tasks.append(
            {
                "problem": problem,
                "rtl": rtl,
                "rtl_sha256": sha256_text(rtl),
                "rtl_path": rtl_path,
                "test_path": test_path,
                "spec_path": spec_path,
                "implementation_source": IMPLEMENTATION_SOURCE[problem],
            }
        )
    return tasks


def prepare(args: argparse.Namespace, flow: DCPTFlow) -> dict[str, Any]:
    benchmark = args.benchmark.resolve()
    work_root = args.work_root.resolve()
    tasks = source_tasks(benchmark, args.problems)
    rows = []
    for task in tasks:
        for clock in CLOCKS:
            rows.append(
                {
                    "SDC": render_sdc(task["problem"], clock),
                    "area": "",
                    "area_value": 0.0,
                    "clock": clock,
                    "delay": "",
                    "delay_value": 0.0,
                    "design_name": task["problem"],
                    "has_clock": True,
                    "rtl": task["rtl"],
                    "static_power": "",
                    "static_power_value": 0.0,
                    "switching_power": "",
                    "switching_power_value": 0.0,
                    "synth": "",
                    "synthesis_time": 0.0,
                    "tns": 0.0,
                    "total_power_value": 0.0,
                    "wns": 0.0,
                }
            )
    input_root = work_root / "input"
    input_root.mkdir(parents=True, exist_ok=True)
    input_path = input_root / "icrtl_ground_truth.jsonl"
    input_path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    config_path = work_root / "server_config.json"
    if not config_path.is_file():
        flow.core.write_json_atomic(config_path, flow.core.default_config().as_dict())
    manifest = flow.core.prepare_manifest(
        input_root, work_root, None, None, args.overwrite
    )
    return {"rows": len(rows), "manifest": manifest, "config": str(config_path)}


def run(args: argparse.Namespace, flow: DCPTFlow) -> dict[str, Any]:
    work_root = args.work_root.resolve()
    config_path = work_root / "server_config.json"
    config = flow.core.ToolConfig.from_json(config_path)
    preflight = flow.preflight(work_root, args.workers, config_path)
    dc = flow.core.run_stage(work_root, config, "dc", args.workers, 0, True)
    pt = flow.core.run_stage(work_root, config, "pt", args.workers, 0, True)
    final_root = work_root / "final"
    export = (
        {"output_root": str(final_root), "reused": True}
        if final_root.exists()
        else flow.core.export_dataset(work_root, final_root)
    )
    return {"preflight": preflight, "dc": dc, "pt": pt, "export": export}


def final_rows(work_root: Path) -> dict[str, list[dict[str, Any]]]:
    path = work_root / "final" / "icrtl_ground_truth.jsonl"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        grouped[str(row["design_name"])].append(row)
    return grouped


def publish(args: argparse.Namespace, flow: DCPTFlow) -> dict[str, Any]:
    del flow
    benchmark = args.benchmark.resolve()
    tasks = source_tasks(benchmark, args.problems)
    grouped = final_rows(args.work_root.resolve())
    expected_clocks = list(CLOCKS)
    for task in tasks:
        problem = task["problem"]
        rows = grouped.get(problem, [])
        if sorted(float(row["clock"]) for row in rows) != expected_clocks:
            raise DCPTError(f"{problem}: expected six completed clock rows")
        if any(
            float(row["area_value"]) <= 0
            or float(row["total_power_value"]) <= 0
            for row in rows
        ):
            raise DCPTError(f"{problem}: invalid area or power in synthesized rows")

        shutil.copy2(task["rtl_path"], benchmark / f"{problem}_ref.sv")
        shutil.copy2(task["spec_path"], benchmark / f"{problem}_prompt.txt")
        test_text = task["test_path"].read_text(encoding="utf-8")
        test_text = test_text.replace(
            "./00_TB/", f"./{task['implementation_source']}/00_TB/"
        )
        (benchmark / f"{problem}_test.sv").write_text(test_text, encoding="utf-8")

        points = [
            {
                "clock_ns": float(row["clock"]),
                "sdc": row["SDC"],
                "tns_ns": float(row["tns"]),
                "wns_ns": float(row["wns"]),
                "eff_clk_period_ns": float(row["delay_value"]),
                "power_mw": float(row["total_power_value"]),
                "area_um2": float(row["area_value"]),
            }
            for row in sorted(rows, key=lambda item: float(item["clock"]))
        ]
        sidecar = {
            "schema_version": 1,
            "problem": problem,
            "reference_top_module": "TOP",
            "candidate_top_module": "TOP",
            "reference_rtl_sha256": task["rtl_sha256"],
            "clock_port": "clk",
            "has_clock": True,
            "points": points,
        }
        (benchmark / f"{problem}_dcpt.json").write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        primary = next(
            point for point in points if point["clock_ns"] == OPTIMIZATION_CLOCK
        )
        (benchmark / f"{problem}_ppa.txt").write_text(
            "tns,wns,eff_clk_period,power,area\n"
            f"{primary['tns_ns']},{primary['wns_ns']},"
            f"{primary['eff_clk_period_ns']},{primary['power_mw']},"
            f"{primary['area_um2']}\n",
            encoding="utf-8",
        )

    (benchmark / "problems.txt").write_text("\n".join(args.problems) + "\n")
    (benchmark / "synthesis_top_module_names.json").write_text(
        json.dumps({problem: "TOP" for problem in args.problems}, indent=2) + "\n"
    )
    manifest = {
        "schema_version": 1,
        "name": "ICRTL",
        "source_benchmark": "weiber2002/ICRTL-Benchmark compiled six-problem subset",
        "problem_count": len(args.problems),
        "ground_truth_clocks_ns": list(CLOCKS),
        "optimization_clock_ns": OPTIMIZATION_CLOCK,
        "metric_units": {"timing": "ns", "power": "mW", "area": "um^2"},
    }
    (benchmark / "benchmark.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return {"benchmark": str(benchmark), "problems": len(args.problems)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile the local six-problem ICRTL benchmark for DC/PT."
    )
    parser.add_argument("command", choices=("prepare", "run", "publish", "all"))
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--flow-root", type=Path, required=True
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--problems",
        nargs="+",
        choices=PROBLEMS,
        default=list(PROBLEMS),
        help="Compiled ICRTL problems to synthesize and publish.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        flow = DCPTFlow.load(args.flow_root)
        result = {}
        if args.command in {"prepare", "all"}:
            result["prepare"] = prepare(args, flow)
        if args.command in {"run", "all"}:
            result["run"] = run(args, flow)
        if args.command in {"publish", "all"}:
            result["publish"] = publish(args, flow)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (DCPTError, OSError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
