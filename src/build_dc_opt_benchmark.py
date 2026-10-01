#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from fix_verilogeval_testbenches import fix_text
from revolution.dcpt import DCPTError, DCPTFlow


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_BENCH = REPO_ROOT / "bench" / "VerilogEval-Spec-to-RTL"
DEST_BENCH = REPO_ROOT / "bench" / "VerilogEval-Spec-to-RTL-DC-opt"
DEFAULT_WORK_ROOT = REPO_ROOT / "output" / "dcpt_ground_truth" / DEST_BENCH.name
CLOCKS = (1.0, 3.0, 5.0, 8.0, 10.0, 15.0)
OPTIMIZATION_CLOCK = 1.0
DC_COMPATIBILITY_REPLACEMENTS = {
    "Prob118_history_shift": {"predict_history = 0;": "predict_history <= 0;"},
    "Prob153_gshare": {
        "pht[i] = LNT;": "pht[i] <= LNT;",
        "predict_history_r = 0;": "predict_history_r <= 0;",
    },
}
ROW_KEYS = {
    "SDC": "",
    "area": "",
    "area_value": 0.0,
    "clock": 0.0,
    "delay": "",
    "delay_value": 0.0,
    "design_name": "",
    "has_clock": False,
    "rtl": "",
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def dc_compatible_rtl(problem: str, rtl: str) -> tuple[str, list[str]]:
    applied: list[str] = []
    for old, new in DC_COMPATIBILITY_REPLACEMENTS.get(problem, {}).items():
        count = rtl.count(old)
        if count != 1:
            raise DCPTError(
                f"{problem}: expected one occurrence of {old!r}; found {count}"
            )
        rtl = rtl.replace(old, new)
        applied.append(f"{old} -> {new}")
    return rtl, applied


def problem_names(source_bench: Path) -> list[str]:
    path = source_bench / "problem_large_but_solvable.txt"
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(names) != 18 or len(set(names)) != 18:
        raise DCPTError(f"Expected 18 unique problems in {path}; found {len(names)}")
    return names


def source_tasks(source_bench: Path, flow: DCPTFlow) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    helper = flow.spec2rtl
    for problem in problem_names(source_bench):
        path = source_bench / f"{problem}_ref.sv"
        source_rtl = path.read_text(encoding="utf-8")
        rtl, compatibility_changes = dc_compatible_rtl(problem, source_rtl)
        modules = flow.core.module_blocks(rtl)
        if list(modules) != ["RefModule"]:
            raise DCPTError(f"{path} must contain exactly one RefModule")
        header = helper.MODULE_HEADER_RE.search(rtl)
        if header is None:
            raise DCPTError(f"Cannot parse RefModule ports: {path}")
        ports = helper.ansi_ports(header.group(1))
        clock_port = next(
            (name for name in ports["input"] if name in {"clk", "clock"}), None
        )
        tasks.append(
            {
                "problem": problem,
                "path": path,
                "rtl": rtl,
                "rtl_sha256": sha256_text(rtl),
                "source_rtl_sha256": sha256_text(source_rtl),
                "dc_compatibility_changes": compatibility_changes,
                "clock_port": clock_port,
                "input_ports": ports["input"],
                "output_ports": ports["output"],
            }
        )
    return tasks


def make_rows(tasks: list[dict[str, Any]], flow: DCPTFlow) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for clock in CLOCKS:
            row = dict(ROW_KEYS)
            row.update(
                {
                    "SDC": flow.spec2rtl.render_sdc(
                        clock,
                        task["clock_port"],
                        task["input_ports"],
                        task["output_ports"],
                    ),
                    "clock": clock,
                    "design_name": task["problem"],
                    "has_clock": task["clock_port"] is not None,
                    "rtl": task["rtl"],
                }
            )
            rows.append(row)
    return rows


def prepare(args: argparse.Namespace, flow: DCPTFlow) -> dict[str, Any]:
    work_root = args.work_root.resolve()
    tasks = source_tasks(args.source_bench.resolve(), flow)
    rows = make_rows(tasks, flow)
    input_root = work_root / "input"
    input_root.mkdir(parents=True, exist_ok=True)
    input_path = input_root / "dc_opt_ground_truth.jsonl"
    payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    input_path.write_text(payload, encoding="utf-8")
    inventory = {
        "schema_version": 1,
        "source_benchmark": str(args.source_bench.resolve()),
        "problems": len(tasks),
        "rows": len(rows),
        "clocks_ns": list(CLOCKS),
        "sources": [
            {
                "problem": task["problem"],
                "rtl_sha256": task["rtl_sha256"],
                "source_rtl_sha256": task["source_rtl_sha256"],
                "dc_compatibility_changes": task["dc_compatibility_changes"],
                "clock_port": task["clock_port"],
            }
            for task in tasks
        ],
    }
    flow.core.write_json_atomic(
        work_root / "benchmark_source_inventory.json", inventory
    )
    config_path = work_root / "server_config.json"
    if not config_path.is_file():
        flow.core.write_json_atomic(config_path, flow.core.default_config().as_dict())
    manifest = flow.core.prepare_manifest(
        input_root, work_root, None, None, args.overwrite
    )
    return {"inventory": inventory, "manifest": manifest, "config": str(config_path)}


def run(args: argparse.Namespace, flow: DCPTFlow) -> dict[str, Any]:
    work_root = args.work_root.resolve()
    config_path = work_root / "server_config.json"
    config = flow.core.ToolConfig.from_json(config_path)
    preflight = flow.preflight(work_root, args.workers, config_path)
    dc = flow.core.run_stage(work_root, config, "dc", args.workers, 0, True)
    pt = flow.core.run_stage(work_root, config, "pt", args.workers, 0, True)
    final_root = work_root / "final"
    if not final_root.exists():
        export = flow.core.export_dataset(work_root, final_root)
    else:
        export = {"output_root": str(final_root), "reused": True}
    verification = verify_export(work_root, args.source_bench.resolve())
    return {
        "preflight": preflight,
        "dc": dc,
        "pt": pt,
        "export": export,
        "verification": verification,
    }


def verify_export(work_root: Path, source_bench: Path) -> dict[str, Any]:
    grouped = read_final_rows(work_root)
    counts = {problem: len(rows) for problem, rows in grouped.items()}
    expected_problems = set(problem_names(source_bench))
    if set(grouped) != expected_problems or any(
        count != len(CLOCKS) for count in counts.values()
    ):
        raise DCPTError(f"Exported shard has invalid problem counts: {counts}")
    return {
        "ok": True,
        "rows": sum(counts.values()),
        "problems": len(counts),
        "counts": counts,
    }


def read_final_rows(work_root: Path) -> dict[str, list[dict[str, Any]]]:
    path = work_root / "final" / "dc_opt_ground_truth.jsonl"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        grouped[str(row["design_name"])].append(row)
    return grouped


def publish(args: argparse.Namespace, flow: DCPTFlow) -> dict[str, Any]:
    source = args.source_bench.resolve()
    destination = args.destination.resolve()
    tasks = source_tasks(source, flow)
    grouped = read_final_rows(args.work_root.resolve())
    errors: list[str] = []
    for task in tasks:
        rows = grouped.get(task["problem"], [])
        if sorted(float(row["clock"]) for row in rows) != list(CLOCKS):
            errors.append(f"{task['problem']}: expected six clock rows")
        for row in rows:
            if sha256_text(str(row["rtl"])) != task["rtl_sha256"]:
                errors.append(f"{task['problem']}: synthesized RTL hash mismatch")
            for key in ("area_value", "delay_value", "total_power_value", "tns", "wns"):
                value = float(row[key])
                if value < 0 or value != value:
                    errors.append(
                        f"{task['problem']} clock={row['clock']}: invalid {key}"
                    )
    if errors:
        raise DCPTError("Cannot publish benchmark: " + "; ".join(errors[:20]))

    destination.mkdir(parents=True, exist_ok=True)
    problems = [task["problem"] for task in tasks]
    for task in tasks:
        problem = task["problem"]
        shutil.copy2(
            source / f"{problem}_prompt.txt", destination / f"{problem}_prompt.txt"
        )
        test_path = destination / f"{problem}_test.sv"
        test_text = (source / f"{problem}_test.sv").read_text(encoding="utf-8")
        fixed_test_text, _, _ = fix_text(test_text)
        test_path.write_text(fixed_test_text, encoding="utf-8")
        (destination / f"{problem}_ref.sv").write_text(task["rtl"], encoding="utf-8")
        points = []
        for row in sorted(grouped[problem], key=lambda item: float(item["clock"])):
            points.append(
                {
                    "clock_ns": float(row["clock"]),
                    "sdc": row["SDC"],
                    "tns_ns": float(row["tns"]),
                    "wns_ns": float(row["wns"]),
                    "eff_clk_period_ns": float(row["delay_value"]),
                    "power_mw": float(row["total_power_value"]),
                    "area_um2": float(row["area_value"]),
                }
            )
        sidecar = {
            "schema_version": 1,
            "problem": problem,
            "reference_top_module": "RefModule",
            "candidate_top_module": "TopModule",
            "reference_rtl_sha256": task["rtl_sha256"],
            "source_reference_rtl_sha256": task["source_rtl_sha256"],
            "dc_compatibility_changes": task["dc_compatibility_changes"],
            "clock_port": task["clock_port"],
            "has_clock": task["clock_port"] is not None,
            "points": points,
        }
        (destination / f"{problem}_dcpt.json").write_text(
            json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        one_ns = next(
            point for point in points if point["clock_ns"] == OPTIMIZATION_CLOCK
        )
        (destination / f"{problem}_ppa.txt").write_text(
            "tns,wns,eff_clk_period,power,area\n"
            f"{one_ns['tns_ns']},{one_ns['wns_ns']},{one_ns['eff_clk_period_ns']},"
            f"{one_ns['power_mw']},{one_ns['area_um2']}\n",
            encoding="utf-8",
        )
    (destination / "problems.txt").write_text(
        "\n".join(problems) + "\n", encoding="utf-8"
    )
    shutil.copy2(
        source / "problem_large_but_solvable.txt",
        destination / "problem_large_but_solvable.txt",
    )
    top_modules = {problem: "TopModule" for problem in problems}
    (destination / "synthesis_top_module_names.json").write_text(
        json.dumps(top_modules, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    preflight = json.loads(
        (args.work_root.resolve() / "preflight.json").read_text(encoding="utf-8")
    )
    manifest = {
        "schema_version": 1,
        "name": destination.name,
        "source_benchmark": source.name,
        "problem_count": len(problems),
        "ground_truth_clocks_ns": list(CLOCKS),
        "optimization_clock_ns": OPTIMIZATION_CLOCK,
        "metric_units": {"timing": "ns", "power": "mW", "area": "um^2"},
        "tools": preflight.get("versions", {}),
    }
    (destination / "benchmark.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "destination": str(destination),
        "problems": len(problems),
        "clock_points": len(problems) * len(CLOCKS),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the VerilogEval DC/PT optimization benchmark."
    )
    parser.add_argument("command", choices=("prepare", "run", "publish", "all"))
    parser.add_argument("--source-bench", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--flow-root", type=Path, required=True
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        flow = DCPTFlow.load(args.flow_root)
        result: dict[str, Any] = {}
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
