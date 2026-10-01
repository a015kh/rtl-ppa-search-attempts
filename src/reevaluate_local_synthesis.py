#!/usr/bin/env python3
"""Re-evaluate generated RTL candidates with local Yosys/OpenROAD flow only.

This script is designed for recovering runs that used proxy synthesis and therefore
need real synthesis artifacts (.rpt/.ppa) without any remote LLM calls.

It:
1. Reads a run-level summary results file (e.g. 20260421_181056_summary_results.txt).
2. Discovers generated candidate RTL files under an experiment directory.
3. Re-runs synthesis via the original REvolution SynthesisEvaluator (Yosys/OpenROAD),
   or bypasses synthesis by validating existing .proxy.json files.
4. Collects successful candidates into a DC/PT-friendly one-design-per-folder layout.

The script never edits original experiment files. All outputs are written to a
separate workspace under --out-root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

EVALUATION_PY = SCRIPT_DIR / "revolution" / "evaluation.py"
_spec = importlib.util.spec_from_file_location("revolution_evaluation_local", EVALUATION_PY)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load evaluation module from: {EVALUATION_PY}")
_evaluation_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluation_mod)

SynthesisEvaluator = _evaluation_mod.SynthesisEvaluator


PASS_PATTERNS = (
    re.compile(r"^Mismatches:\s*0\b", re.MULTILINE),
    re.compile(r"===========Your Design Passed==========="),
    re.compile(r"All tests PASS!"),
)


@dataclass
class Candidate:
    problem: str
    source_sv: Path
    gen_name: str
    stem: str

    @property
    def unique_id(self) -> str:
        return f"{self.problem}__{self.gen_name}__{self.stem}"


def parse_summary_results(summary_results_path: Path, status_filter: str) -> list[str]:
    problems: list[str] = []
    normalized_filter = status_filter.strip().lower()
    with summary_results_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Support both 2-column and extended summary formats:
            #   problem,status
            #   problem,status,sv_path,proxy_path,score
            problem = row[0].strip() if len(row) >= 1 else ""
            status = row[1].strip().lower() if len(row) >= 2 else ""
            if not problem or not status:
                continue
            if normalized_filter == "all" or status == normalized_filter:
                problems.append(problem)
    return problems


def log_indicates_functional_pass(sim_log_path: Path) -> bool:
    if not sim_log_path.exists():
        return False
    try:
        text = sim_log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False

    for pat in PASS_PATTERNS:
        if pat.search(text):
            return True
    return False


def sha256_text(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def discover_candidates(
    problem: str,
    experiment_root: Path,
    only_sim_passed: bool,
    dedup_by_content: bool,
    max_candidates_per_problem: int,
) -> list[Candidate]:
    problem_dir = experiment_root / problem
    if not problem_dir.exists():
        return []

    all_sv = sorted(problem_dir.glob("Gen*/*.sv"))
    picked: list[Candidate] = []
    seen_hashes: set[str] = set()

    for sv in all_sv:
        # Keep only generated candidate RTL files.
        if sv.name.endswith(".syn.v"):
            continue
        if sv.name.endswith("_test.sv") or sv.name.endswith("_ref.sv"):
            continue

        if only_sim_passed:
            sim_log = sv.with_name(f"{sv.stem}_simulation.log")
            if not log_indicates_functional_pass(sim_log):
                continue

        if dedup_by_content:
            digest = sha256_text(sv)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

        picked.append(
            Candidate(
                problem=problem,
                source_sv=sv,
                gen_name=sv.parent.name,
                stem=sv.stem,
            )
        )

        if max_candidates_per_problem > 0 and len(picked) >= max_candidates_per_problem:
            break

    return picked


def load_top_module_names(benchmark_path: Path) -> dict[str, str]:
    top_map_path = benchmark_path / "synthesis_top_module_names.json"
    if not top_map_path.exists():
        return {}
    try:
        with top_map_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        return None
    return None


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def find_default_benchmark_path(experiment_root: Path) -> Path:
    benchmark_name = experiment_root.name
    return (SCRIPT_DIR.parent / "data" / "bench" / benchmark_name).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate generated RTL using Yosys/OpenROAD and collect synthesizable designs."
    )
    parser.add_argument(
        "--summary-results",
        type=Path,
        required=True,
        help="Path to run-level summary results file (problem,status).",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="Experiment root containing per-problem folders (e.g. exp/.../VerilogEval-Spec-to-RTL-DC-opt).",
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=None,
        help="Benchmark data path containing *_test.sv, *_ref.sv, and synthesis_top_module_names.json. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("synthesizable/no_feedback"),
        help="Output root for re-eval artifacts and DC/PT-ready designs.",
    )
    parser.add_argument(
        "--status-filter",
        "--static-filter",
        dest="status_filter",
        type=str,
        default="failed",
        help="Only include problems with this status from summary results (default: failed). Use 'all' to include all listed problems. '--static-filter' is accepted as a compatibility alias.",
    )
    parser.add_argument(
        "--only-sim-passed",
        action="store_true",
        help="Only reevaluate candidates whose existing simulation logs indicate functional pass.",
    )
    parser.add_argument(
        "--dedup-by-content",
        action="store_true",
        help="Deduplicate candidate RTL files by SHA256 content per problem.",
    )
    parser.add_argument(
        "--max-candidates-per-problem",
        type=int,
        default=0,
        help="Limit candidates per problem after filtering (0 means no limit).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip candidates that already have re-eval_result.json in workspace.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover candidates and print counts; do not run synthesis.",
    )
    parser.add_argument(
        "--yosys-bin",
        type=str,
        default="yosys",
        help="Yosys executable path or command name (default: yosys).",
    )
    parser.add_argument(
        "--openroad-bin",
        type=str,
        default="openroad",
        help="OpenROAD executable path or command name (default: openroad).",
    )
    parser.add_argument(
        "--bypass-synthesis-tools",
        action="store_true",
        help="Skip yosys/openroad and validate metrics from existing .proxy.json files.",
    )
    return parser.parse_args()


def resolve_executable(executable: str) -> str | None:
    if "/" in executable:
        p = Path(executable).expanduser().resolve()
        if p.exists() and p.is_file():
            return str(p)
        return None
    found = shutil.which(executable)
    if found:
        return found
    return None


def bind_custom_synthesis_runner(
    synthesis_evaluator: Any, yosys_bin: str, openroad_bin: str
) -> None:
    def _run_synthesis_with_custom_bins(
        self: Any,
        verilog_file: str,
        problem_name: str,
        synth_top_module_name: str,
        output_directory: str,
        report_base_path: str,
        synthesized_netlist_path: str,
    ) -> tuple[bool, str]:
        clk_period = self.clk_period  # ns
        output_file = synthesized_netlist_path

        sdc_file_path = self._create_sdc_file(
            verilog_file, synth_top_module_name, output_directory, clk_period=clk_period
        )
        yosys_script_path = self._create_yosys_script(
            verilog_file,
            synth_top_module_name,
            output_directory,
            clk_period,
            output_file,
        )
        openroad_script_path = self._create_openroad_script(
            sdc_file_path, synth_top_module_name, output_directory, output_file
        )

        report_path = report_base_path + "_synthesis_report.rpt"
        command = (
            f"{shlex.quote(yosys_bin)} {shlex.quote(yosys_script_path)} && "
            f"{shlex.quote(openroad_bin)} {shlex.quote(openroad_script_path)} | "
            f"tee {shlex.quote(report_path)}"
        )

        print(f"INFO: Running synthesis command: {command}")
        process = subprocess.run(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        if process.returncode == 0:
            print(f"Synthesis completed successfully. Report saved to {report_path}")
            return True, report_path

        stderr_text = process.stderr.decode(errors="ignore")
        print(f"Synthesis failed. Error: {stderr_text}")
        with open(report_path, "a", encoding="utf-8") as f:
            f.write("\n\n--- SYNTHESIS FAILED ---\n")
            f.write(stderr_text)
        return False, report_path

    synthesis_evaluator._run_synthesis = types.MethodType(
        _run_synthesis_with_custom_bins, synthesis_evaluator
    )


def load_proxy_ppa_metrics(proxy_json_path: Path) -> tuple[bool, dict[str, Any] | None, str]:
    data = safe_read_json(proxy_json_path)
    if data is None:
        return False, None, f"missing or invalid proxy json: {proxy_json_path}"

    raw_metrics = data.get("proxy_metrics", data)
    if not isinstance(raw_metrics, dict):
        return False, None, f"proxy json has invalid 'proxy_metrics': {proxy_json_path}"

    if raw_metrics.get("success") is False:
        return False, None, f"proxy metric success=false: {proxy_json_path}"

    area = raw_metrics.get("area")
    eff_clk_period = raw_metrics.get("delay")
    tns = raw_metrics.get("tns", 0.0)
    wns = raw_metrics.get("wns", 0.0)

    power = raw_metrics.get("total_power")
    if power is None:
        static_power = raw_metrics.get("static_power")
        switching_power = raw_metrics.get("switching_power")
        if static_power is not None and switching_power is not None:
            power = static_power + switching_power

    if area is None or eff_clk_period is None or power is None:
        return (
            False,
            None,
            f"proxy metrics missing required fields (area/delay/power): {proxy_json_path}",
        )

    return (
        True,
        {
            "area": area,
            "power": power,
            "eff_clk_period": eff_clk_period,
            "tns": tns,
            "wns": wns,
        },
        "",
    )


def main() -> int:
    args = parse_args()

    summary_results = args.summary_results.resolve()
    experiment_root = args.experiment_root.resolve()
    if args.benchmark_path is not None:
        benchmark_path = args.benchmark_path.resolve()
    else:
        benchmark_path = find_default_benchmark_path(experiment_root)

    out_root = (Path.cwd() / args.out_root).resolve()
    workspace_root = out_root / "reeval_workspace"
    dcpt_root = out_root / "dcpt_inputs"
    manifest_path = out_root / "reeval_manifest.csv"
    summary_json_path = out_root / "reeval_summary.json"

    out_root.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    dcpt_root.mkdir(parents=True, exist_ok=True)

    if not summary_results.exists():
        raise FileNotFoundError(f"summary results file not found: {summary_results}")
    if not experiment_root.exists():
        raise FileNotFoundError(f"experiment root not found: {experiment_root}")
    if not benchmark_path.exists():
        raise FileNotFoundError(f"benchmark path not found: {benchmark_path}")

    problems = parse_summary_results(summary_results, args.status_filter)
    if not problems:
        print("No matching problems found from summary results.")
        return 0

    top_modules = load_top_module_names(benchmark_path)

    discovered: list[Candidate] = []
    discovered_count_by_problem: dict[str, int] = {}

    for problem in problems:
        cands = discover_candidates(
            problem=problem,
            experiment_root=experiment_root,
            only_sim_passed=args.only_sim_passed,
            dedup_by_content=args.dedup_by_content,
            max_candidates_per_problem=args.max_candidates_per_problem,
        )
        discovered.extend(cands)
        discovered_count_by_problem[problem] = len(cands)

    print("Discovered candidates per problem:")
    for p in problems:
        print(f"  {p}: {discovered_count_by_problem.get(p, 0)}")
    print(f"Total discovered candidates: {len(discovered)}")

    if args.dry_run:
        print("Dry run complete. No synthesis executed.")
        return 0

    if not discovered:
        print("No candidates to evaluate after filters.")
        return 0

    synthesis_evaluator: Any = None
    if not args.bypass_synthesis_tools:
        yosys_bin = resolve_executable(args.yosys_bin)
        openroad_bin = resolve_executable(args.openroad_bin)
        if yosys_bin is None:
            raise FileNotFoundError(
                f"Yosys executable not found: {args.yosys_bin}. "
                "Use --yosys-bin with a valid path or command."
            )
        if openroad_bin is None:
            raise FileNotFoundError(
                f"OpenROAD executable not found: {args.openroad_bin}. "
                "Use --openroad-bin with a valid path or command."
            )
        print(f"Using yosys: {yosys_bin}")
        print(f"Using openroad: {openroad_bin}")

        synthesis_evaluator = SynthesisEvaluator(yosys_path="yosys", openroad_path="openroad")
        bind_custom_synthesis_runner(synthesis_evaluator, yosys_bin, openroad_bin)
    else:
        print("Bypassing synthesis tools; validating existing .proxy.json files.")

    fieldnames = [
        "problem",
        "candidate_id",
        "source_sv",
        "workspace_dir",
        "status",
        "synthesis_success",
        "synthesis_functionality_success",
        "ppa_success",
        "area",
        "power",
        "eff_clk_period",
        "tns",
        "wns",
        "report_path",
        "dcpt_design_dir",
        "error",
    ]

    rows: list[dict[str, Any]] = []
    total = len(discovered)
    success_count = 0

    for idx, cand in enumerate(discovered, start=1):
        print(f"[{idx}/{total}] Reevaluating {cand.unique_id}")

        run_dir = workspace_root / cand.problem / cand.unique_id
        run_dir.mkdir(parents=True, exist_ok=True)
        result_json_path = run_dir / "reeval_result.json"

        if args.skip_existing:
            cached = safe_read_json(result_json_path)
            if cached is not None:
                row = {
                    "problem": cand.problem,
                    "candidate_id": cand.unique_id,
                    "source_sv": str(cand.source_sv),
                    "workspace_dir": str(run_dir),
                    "status": cached.get("status", "unknown"),
                    "synthesis_success": cached.get("synthesis_success", False),
                    "synthesis_functionality_success": cached.get(
                        "synthesis_functionality_success", False
                    ),
                    "ppa_success": cached.get("ppa_success", False),
                    "area": cached.get("area"),
                    "power": cached.get("power"),
                    "eff_clk_period": cached.get("eff_clk_period"),
                    "tns": cached.get("tns"),
                    "wns": cached.get("wns"),
                    "report_path": cached.get("report_path"),
                    "dcpt_design_dir": cached.get("dcpt_design_dir", ""),
                    "error": cached.get("error", ""),
                }
                rows.append(row)
                if row["status"] == "success":
                    success_count += 1
                continue

        local_sv = run_dir / f"{cand.unique_id}.sv"
        shutil.copy2(cand.source_sv, local_sv)

        top_module = top_modules.get(cand.problem, "TopModule")
        report_base_path = run_dir / cand.unique_id
        synthesized_netlist_path = local_sv.with_suffix(".syn.v")

        row: dict[str, Any] = {
            "problem": cand.problem,
            "candidate_id": cand.unique_id,
            "source_sv": str(cand.source_sv),
            "workspace_dir": str(run_dir),
            "status": "failed",
            "synthesis_success": False,
            "synthesis_functionality_success": False,
            "ppa_success": False,
            "area": None,
            "power": None,
            "eff_clk_period": None,
            "tns": None,
            "wns": None,
            "report_path": "",
            "dcpt_design_dir": "",
            "error": "",
        }

        if args.bypass_synthesis_tools:
            proxy_json_path = cand.source_sv.with_suffix(".proxy.json")
            if proxy_json_path.exists():
                shutil.copy2(proxy_json_path, run_dir / proxy_json_path.name)

            proxy_ok, proxy_ppa, proxy_error = load_proxy_ppa_metrics(proxy_json_path)
            row["synthesis_success"] = bool(proxy_ok)
            row["synthesis_functionality_success"] = bool(proxy_ok)
            row["ppa_success"] = bool(proxy_ok)

            if proxy_ok and proxy_ppa is not None:
                row["area"] = proxy_ppa.get("area")
                row["power"] = proxy_ppa.get("power")
                row["eff_clk_period"] = proxy_ppa.get("eff_clk_period")
                row["tns"] = proxy_ppa.get("tns")
                row["wns"] = proxy_ppa.get("wns")
                proxy_ppa_path = run_dir / f"{cand.unique_id}.proxy.ppa"
                with proxy_ppa_path.open("w", encoding="utf-8") as f:
                    f.write("tns,wns,eff_clk_period,power,area\n")
                    f.write(
                        f"{row['tns']},{row['wns']},{row['eff_clk_period']},{row['power']},{row['area']}"
                    )
                row["report_path"] = str(proxy_ppa_path)
            else:
                row["error"] = proxy_error
        else:
            try:
                synthesis_success, synthesis_report_path = synthesis_evaluator._run_synthesis(
                    verilog_file=str(local_sv),
                    problem_name=cand.problem,
                    synth_top_module_name=top_module,
                    output_directory=str(run_dir),
                    report_base_path=str(report_base_path),
                    synthesized_netlist_path=str(synthesized_netlist_path),
                )
            except Exception as e:  # pragma: no cover - defensive capture around toolchain
                row["error"] = f"exception during synthesis evaluate: {e}"
                write_json(result_json_path, row)
                rows.append(row)
                continue

            row["synthesis_success"] = bool(synthesis_success)
            # Skip iverilog post-synthesis simulation for this recovery flow.
            row["synthesis_functionality_success"] = bool(synthesis_success)

            if synthesis_success:
                ppa = synthesis_evaluator._parse_ppa_log(synthesis_report_path)
                row["ppa_success"] = True
                row["area"] = ppa.get("area")
                row["power"] = ppa.get("power")
                row["eff_clk_period"] = ppa.get("eff_clk_period")
                row["tns"] = ppa.get("tns")
                row["wns"] = ppa.get("wns")
                row["report_path"] = ppa.get("report_path") or ""
            else:
                row["ppa_success"] = False
                row["error"] = f"synthesis failed: {synthesis_report_path}"

        if (
            row["synthesis_success"]
            and row["synthesis_functionality_success"]
            and row["ppa_success"]
        ):
            row["status"] = "success"
            success_count += 1

            # DC/PT handoff format: one design per folder with one .v file.
            dcpt_design_dir = dcpt_root / cand.unique_id
            dcpt_design_dir.mkdir(parents=True, exist_ok=True)
            dcpt_rtl_path = dcpt_design_dir / f"{cand.unique_id}.v"
            shutil.copy2(local_sv, dcpt_rtl_path)

            if row["report_path"]:
                ppa_src = Path(str(row["report_path"]))
                if ppa_src.exists():
                    shutil.copy2(ppa_src, dcpt_design_dir / "ppa.csv")

            metadata = {
                "problem": cand.problem,
                "candidate_id": cand.unique_id,
                "source_sv": str(cand.source_sv),
                "workspace_dir": str(run_dir),
                "top_module": top_module,
                "ppa_metrics": {
                    "area": row["area"],
                    "power": row["power"],
                    "eff_clk_period": row["eff_clk_period"],
                    "tns": row["tns"],
                    "wns": row["wns"],
                    "report_path": row["report_path"],
                },
            }
            write_json(dcpt_design_dir / "metadata.json", metadata)
            row["dcpt_design_dir"] = str(dcpt_design_dir)

        write_json(result_json_path, row)
        rows.append(row)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary: dict[str, Any] = {
        "summary_results": str(summary_results),
        "experiment_root": str(experiment_root),
        "benchmark_path": str(benchmark_path),
        "out_root": str(out_root),
        "evaluation_mode": (
            "proxy_json_only" if args.bypass_synthesis_tools else "yosys_openroad"
        ),
        "total_problems": len(problems),
        "total_candidates_discovered": len(discovered),
        "total_candidates_evaluated": len(rows),
        "total_success": success_count,
        "dcpt_inputs_dir": str(dcpt_root),
        "manifest_csv": str(manifest_path),
        "problem_candidate_counts": discovered_count_by_problem,
        "problem_success_counts": {},
    }

    per_problem_success: dict[str, int] = {}
    for row in rows:
        if row.get("status") == "success":
            p = str(row["problem"])
            per_problem_success[p] = per_problem_success.get(p, 0) + 1
    summary["problem_success_counts"] = per_problem_success

    write_json(summary_json_path, summary)

    print("\nRe-evaluation completed.")
    print(f"Success candidates: {success_count}/{len(rows)}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_json_path}")
    print(f"DC/PT-ready designs: {dcpt_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
