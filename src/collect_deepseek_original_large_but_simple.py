#!/usr/bin/env python3
"""Collect synthesizable DeepSeek solutions for large-but-simple problems.

The DeepSeek clean result tree contains Gen0 plus evolutionary generations.
This script collects every candidate with a matching ``*_synthesis_report.ppa``
and writes a flat DC/PT input layout like ``src/synthesizable/ppa_feedback/
dcpt_inputs``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_SOURCE_ROOT = REPO_ROOT / "exp" / "deepseek_clean_results" / "VerilogEval-Spec-to-RTL"
DEFAULT_PROBLEM_LIST = (
    REPO_ROOT
    / "data"
    / "bench"
    / "VerilogEval-Spec-to-RTL-DC-opt_resynth"
    / "problem_large_but_simple.txt"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "src"
    / "synthesizable"
    / "deepseek_ref"
    / "dcpt_inputs"
)
DEFAULT_BENCHMARK_ROOT = (
    REPO_ROOT / "data" / "bench" / "VerilogEval-Spec-to-RTL-DC-opt_resynth"
)


@dataclass
class ManifestRow:
    candidate_id: str
    problem: str
    generation: str
    candidate: str
    source_sv: str
    source_ppa: str
    output_sv: str
    output_ppa: str
    output_metadata: str


def read_problem_list(path: Path) -> list[str]:
    problems: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            problem = line.strip()
            if not problem or problem.startswith("#"):
                continue
            problems.append(problem)
    return problems


def parse_sample_number(path: Path) -> int:
    marker = "_sample"
    if marker not in path.stem:
        return -1
    sample_part = path.stem.rsplit(marker, 1)[-1]
    try:
        return int(sample_part)
    except ValueError:
        return -1


def parse_generation_number(gen_dir: Path) -> int:
    if not gen_dir.name.startswith("Gen"):
        return -1
    try:
        return int(gen_dir.name.removeprefix("Gen"))
    except ValueError:
        return -1


def discover_synthesizable_sv_files(problem_dir: Path, problem: str) -> list[Path]:
    return sorted(
        (
            sv_path
            for sv_path in problem_dir.glob("Gen*/*.sv")
            if sv_path.with_name(f"{sv_path.stem}_synthesis_report.ppa").exists()
            and sv_path.name.startswith(f"{problem}_")
        ),
        key=lambda p: (parse_generation_number(p.parent), parse_sample_number(p), p.name),
    )


def copy_file(src: Path, dst: Path, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def load_top_modules(benchmark_root: Path) -> dict[str, str]:
    path = benchmark_root / "synthesis_top_module_names.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def parse_ppa_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    metrics: dict[str, Any] = {}
    for key in ("tns", "wns", "eff_clk_period", "power", "area"):
        value = row.get(key, "")
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            metrics[key] = value
    metrics["report_path"] = str(path.resolve())
    return metrics


def write_ppa_csv(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tns", "wns", "eff_clk_period", "power", "area"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: metrics.get(key, "") for key in fieldnames})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def collect_problem(
    problem: str,
    source_root: Path,
    output_root: Path,
    top_modules: dict[str, str],
    overwrite: bool,
) -> tuple[list[ManifestRow], list[str]]:
    problem_dir = source_root / problem
    warnings: list[str] = []

    if not problem_dir.exists():
        warnings.append(f"{problem}: missing source directory {problem_dir}")
        return [], warnings

    sv_files = discover_synthesizable_sv_files(problem_dir, problem)
    if not sv_files:
        warnings.append(f"{problem}: no .sv candidates with .ppa reports found")
        return [], warnings

    rows: list[ManifestRow] = []
    for sv_path in sv_files:
        ppa_path = sv_path.with_name(f"{sv_path.stem}_synthesis_report.ppa")
        candidate_id = f"{problem}__{sv_path.parent.name}__{sv_path.stem}"
        candidate_dir = output_root / candidate_id
        output_sv = candidate_dir / f"{candidate_id}.v"
        output_ppa = candidate_dir / "ppa.csv"
        output_metadata = candidate_dir / "metadata.json"

        copy_file(sv_path, output_sv, overwrite)
        metrics = parse_ppa_report(ppa_path)
        write_ppa_csv(output_ppa, metrics)

        metadata = {
            "candidate_id": candidate_id,
            "ppa_metrics": metrics,
            "problem": problem,
            "source_sv": str(sv_path.resolve()),
            "source_ppa": str(ppa_path.resolve()),
            "top_module": top_modules.get(problem, "TopModule"),
            "workspace_dir": str(candidate_dir.resolve()),
        }
        write_json(output_metadata, metadata)

        rows.append(
            ManifestRow(
                candidate_id=candidate_id,
                problem=problem,
                generation=sv_path.parent.name,
                candidate=sv_path.stem,
                source_sv=str(sv_path.resolve()),
                source_ppa=str(ppa_path.resolve()),
                output_sv=str(output_sv.resolve()),
                output_ppa=str(output_ppa.resolve()),
                output_metadata=str(output_metadata.resolve()),
            )
        )

    return rows, warnings


def write_manifest(output_root: Path, rows: list[ManifestRow], warnings: list[str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ManifestRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    json_path = output_root / "manifest.json"
    manifest = {
        "num_problems": len({row.problem for row in rows}),
        "num_candidates": len(rows),
        "generations": sorted(
            {row.generation for row in rows},
            key=lambda gen: int(gen.removeprefix("Gen")) if gen.removeprefix("Gen").isdigit() else gen,
        ),
        "warnings": warnings,
        "rows": [asdict(row) for row in rows],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect all synthesizable DeepSeek candidates for the "
            "large-but-simple VerilogEval problems in DC/PT input format."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"DeepSeek clean result root. Default: {DEFAULT_SOURCE_ROOT}",
    )
    parser.add_argument(
        "--problem-list",
        type=Path,
        default=DEFAULT_PROBLEM_LIST,
        help=f"Problem list to collect. Default: {DEFAULT_PROBLEM_LIST}",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help=f"Benchmark metadata root. Default: {DEFAULT_BENCHMARK_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Destination directory. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files already present in the output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    problem_list = args.problem_list.resolve()
    benchmark_root = args.benchmark_root.resolve()
    output_root = args.output_root.resolve()

    problems = read_problem_list(problem_list)
    top_modules = load_top_modules(benchmark_root)
    all_rows: list[ManifestRow] = []
    all_warnings: list[str] = []

    for problem in problems:
        rows, warnings = collect_problem(
            problem=problem,
            source_root=source_root,
            output_root=output_root,
            top_modules=top_modules,
            overwrite=args.overwrite,
        )
        all_rows.extend(rows)
        all_warnings.extend(warnings)

    write_manifest(output_root, all_rows, all_warnings)

    print(f"Collected {len(all_rows)} synthesizable DeepSeek candidates from {len(problems)} problems.")
    print(f"Output root: {output_root}")
    if all_warnings:
        print(f"Warnings: {len(all_warnings)}")
        for warning in all_warnings:
            print(f"  - {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
