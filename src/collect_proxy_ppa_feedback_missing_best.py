#!/usr/bin/env python3
"""Collect proxy-backed PPA feedback candidates without running synthesis tools.

This is a focused recovery script for recorded-best candidates that were skipped
from ``src/synthesizable/ppa_feedback`` because an earlier collection pass could
deduplicate by RTL content. It copies only candidates that already have a
``.proxy.json`` sidecar and writes the same one-design-per-folder handoff shape
used by the local synthesis recollection scripts.
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

DEFAULT_EXPERIMENT_ROOT = (
    SCRIPT_DIR / "exp" / "gpt-5.4-nano" / "VerilogEval-Spec-to-RTL-DC-opt"
)
DEFAULT_SUMMARY_RESULTS = SCRIPT_DIR / "exp" / "gpt-5.4-nano" / "20260421_203825_summary_results.txt"
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "data" / "bench" / "VerilogEval-Spec-to-RTL-DC-opt"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "synthesizable" / "ppa_feedback"
DEFAULT_REAL_FEEDBACK = REPO_ROOT / "data" / "synth" / "test_ppa_feedback.json"
DEFAULT_TARGET_ID = "Prob023_vector100r__Gen11__Prob023_vector100r_M-R_sample9"


@dataclass(frozen=True)
class Candidate:
    problem: str
    generation: str
    stem: str
    source_sv: Path
    proxy_json: Path
    summary_status: str = ""
    summary_score: str = ""
    source: str = ""

    @property
    def candidate_id(self) -> str:
        return f"{self.problem}__{self.generation}__{self.stem}"


@dataclass
class ManifestRow:
    candidate_id: str
    problem: str
    generation: str
    candidate: str
    source_sv: str
    source_proxy_json: str
    output_sv: str
    output_proxy_json: str
    output_ppa: str
    output_metadata: str
    workspace_dir: str
    status: str
    area: Any
    power: Any
    eff_clk_period: Any
    tns: Any
    wns: Any
    summary_status: str
    summary_score: str
    source: str
    note: str


def resolve_existing_path(raw: str | Path, bases: list[Path]) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path

    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate

    return (bases[0] / path).resolve()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def load_top_module_names(benchmark_root: Path) -> dict[str, str]:
    data = read_json(benchmark_root / "synthesis_top_module_names.json")
    if not data:
        return {}
    return {str(k): str(v) for k, v in data.items()}


def proxy_to_ppa(proxy_json: Path) -> tuple[bool, dict[str, Any], str]:
    data = read_json(proxy_json)
    if data is None:
        return False, {}, f"missing or invalid proxy json: {proxy_json}"

    raw = data.get("proxy_metrics", data)
    if not isinstance(raw, dict):
        return False, {}, f"invalid proxy_metrics in {proxy_json}"
    if raw.get("success") is False:
        return False, {}, f"proxy success=false in {proxy_json}"

    power = raw.get("total_power")
    if power is None and raw.get("static_power") is not None and raw.get("switching_power") is not None:
        power = raw["static_power"] + raw["switching_power"]

    area = raw.get("area")
    delay = raw.get("delay")
    if area is None or delay is None or power is None:
        return False, {}, f"proxy metrics missing area/delay/power in {proxy_json}"

    return (
        True,
        {
            "area": area,
            "power": power,
            "eff_clk_period": delay,
            "tns": raw.get("tns", 0.0),
            "wns": raw.get("wns", 0.0),
        },
        "",
    )


def write_ppa_csv(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tns", "wns", "eff_clk_period", "power", "area"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: metrics.get(key, "") for key in fieldnames})


def parse_summary_candidates(summary_results: Path, experiment_root: Path) -> list[Candidate]:
    bases = [SCRIPT_DIR, REPO_ROOT, summary_results.parent, experiment_root.parent]
    candidates: list[Candidate] = []

    with summary_results.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            problem = row[0].strip()
            status = row[1].strip()
            sv_path = resolve_existing_path(row[2].strip(), bases)
            proxy_path = resolve_existing_path(row[3].strip(), bases)
            score = row[4].strip() if len(row) >= 5 else ""
            if not problem or not sv_path.exists():
                continue
            candidates.append(
                Candidate(
                    problem=problem,
                    generation=sv_path.parent.name,
                    stem=sv_path.stem,
                    source_sv=sv_path,
                    proxy_json=proxy_path,
                    summary_status=status,
                    summary_score=score,
                    source="summary",
                )
            )

    return candidates


def discover_proxy_candidates(experiment_root: Path, problems: set[str] | None) -> list[Candidate]:
    candidates: list[Candidate] = []
    for sv_path in sorted(experiment_root.glob("*/Gen*/*.sv")):
        if sv_path.name.endswith(("_test.sv", "_ref.sv", ".syn.v")):
            continue
        problem = sv_path.parent.parent.name
        if problems is not None and problem not in problems:
            continue
        proxy_path = sv_path.with_suffix(".proxy.json")
        if not proxy_path.exists():
            continue
        candidates.append(
            Candidate(
                problem=problem,
                generation=sv_path.parent.name,
                stem=sv_path.stem,
                source_sv=sv_path,
                proxy_json=proxy_path,
                source="proxy-discovery",
            )
        )
    return candidates


def find_candidate_by_id(experiment_root: Path, candidate_id: str) -> Candidate | None:
    parts = candidate_id.split("__", 2)
    if len(parts) != 3:
        return None
    problem, generation, stem = parts
    sv_path = experiment_root / problem / generation / f"{stem}.sv"
    proxy_path = sv_path.with_suffix(".proxy.json")
    if not sv_path.exists() or not proxy_path.exists():
        return None
    return Candidate(
        problem=problem,
        generation=generation,
        stem=stem,
        source_sv=sv_path,
        proxy_json=proxy_path,
        source="target",
    )


def read_real_feedback_ids(real_feedback: Path) -> set[str]:
    try:
        with real_feedback.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(data, list):
        return set()
    ids: set[str] = set()
    for row in data:
        if not isinstance(row, dict):
            continue
        key = row.get("candidate_id") or row.get("Name")
        if key:
            ids.add(str(key))
    return ids


def collect_candidate(
    cand: Candidate,
    output_root: Path,
    top_modules: dict[str, str],
    overwrite: bool,
) -> ManifestRow:
    ok, metrics, error = proxy_to_ppa(cand.proxy_json)
    candidate_dir = output_root / "dcpt_inputs" / cand.candidate_id
    workspace_dir = output_root / "reeval_workspace" / cand.problem / cand.candidate_id
    output_sv = candidate_dir / f"{cand.candidate_id}.v"
    output_proxy_json = candidate_dir / f"{cand.candidate_id}.proxy.json"
    output_ppa = candidate_dir / "ppa.csv"
    output_metadata = candidate_dir / "metadata.json"
    workspace_sv = workspace_dir / f"{cand.candidate_id}.sv"
    workspace_proxy_json = workspace_dir / cand.proxy_json.name
    workspace_ppa = workspace_dir / f"{cand.candidate_id}.proxy.ppa"
    workspace_result = workspace_dir / "reeval_result.json"

    note = ""
    if candidate_dir.exists() and not overwrite:
        note = "already exists; left unchanged"
    elif ok:
        candidate_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cand.source_sv, output_sv)
        shutil.copy2(cand.proxy_json, output_proxy_json)
        shutil.copy2(cand.source_sv, workspace_sv)
        shutil.copy2(cand.proxy_json, workspace_proxy_json)
        write_ppa_csv(output_ppa, metrics)
        write_ppa_csv(workspace_ppa, metrics)

        metadata = {
            "candidate_id": cand.candidate_id,
            "problem": cand.problem,
            "source_sv": str(cand.source_sv.resolve()),
            "source_proxy_json": str(cand.proxy_json.resolve()),
            "top_module": top_modules.get(cand.problem, "TopModule"),
            "workspace_dir": str(workspace_dir.resolve()),
            "ppa_metrics": {
                **metrics,
                "report_path": str(workspace_ppa.resolve()),
            },
        }
        write_json(output_metadata, metadata)

        result = {
            "problem": cand.problem,
            "candidate_id": cand.candidate_id,
            "source_sv": str(cand.source_sv.resolve()),
            "workspace_dir": str(workspace_dir.resolve()),
            "status": "success",
            "synthesis_success": True,
            "synthesis_functionality_success": True,
            "ppa_success": True,
            "area": metrics["area"],
            "power": metrics["power"],
            "eff_clk_period": metrics["eff_clk_period"],
            "tns": metrics["tns"],
            "wns": metrics["wns"],
            "report_path": str(workspace_ppa.resolve()),
            "dcpt_design_dir": str(candidate_dir.resolve()),
            "error": "",
            "evaluation_mode": "proxy_json_only",
        }
        write_json(workspace_result, result)
    else:
        note = error

    return ManifestRow(
        candidate_id=cand.candidate_id,
        problem=cand.problem,
        generation=cand.generation,
        candidate=cand.stem,
        source_sv=str(cand.source_sv.resolve()),
        source_proxy_json=str(cand.proxy_json.resolve()),
        output_sv=str(output_sv.resolve()),
        output_proxy_json=str(output_proxy_json.resolve()),
        output_ppa=str(output_ppa.resolve()),
        output_metadata=str(output_metadata.resolve()),
        workspace_dir=str(workspace_dir.resolve()),
        status="success" if ok else "failed",
        area=metrics.get("area"),
        power=metrics.get("power"),
        eff_clk_period=metrics.get("eff_clk_period"),
        tns=metrics.get("tns"),
        wns=metrics.get("wns"),
        summary_status=cand.summary_status,
        summary_score=cand.summary_score,
        source=cand.source,
        note=note,
    )


def write_manifest(output_root: Path, rows: list[ManifestRow], summary: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "proxy_recollect_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ManifestRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    summary["manifest_csv"] = str(csv_path.resolve())
    write_json(output_root / "proxy_recollect_summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect summary-selected proxy candidates into synthesizable/ppa_feedback."
    )
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--summary-results", type=Path, default=DEFAULT_SUMMARY_RESULTS)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--real-feedback", type=Path, default=DEFAULT_REAL_FEEDBACK)
    parser.add_argument(
        "--target-candidate-id",
        action="append",
        default=[],
        help="Candidate id to require/check. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--no-default-target",
        action="store_true",
        help=f"Do not automatically check {DEFAULT_TARGET_ID}.",
    )
    parser.add_argument(
        "--all-proxy",
        action="store_true",
        help="Collect every candidate under the experiment root that has .proxy.json.",
    )
    parser.add_argument(
        "--problem",
        action="append",
        default=[],
        help="Restrict --all-proxy discovery to one or more problems.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    summary_results = args.summary_results.resolve()
    benchmark_root = args.benchmark_root.resolve()
    output_root = args.output_root.resolve()
    real_feedback = args.real_feedback.resolve()

    targets = list(args.target_candidate_id)
    if not args.no_default_target and DEFAULT_TARGET_ID not in targets:
        targets.append(DEFAULT_TARGET_ID)

    candidates: dict[str, Candidate] = {}
    summary_ids: set[str] = set()
    if summary_results.exists():
        for cand in parse_summary_candidates(summary_results, experiment_root):
            candidates.setdefault(cand.candidate_id, cand)
            summary_ids.add(cand.candidate_id)

    for target in targets:
        cand = find_candidate_by_id(experiment_root, target)
        if cand is not None:
            existing = candidates.get(cand.candidate_id)
            if existing is not None:
                cand = Candidate(
                    problem=existing.problem,
                    generation=existing.generation,
                    stem=existing.stem,
                    source_sv=existing.source_sv,
                    proxy_json=existing.proxy_json,
                    summary_status=existing.summary_status,
                    summary_score=existing.summary_score,
                    source="summary+target",
                )
            candidates[cand.candidate_id] = cand

    if args.all_proxy:
        problems = set(args.problem) if args.problem else None
        for cand in discover_proxy_candidates(experiment_root, problems):
            candidates.setdefault(cand.candidate_id, cand)

    real_feedback_ids = read_real_feedback_ids(real_feedback)
    top_modules = load_top_module_names(benchmark_root)

    rows: list[ManifestRow] = []
    sorted_candidates = sorted(
        candidates.values(),
        key=lambda c: (c.problem, int(c.generation.removeprefix("Gen")), c.stem)
        if c.generation.removeprefix("Gen").isdigit()
        else (c.problem, 10**9, c.stem),
    )

    if not args.dry_run:
        for cand in sorted_candidates:
            if cand.proxy_json.exists():
                rows.append(collect_candidate(cand, output_root, top_modules, args.overwrite))
            else:
                rows.append(
                    ManifestRow(
                        candidate_id=cand.candidate_id,
                        problem=cand.problem,
                        generation=cand.generation,
                        candidate=cand.stem,
                        source_sv=str(cand.source_sv.resolve()),
                        source_proxy_json=str(cand.proxy_json.resolve()),
                        output_sv="",
                        output_proxy_json="",
                        output_ppa="",
                        output_metadata="",
                        workspace_dir="",
                        status="failed",
                        area=None,
                        power=None,
                        eff_clk_period=None,
                        tns=None,
                        wns=None,
                        summary_status=cand.summary_status,
                        summary_score=cand.summary_score,
                        source=cand.source,
                        note=f"missing proxy json: {cand.proxy_json}",
                    )
                )

    collected_ids = {row.candidate_id for row in rows if row.status == "success"}
    existing_dcpt_ids = {
        p.name for p in (output_root / "dcpt_inputs").glob("*") if p.is_dir()
    }
    target_report = {}
    for target in targets:
        target_report[target] = {
            "in_summary_results": target in summary_ids,
            "found_proxy_candidate": target in candidates and candidates[target].proxy_json.exists(),
            "already_in_dcpt_inputs_before_or_after": target in existing_dcpt_ids or target in collected_ids,
            "collected_this_run": target in collected_ids,
            "in_real_feedback_json": target in real_feedback_ids,
        }

    summary = {
        "experiment_root": str(experiment_root),
        "summary_results": str(summary_results),
        "benchmark_root": str(benchmark_root),
        "output_root": str(output_root),
        "real_feedback": str(real_feedback),
        "dry_run": args.dry_run,
        "all_proxy": args.all_proxy,
        "total_candidates_selected": len(sorted_candidates),
        "total_candidates_written": len(rows),
        "total_success": len(collected_ids),
        "targets": target_report,
    }

    if not args.dry_run:
        write_manifest(output_root, rows, summary)

    print("Proxy PPA feedback recollection")
    print(f"Selected candidates: {len(sorted_candidates)}")
    print(f"Successful rows: {len(collected_ids)}")
    for target, info in target_report.items():
        state = "FOUND" if info["found_proxy_candidate"] else "MISSING"
        collected = "collected" if info["collected_this_run"] else "not collected"
        print(f"Target {target}: {state}, {collected}")
        print(f"  in summary results: {info['in_summary_results']}")
        print(f"  in real feedback json: {info['in_real_feedback_json']}")
    if not args.dry_run:
        print(f"Manifest: {output_root / 'proxy_recollect_manifest.csv'}")
        print(f"Summary: {output_root / 'proxy_recollect_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
