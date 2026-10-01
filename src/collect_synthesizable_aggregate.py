#!/usr/bin/env python3
"""Aggregate synthesizable RTL candidates into a DC/PT handoff folder.

This collector combines the discovery styles used by the older collection
scripts:

* original RTL with a sibling ``*.syn.v`` or ``*.syn.sv`` synthesis artifact
* original RTL with a sibling ``*_synthesis_report.ppa``
* original RTL with a sibling ``*.proxy.json`` containing proxy PPA metrics

Outputs are written as one design per folder under ``src/dcpt`` by default:

    src/dcpt/<candidate_id>/<candidate_id>.v
    src/dcpt/<candidate_id>/ppa.csv          # when PPA/proxy metrics exist
    src/dcpt/<candidate_id>/metadata.json

The script never edits the source experiment trees.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_SOURCE_ROOTS = [REPO_ROOT / "exp", SCRIPT_DIR / "exp"]
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "dcpt"
DEFAULT_BENCHMARK_ROOTS = [REPO_ROOT / "data" / "bench", REPO_ROOT / "bench"]

SKIP_SUFFIXES = ("_test.sv", "_ref.sv", ".syn.v", ".syn.sv")
PPA_FIELDS = ["tns", "wns", "eff_clk_period", "power", "area"]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    problem: str
    generation: str
    source_sv: Path
    evidence: str
    ppa_path: Path | None = None
    proxy_json: Path | None = None


@dataclass
class ManifestRow:
    candidate_id: str
    problem: str
    generation: str
    source_sv: str
    evidence: str
    output_dir: str
    output_sv: str
    output_ppa: str
    output_metadata: str
    source_ppa: str
    source_proxy_json: str
    top_module: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect synthesizable RTL candidates into src/dcpt."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Experiment/source root to scan. Can be supplied multiple times. "
            f"Default: {', '.join(str(p) for p in DEFAULT_SOURCE_ROOTS)}"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"DC/PT output folder. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Benchmark metadata root containing synthesis_top_module_names.json. "
            "Can be supplied multiple times."
        ),
    )
    parser.set_defaults(include_proxy_only=True)
    parser.add_argument(
        "--no-include-proxy-only",
        dest="include_proxy_only",
        action="store_false",
        help="Do not collect candidates whose only evidence is .proxy.json.",
    )
    parser.add_argument(
        "--dedup-by-content",
        action="store_true",
        help="Keep only the first RTL file for each SHA256 content hash.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing design folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover candidates and print counts.",
    )
    return parser.parse_args()


def natural_int(text: str) -> int:
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else 10**9


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "candidate"


def infer_problem(sv_path: Path) -> str:
    for parent in sv_path.parents:
        if parent.name.startswith("Prob") or parent.name.startswith("Q"):
            return parent.name

    stem = sv_path.stem
    if "_sample" in stem:
        return stem.split("_sample", 1)[0]
    return stem.split("_", 1)[0]


def infer_generation(sv_path: Path) -> str:
    if sv_path.parent.name.startswith("Gen"):
        return sv_path.parent.name
    return "Ungrouped"


def make_candidate_id(root: Path, sv_path: Path) -> str:
    problem = infer_problem(sv_path)
    generation = infer_generation(sv_path)
    stem = sv_path.stem

    if generation != "Ungrouped":
        return safe_name(f"{problem}__{generation}__{stem}")

    try:
        rel_parent = sv_path.parent.relative_to(root)
        prefix = "__".join(safe_name(part) for part in rel_parent.parts)
        return safe_name(f"{prefix}__{stem}" if prefix else stem)
    except ValueError:
        return safe_name(stem)


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


def load_top_modules(benchmark_roots: list[Path]) -> dict[str, str]:
    top_modules: dict[str, str] = {}
    for root in benchmark_roots:
        if not root.exists():
            continue
        for path in root.rglob("synthesis_top_module_names.json"):
            data = read_json(path)
            if isinstance(data, dict):
                top_modules.update({str(k): str(v) for k, v in data.items()})
    return top_modules


def parse_ppa_report(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            row = next(csv.DictReader(f))
    except (OSError, StopIteration, csv.Error):
        return None

    metrics: dict[str, Any] = {}
    for key in PPA_FIELDS:
        value = row.get(key, "")
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            metrics[key] = value
    metrics["report_path"] = str(path.resolve())
    return metrics


def proxy_to_ppa(path: Path) -> dict[str, Any] | None:
    data = read_json(path)
    if data is None:
        return None
    raw = data.get("proxy_metrics", data)
    if not isinstance(raw, dict) or raw.get("success") is False:
        return None

    power = raw.get("total_power")
    if power is None and raw.get("static_power") is not None and raw.get("switching_power") is not None:
        power = raw["static_power"] + raw["switching_power"]

    area = raw.get("area")
    delay = raw.get("delay")
    if area is None or delay is None or power is None:
        return None

    return {
        "tns": raw.get("tns", 0.0),
        "wns": raw.get("wns", 0.0),
        "eff_clk_period": delay,
        "power": power,
        "area": area,
        "report_path": str(path.resolve()),
    }


def write_ppa_csv(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PPA_FIELDS)
        writer.writeheader()
        writer.writerow({key: metrics.get(key, "") for key in PPA_FIELDS})


def synthesis_artifact_for(sv_path: Path) -> Path | None:
    for artifact in (
        sv_path.with_suffix(".syn.v"),
        sv_path.with_suffix(".syn.sv"),
        sv_path.with_name(f"{sv_path.stem}.syn.v"),
        sv_path.with_name(f"{sv_path.stem}.syn.sv"),
    ):
        if artifact.exists():
            return artifact
    return None


def discover_candidates(source_roots: list[Path], include_proxy_only: bool) -> list[Candidate]:
    candidates: dict[str, Candidate] = {}

    for root in source_roots:
        if not root.exists():
            continue
        for sv_path in sorted(root.rglob("*.sv")):
            if sv_path.name.endswith(SKIP_SUFFIXES):
                continue

            ppa_path = sv_path.with_name(f"{sv_path.stem}_synthesis_report.ppa")
            proxy_json = sv_path.with_suffix(".proxy.json")
            syn_artifact = synthesis_artifact_for(sv_path)

            evidence_parts: list[str] = []
            if ppa_path.exists():
                evidence_parts.append("ppa")
            if syn_artifact is not None:
                evidence_parts.append("syn")
            if include_proxy_only and proxy_json.exists():
                evidence_parts.append("proxy")
            if not evidence_parts:
                continue

            candidate_id = make_candidate_id(root, sv_path)
            if candidate_id in candidates:
                root_tag = safe_name(root.name)
                candidate_id = safe_name(f"{root_tag}__{candidate_id}")

            candidates[candidate_id] = Candidate(
                candidate_id=candidate_id,
                problem=infer_problem(sv_path),
                generation=infer_generation(sv_path),
                source_sv=sv_path,
                evidence="+".join(evidence_parts),
                ppa_path=ppa_path if ppa_path.exists() else None,
                proxy_json=proxy_json if proxy_json.exists() else None,
            )

    return sorted(
        candidates.values(),
        key=lambda c: (c.problem, natural_int(c.generation), c.candidate_id),
    )


def collect_candidate(
    cand: Candidate,
    output_root: Path,
    top_modules: dict[str, str],
    overwrite: bool,
) -> ManifestRow:
    design_dir = output_root / cand.candidate_id
    output_sv = design_dir / f"{cand.candidate_id}.v"
    output_ppa = design_dir / "ppa.csv"
    output_metadata = design_dir / "metadata.json"

    note = ""
    if design_dir.exists() and not overwrite:
        note = "already exists; left unchanged"
    else:
        if design_dir.exists():
            shutil.rmtree(design_dir)
        design_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cand.source_sv, output_sv)

        metrics = None
        if cand.ppa_path is not None:
            metrics = parse_ppa_report(cand.ppa_path)
        if metrics is None and cand.proxy_json is not None:
            metrics = proxy_to_ppa(cand.proxy_json)
        if metrics is not None:
            write_ppa_csv(output_ppa, metrics)
        else:
            note = "no parseable PPA metrics"

        if cand.proxy_json is not None:
            shutil.copy2(cand.proxy_json, design_dir / f"{cand.candidate_id}.proxy.json")

        metadata = {
            "candidate_id": cand.candidate_id,
            "problem": cand.problem,
            "generation": cand.generation,
            "source_sv": str(cand.source_sv.resolve()),
            "source_ppa": str(cand.ppa_path.resolve()) if cand.ppa_path else "",
            "source_proxy_json": str(cand.proxy_json.resolve()) if cand.proxy_json else "",
            "evidence": cand.evidence,
            "top_module": top_modules.get(cand.problem, "TopModule"),
            "workspace_dir": str(design_dir.resolve()),
            "ppa_metrics": metrics or {},
        }
        write_json(output_metadata, metadata)

    return ManifestRow(
        candidate_id=cand.candidate_id,
        problem=cand.problem,
        generation=cand.generation,
        source_sv=str(cand.source_sv.resolve()),
        evidence=cand.evidence,
        output_dir=str(design_dir.resolve()),
        output_sv=str(output_sv.resolve()),
        output_ppa=str(output_ppa.resolve()) if output_ppa.exists() else "",
        output_metadata=str(output_metadata.resolve()) if output_metadata.exists() else "",
        source_ppa=str(cand.ppa_path.resolve()) if cand.ppa_path else "",
        source_proxy_json=str(cand.proxy_json.resolve()) if cand.proxy_json else "",
        top_module=top_modules.get(cand.problem, "TopModule"),
        note=note,
    )


def write_manifest(output_root: Path, rows: list[ManifestRow], summary: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ManifestRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    summary["manifest_csv"] = str(csv_path.resolve())
    write_json(output_root / "manifest.json", summary)


def main() -> int:
    args = parse_args()
    source_roots = [p.resolve() for p in (args.source_root or DEFAULT_SOURCE_ROOTS)]
    benchmark_roots = [p.resolve() for p in (args.benchmark_root or DEFAULT_BENCHMARK_ROOTS)]
    output_root = args.output_root.resolve()

    candidates = discover_candidates(source_roots, args.include_proxy_only)

    if args.dedup_by_content:
        deduped: list[Candidate] = []
        seen: set[str] = set()
        for cand in candidates:
            digest = sha256_file(cand.source_sv)
            if digest in seen:
                continue
            seen.add(digest)
            deduped.append(cand)
        candidates = deduped

    print("Aggregate synthesizable collection")
    print(f"Source roots: {', '.join(str(p) for p in source_roots)}")
    print(f"Discovered candidates: {len(candidates)}")
    print(f"Output root: {output_root}")

    if args.dry_run:
        return 0

    top_modules = load_top_modules(benchmark_roots)
    rows = [
        collect_candidate(cand, output_root, top_modules, args.overwrite)
        for cand in candidates
    ]

    by_evidence: dict[str, int] = {}
    for cand in candidates:
        by_evidence[cand.evidence] = by_evidence.get(cand.evidence, 0) + 1

    summary = {
        "source_roots": [str(p) for p in source_roots],
        "benchmark_roots": [str(p) for p in benchmark_roots],
        "output_root": str(output_root),
        "total_candidates": len(candidates),
        "evidence_counts": by_evidence,
        "dedup_by_content": args.dedup_by_content,
        "include_proxy_only": args.include_proxy_only,
    }
    write_manifest(output_root, rows, summary)

    print(f"Wrote {len(rows)} design folders.")
    print(f"Manifest: {output_root / 'manifest.csv'}")
    print(f"Summary: {output_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
