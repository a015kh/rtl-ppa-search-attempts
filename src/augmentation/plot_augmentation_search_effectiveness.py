from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import matplotlib.pyplot as plt


DEFAULT_RUNS = (
    (
        "aug_100",
        "data/dataset_v6.2.0_augmentation_random_000_099",
        "outputs/augmentation/eoh_v6.2.0_random_000_099",
        "data/dataset_v6.2.0_augmentation_random_000_099",
        "examples/symbolic/test_aug_100.json",
    ),
    (
        "aug_200",
        "data/dataset_v6.2.0_augmentation_random_100_299",
        "outputs/augmentation/eoh_v6.2.0_random_100_299",
        "data/dataset_v6.2.0_augmentation_random_000_099",
        "examples/symbolic/test_aug_200.json",
    ),
    (
        "pyra_200",
        "data/pyra_augmentation_random_000_199",
        "outputs/augmentation/eoh_pyra_random_000_199",
        "ldata/pyra_desc_yosys_filtered",
        "examples/symbolic/test_pyra_200.json",
    ),
)
PPA_KEYS = ("area", "power", "eff_clk_period")
TARGETS = ("area", "power", "delay")
VERILOG_COLUMNS = (
    "verilog",
    "verilog_code",
    "rtl",
    "code",
    "module",
    "original_verilog",
    "generated_verilog",
)


@dataclass(frozen=True)
class RunSpec:
    label: str
    record_path: Path
    work_dir: Path
    baseline_path: Path | None = None
    synth_json: Path | None = None


@dataclass
class Candidate:
    run_label: str
    task_key: str
    generation: int
    candidate_index: int
    target: str
    recorded_improvement: float | None
    synthesized_improvement: float | None
    synthesis_success: bool


def as_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def normalize_ppa(data: dict[str, Any] | None) -> dict[str, float | None]:
    data = data or {}
    eff_clk_period = data.get("eff_clk_period", data.get("delay", data.get("Delay")))
    return {
        "area": as_float(data.get("area", data.get("Total Area"))),
        "power": as_float(data.get("power", data.get("Total Power"))),
        "eff_clk_period": as_float(eff_clk_period),
    }


def normalize_verilog(verilog: str) -> str:
    text = verilog.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"//.*?$", "", text, flags=re.M)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def verilog_key(verilog: str) -> str:
    return hashlib.sha256(normalize_verilog(verilog).encode("utf-8")).hexdigest()


def first_present(row: dict[str, Any], columns: tuple[str, ...]) -> Any | None:
    for column in columns:
        value = row.get(column)
        if value not in (None, ""):
            return value
    return None


def average_improvement(
    original: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> float | None:
    original_ppa = normalize_ppa(original)
    candidate_ppa = normalize_ppa(candidate)
    values: list[float] = []
    for key in PPA_KEYS:
        ref = original_ppa[key]
        value = candidate_ppa[key]
        if ref in (None, 0) or value is None:
            continue
        values.append((ref - value) / abs(ref))
    return mean(values) if values else None


def target_metric_improvement(
    original: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    target: str,
) -> float | None:
    original_value = target_value(original, target)
    candidate_value = target_value(candidate, target)
    if original_value in (None, 0) or candidate_value is None:
        return None
    return (original_value - candidate_value) / abs(original_value)


def target_value(metrics: dict[str, Any] | None, target: str) -> float | None:
    ppa = normalize_ppa(metrics)
    if target == "delay":
        return ppa["eff_clk_period"]
    return ppa.get(target)


def normalize_target(target: Any) -> str:
    value = str(target or "").strip().lower()
    return "delay" if value in {"delay", "eff_clk_period", "period", "timing"} else value


def target_improvement(
    record: dict[str, Any],
    baseline_ppa: dict[str, Any] | None = None,
) -> float | None:
    explicit = as_float(record.get("improvement"))
    if explicit is not None and baseline_ppa is None:
        return explicit
    target = str(record.get("target") or "")
    original = target_value(baseline_ppa or record.get("original_ppa"), target)
    candidate = target_value(record.get("generated_ppa"), target)
    if original in (None, 0) or candidate is None:
        return None
    return (original - candidate) / abs(original)


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("records", "results", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [data]
    raise ValueError(f"Unsupported JSON root in {path}")


def dataset_file(path: Path) -> Path:
    if not path.exists() and path.parts and path.parts[0] == "ldata":
        data_path = Path("data", *path.parts[1:])
        if data_path.exists():
            path = data_path
    if path.is_file():
        return path
    for name in ("train.jsonl", "data.jsonl", "test.jsonl", "valid.jsonl"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No JSONL dataset file found under {path}")


def load_baseline_by_verilog(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    baseline_by_verilog: dict[str, dict[str, Any]] = {}
    try:
        rows = load_json_or_jsonl(dataset_file(path))
    except FileNotFoundError:
        return {}
    for row in rows:
        verilog = first_present(row, VERILOG_COLUMNS)
        if not verilog:
            continue
        metrics = normalize_ppa(
            row.get("ppa")
            or row.get("ppa_metrics")
            or row.get("original_ppa")
            or row
        )
        if any(metrics.get(key) is not None for key in PPA_KEYS):
            baseline_by_verilog[verilog_key(str(verilog))] = metrics
    return baseline_by_verilog


def candidate_names(record: dict[str, Any]) -> set[str]:
    sample_id = str(record.get("sample_id") or "")
    target = str(record.get("target") or "")
    generation = record.get("generation")
    candidate_index = record.get("candidate_index")
    names = {
        str(record.get("candidate_id") or ""),
        str(record.get("Name") or ""),
        str(record.get("verilog_path") or ""),
    }
    if sample_id and target and generation is not None and candidate_index is not None:
        names.add(f"{sample_id}:{target}:{generation}:{candidate_index}")
        names.add(f"{sample_id}_{target}_gen{generation}_cand{candidate_index}")
    path = str(record.get("verilog_path") or "")
    if path:
        names.add(Path(path).stem)
    return {name for name in names if name}


def synth_entry_names(row: dict[str, Any]) -> set[str]:
    names = {
        str(row.get("candidate_id") or ""),
        str(row.get("Name") or ""),
        str(row.get("name") or ""),
        str(row.get("verilog_path") or ""),
        str(row.get("path") or ""),
    }
    for key in ("verilog_path", "path", "Name", "candidate_id", "name"):
        value = row.get(key)
        if value:
            names.add(Path(str(value)).stem)
    return {name for name in names if name}


def best_synth_metrics(row: dict[str, Any], clock: float | None) -> dict[str, Any] | None:
    syn = row.get("Syn") or row.get("syn") or row.get("synthesis") or row.get("synth")
    if isinstance(syn, list):
        items = [
            item
            for item in syn
            if isinstance(item, dict)
            and (
                clock is None
                or as_float(item.get("Clock", item.get("clock"))) in (None, clock)
            )
        ]
        candidates = [normalize_ppa(item) for item in items]
    elif isinstance(syn, dict):
        candidates = [normalize_ppa(syn)]
    else:
        candidates = [normalize_ppa(row)]
    complete = [
        candidate
        for candidate in candidates
        if any(candidate.get(key) is not None for key in PPA_KEYS)
    ]
    if not complete:
        return None
    return min(
        complete,
        key=lambda metrics: sum(
            value for value in (metrics.get(key) for key in PPA_KEYS) if value is not None
        ),
    )


def load_synth_by_name(path: Path | None, clock: float | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    by_name: dict[str, dict[str, Any]] = {}
    for row in load_json_or_jsonl(path):
        metrics = best_synth_metrics(row, clock)
        if metrics is None:
            continue
        for name in synth_entry_names(row):
            by_name[name] = metrics
    return by_name


def load_run_candidates(spec: RunSpec, clock: float | None) -> list[Candidate]:
    output_jsonl = dataset_file(spec.record_path)
    records = load_json_or_jsonl(output_jsonl)
    synth_by_name = load_synth_by_name(spec.synth_json, clock)
    baseline_by_verilog = load_baseline_by_verilog(spec.baseline_path)
    candidates: list[Candidate] = []

    for record in records:
        generation = record.get("generation")
        candidate_index = record.get("candidate_index", 0)
        if generation is None:
            continue
        target = normalize_target(record.get("target"))
        if target not in TARGETS:
            continue
        synth_metrics = next(
            (synth_by_name[name] for name in candidate_names(record) if name in synth_by_name),
            None,
        )
        if synth_metrics is None and record.get("generated_ppa"):
            synth_metrics = normalize_ppa(record.get("generated_ppa"))
        baseline_ppa = None
        if record.get("original_verilog"):
            baseline_ppa = baseline_by_verilog.get(
                verilog_key(str(record["original_verilog"]))
            )
        if baseline_ppa is None:
            baseline_ppa = normalize_ppa(record.get("original_ppa"))
        module_key = str(record.get("sample_id") or "")
        if record.get("original_verilog"):
            module_key = verilog_key(str(record["original_verilog"]))

        candidates.append(
            Candidate(
                run_label=spec.label,
                task_key=f"{target}:{module_key}",
                generation=int(generation),
                candidate_index=int(candidate_index),
                target=target,
                recorded_improvement=target_improvement(record, baseline_ppa),
                synthesized_improvement=target_metric_improvement(
                    baseline_ppa,
                    synth_metrics,
                    target,
                ),
                synthesis_success=bool(record.get("synthesis_success")),
            )
        )
    return candidates


def running_best_by_task(
    candidates: Iterable[Candidate],
    attr: str,
) -> dict[int, float | None]:
    by_task_generation: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    generations: set[int] = set()
    for candidate in candidates:
        generations.add(candidate.generation)
        value = getattr(candidate, attr)
        if value is not None:
            by_task_generation[candidate.task_key][candidate.generation].append(value)

    best_by_task: dict[str, float | None] = {task: None for task in by_task_generation}
    result: dict[int, float | None] = {}
    for generation in sorted(generations):
        values: list[float] = []
        for task, generation_values in by_task_generation.items():
            if generation_values.get(generation):
                best = max(generation_values[generation])
                current = best_by_task[task]
                best_by_task[task] = best if current is None else max(current, best)
            if best_by_task[task] is not None:
                values.append(best_by_task[task])
        result[generation] = mean(values) if values else None
    return result


def build_progress_rows(candidates: list[Candidate]) -> list[dict[str, Any]]:
    by_run_target: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_run_target[(candidate.run_label, candidate.target)].append(candidate)

    rows: list[dict[str, Any]] = []
    for (run_label, target), run_candidates in sorted(by_run_target.items()):
        recorded = running_best_by_task(run_candidates, "recorded_improvement")
        synthesized = running_best_by_task(run_candidates, "synthesized_improvement")
        for generation in sorted(set(recorded) | set(synthesized)):
            rows.append(
                {
                    "run": run_label,
                    "target": target,
                    "generation": generation,
                    "recorded_best_mean_pct": pct(recorded.get(generation)),
                    "synthesized_best_mean_pct": pct(synthesized.get(generation)),
                }
            )
    return rows


def pct(value: float | None) -> float | None:
    return None if value is None else value * 100.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_progress(output_path: Path, rows: list[dict[str, Any]]) -> None:
    by_target_run: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_target_run[(row["target"], row["run"])].append(row)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2), sharey=False)
    fig.patch.set_facecolor("white")
    colors = {
        "aug_100": "#1b9e77",
        "aug_200": "#d95f02",
        "pyra_200": "#3e87e7",
    }
    target_titles = {"area": "Area", "power": "Power", "delay": "Delay"}
    for ax, target in zip(axes, TARGETS):
        ax.set_facecolor("#fbfbfb")
        for run_label in sorted({row["run"] for row in rows}):
            run_rows = by_target_run.get((target, run_label), [])
            if not run_rows:
                continue
            run_rows = sorted(run_rows, key=lambda item: int(item["generation"]))
            x_values = [int(item["generation"]) for item in run_rows]
            recorded = [item["recorded_best_mean_pct"] for item in run_rows]
            synthesized = [item["synthesized_best_mean_pct"] for item in run_rows]
            color = colors.get(run_label)
            ax.plot(
                x_values,
                recorded,
                linewidth=2.2,
                marker="o",
                markersize=3.0,
                color=color,
                label=f"{run_label} recorded",
            )
            ax.plot(
                x_values,
                synthesized,
                linewidth=2.0,
                linestyle=":",
                color=color,
                label=f"{run_label} synthesized",
            )

        ax.axhline(0, color="#777777", linewidth=1.0, alpha=0.65)
        ax.set_title(target_titles[target], fontsize=13, pad=10)
        ax.set_xlabel("Generation", fontsize=11)
        ax.grid(True, color="#d2d2d2", linewidth=0.7, alpha=0.45)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Mean target improvement over baseline (%)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Data Augmentation Search Effectiveness by Target", fontsize=15, y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_run(value: str) -> RunSpec:
    parts = value.split(":", 4)
    if len(parts) < 3:
        raise argparse.ArgumentTypeError(
            "--run must be LABEL:RECORD_PATH:WORK_DIR, LABEL:RECORD_PATH:WORK_DIR:BASELINE, or LABEL:RECORD_PATH:WORK_DIR:BASELINE:SYNTH_JSON"
        )
    return RunSpec(
        label=parts[0],
        record_path=Path(parts[1]),
        work_dir=Path(parts[2]),
        baseline_path=Path(parts[3]) if len(parts) >= 4 and parts[3] else None,
        synth_json=Path(parts[4]) if len(parts) == 5 and parts[4] else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot data-augmentation evolutionary search effectiveness from "
            "run_data_augmentation.py work directories and synthesized result JSON."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        help="Run spec: LABEL:RECORD_PATH:WORK_DIR:BASELINE_PATH:SYNTH_JSON. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/augmentation/search_effectiveness_report"),
    )
    parser.add_argument(
        "--prefix",
        default="augmentation_search_effectiveness",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--clock",
        type=float,
        default=1.0,
        help="Clock period to use from synthesized result JSON Syn entries. Use a negative value to disable filtering.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    specs = args.run or [
        RunSpec(label, Path(record_path), Path(work_dir), Path(baseline_path), Path(synth_json))
        for label, record_path, work_dir, baseline_path, synth_json in DEFAULT_RUNS
    ]

    clock = None if args.clock < 0 else args.clock
    candidates: list[Candidate] = []
    missing: list[str] = []
    for spec in specs:
        try:
            dataset_file(spec.record_path)
        except FileNotFoundError as exc:
            missing.append(str(exc))
            continue
        candidates.extend(load_run_candidates(spec, clock))

    if not candidates:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"No augmentation records found. Missing:\n{details}")

    rows = build_progress_rows(candidates)
    if not rows:
        raise RuntimeError("No progress rows could be computed from augmentation records.")

    csv_path = args.output_dir / f"{args.prefix}.csv"
    png_path = args.output_dir / f"{args.prefix}.png"
    write_csv(csv_path, rows)
    plot_progress(png_path, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    if missing:
        print("Skipped missing augmentation records:")
        for path in missing:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
