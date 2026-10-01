#!/usr/bin/env python3
"""Compare proxy PPA predictions with DC/PrimeTime for a four-pipeline run.

The four-pipeline search evaluates every node with one of three proxies and
only sends finalists through the commercial evaluator.  This report joins the
proxy points with the golden ``dc_pt`` points, computes prediction and ranking
statistics, and lists good golden candidates that the proxy ranking missed.

By default this is a read-only report over the points already in the run
database.  ``--synthesize-missing`` evaluates every candidate which does not
yet have a ``dc_pt`` result, reusing all VCS/DC/PT caches in the run directory.
That option can be expensive, but it is the way to make the missed-best report
complete rather than limited to finalists.

Example:

    python evolve/compare_proxy_accuracy.py \
      --run-dir evolve/runs/q3_hc_four_pipeline_ppa_60 \
      --synthesize-missing
"""

from __future__ import annotations

import os
import sys

# A failed dynamic load leaves the incompatible libsqlite3 in the process, so
# simply deleting LD_LIBRARY_PATH after importing sqlite3 is too late.  Restart
# once with a clean loader path before importing any extension modules.  The
# commercial tools get their own environment from load_eda_environment() later.
if os.environ.get("ICRTL_SQLITE_ENV_CLEAN") != "1" and os.environ.get("LD_LIBRARY_PATH"):
    clean_env = os.environ.copy()
    clean_env.pop("LD_LIBRARY_PATH", None)
    clean_env["ICRTL_SQLITE_ENV_CLEAN"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], clean_env)

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import importlib
from pathlib import Path
import statistics
from typing import Any, Iterable


def _load_sqlite3():
    """Import sqlite3 without letting an EDA libsqlite3 shadow Python's one.

    Synopsys/Verdi environments sometimes prepend a private, older SQLite
    library to ``LD_LIBRARY_PATH``.  The system Python extension then fails at
    import time (for example, ``undefined symbol: sqlite3_trace_v2``).  The
    The report process runs with the clean loader path; DC/PT commands still
    receive the normal EDA environment later through four_pipeline.py.
    """
    return importlib.import_module("sqlite3")


sqlite3 = _load_sqlite3()


DEFAULT_PIPELINES = ("yosys_openroad", "trained_model", "yosys_trained_model")
DEFAULT_PERIODS = (1.0, 3.0, 5.0, 8.0, 10.0, 15.0)
DEFAULT_MARGINS = (0.05, 0.10, 0.20)
METRICS = {
    "area_um2": "area",
    "delay_ns": "delay",
    "power_mw": "power",
}


def as_float(value: Any) -> float | None:
    if value in (None, "", "nan", "NaN", "None"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_periods(raw: str) -> list[float]:
    try:
        periods = [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("periods must be comma-separated numbers") from exc
    if not periods or any(value <= 0 for value in periods):
        raise argparse.ArgumentTypeError("periods must be positive")
    return periods


def parse_pipelines(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = set(DEFAULT_PIPELINES)
    unknown = sorted(set(values) - allowed)
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            "pipelines must be selected from " + ",".join(DEFAULT_PIPELINES)
        )
    return values


def parse_margins(raw: str) -> list[float]:
    """Parse percentage margins (``5,10,20``) into fractions."""
    try:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "margins must be comma-separated percentages, e.g. 5,10,20"
        ) from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("margins must be positive")
    if any(value > 100.0 for value in values):
        raise argparse.ArgumentTypeError("margins cannot exceed 100%")
    fractions = [value / 100.0 for value in values]
    return list(dict.fromkeys(fractions))


def margin_tag(margin: float) -> str:
    """Return a stable field-name tag, such as ``5`` or ``2p5``."""
    percent = margin * 100.0
    text = f"{percent:g}"
    return text.replace(".", "p")


def margin_label(margin: float) -> str:
    return f"{margin * 100:g}%"


def accuracy_at_margin(errors: list[float], margin: float) -> float | None:
    if not errors:
        return None
    return 100.0 * sum(abs(error) <= margin for error in errors) / len(errors)


def json_load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return default


def load_periods(db: sqlite3.Connection, requested: list[float] | None) -> list[float]:
    if requested:
        return requested
    row = db.execute("SELECT value FROM metadata WHERE key='config'").fetchone()
    if row:
        try:
            configured = json.loads(row[0]).get("periods")
            if configured:
                return [float(value) for value in configured]
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    return list(DEFAULT_PERIODS)


def rows_by_key(db: sqlite3.Connection, pipeline: str) -> dict[tuple[str, float], sqlite3.Row]:
    rows = db.execute(
        "SELECT * FROM point_results WHERE pipeline=?", (pipeline,)
    ).fetchall()
    return {(str(row["candidate_sha256"]), float(row["period_ns"])): row for row in rows}


def candidate_catalog(
    db: sqlite3.Connection,
    pipelines: Iterable[str],
    baseline: Path | None,
) -> dict[str, dict[str, Path]]:
    """Return pipeline -> candidate digest -> source path."""
    result: dict[str, dict[str, Path]] = {pipeline: {} for pipeline in pipelines}
    placeholders = ",".join("?" for _ in pipelines)
    rows = db.execute(
        f"SELECT pipeline,candidate_sha256,candidate_path FROM nodes_v2 "
        f"WHERE pipeline IN ({placeholders}) ORDER BY pipeline,chain_id,node_index",
        tuple(pipelines),
    ).fetchall()
    for row in rows:
        path = Path(row["candidate_path"])
        if path.is_file():
            result[row["pipeline"]].setdefault(row["candidate_sha256"], path)

    if baseline is not None and baseline.is_file():
        digest = sha256_file(baseline)
        for pipeline in result:
            result[pipeline].setdefault(digest, baseline)
    return result


def missing_golden_digests(
    db_path: Path,
    candidates: dict[str, dict[str, Path]],
    periods: list[float],
) -> list[tuple[str, Path]]:
    all_candidates: dict[str, Path] = {}
    for pipeline_candidates in candidates.values():
        for digest, path in pipeline_candidates.items():
            all_candidates.setdefault(digest, path)
    db = sqlite3.connect(db_path)
    try:
        present = {
            (str(row[0]), float(row[1]))
            for row in db.execute(
                "SELECT candidate_sha256,period_ns FROM point_results "
                "WHERE pipeline='dc_pt'"
            )
        }
    finally:
        db.close()
    # PipelineEvaluator evaluates all requested periods for a candidate.  A
    # digest with a partial/interrupted golden run must therefore be retried.
    return [
        (digest, path)
        for digest, path in all_candidates.items()
        if any((digest, period) not in present for period in periods)
    ]


def synthesize_missing(
    run_dir: Path,
    db_path: Path,
    candidates: dict[str, dict[str, Path]],
    periods: list[float],
    device: str,
) -> None:
    """Run the existing DC/PT evaluator for candidates without golden points."""
    missing = missing_golden_digests(db_path, candidates, periods)
    if not missing:
        print("[golden] all candidates already have dc_pt results", flush=True)
        return

    # four_pipeline.py is written to be run from the evolve directory.  Add
    # that directory explicitly so this script also works from the repository
    # root without turning evolve into a package.
    evolve_dir = str(Path(__file__).resolve().parent)
    if evolve_dir not in sys.path:
        sys.path.insert(0, evolve_dir)
    from .four_pipeline import (  # pylint: disable=import-outside-toplevel
        FunctionalVCS,
        PipelineEvaluator,
        RAW_CHECKPOINT,
        State,
        YOSYS_CHECKPOINT,
        load_eda_environment,
    )

    env = load_eda_environment()
    state = State(db_path)
    vcs = FunctionalVCS(state, run_dir, env, periods)
    evaluator = PipelineEvaluator(
        "dc_pt", state, run_dir, env, periods, vcs, device,
        RAW_CHECKPOINT, YOSYS_CHECKPOINT,
    )
    try:
        print(f"[golden] synthesizing {len(missing)} missing candidates with DC/PT", flush=True)
        for index, (digest, path) in enumerate(missing, start=1):
            print(
                f"[golden {index}/{len(missing)}] {digest[:12]} {path}",
                flush=True,
            )
            evaluator.evaluate(path, -2, index)
    finally:
        evaluator.close()
        state.close("completed")


def relative_error(proxy: Any, golden: Any) -> float | None:
    proxy_value, golden_value = as_float(proxy), as_float(golden)
    if proxy_value is None or golden_value is None or golden_value == 0:
        return None
    return (proxy_value - golden_value) / golden_value


def rank_desc(values: list[tuple[str, float]]) -> dict[str, float]:
    """Average-tie ranks, with rank 1 assigned to the largest value."""
    ordered = sorted(values, key=lambda item: item[1], reverse=True)
    result: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for digest, _ in ordered[index:end]:
            result[digest] = rank
        index = end
    return result


def spearman(proxy: list[float], golden: list[float]) -> float | None:
    if len(proxy) != len(golden) or len(proxy) < 2:
        return None
    proxy_mean, golden_mean = statistics.mean(proxy), statistics.mean(golden)
    numerator = sum((p - proxy_mean) * (g - golden_mean) for p, g in zip(proxy, golden))
    p_denominator = math.sqrt(sum((p - proxy_mean) ** 2 for p in proxy))
    g_denominator = math.sqrt(sum((g - golden_mean) ** 2 for g in golden))
    if p_denominator == 0 or g_denominator == 0:
        return None
    return numerator / (p_denominator * g_denominator)


def selected_digests(run_dir: Path, pipeline: str) -> set[str]:
    rows = json_load(run_dir / "finalists.json", [])
    selected = {
        row.get("candidate_sha256")
        for row in rows
        if row.get("source_pipeline") == pipeline and row.get("candidate_sha256")
    }
    return {str(value) for value in selected}


def make_records(
    db: sqlite3.Connection,
    candidates: dict[str, dict[str, Path]],
    periods: list[float],
) -> list[dict[str, Any]]:
    golden_rows = rows_by_key(db, "dc_pt")
    records: list[dict[str, Any]] = []
    for pipeline, pipeline_candidates in candidates.items():
        proxy_rows = rows_by_key(db, pipeline)
        for digest, path in pipeline_candidates.items():
            for period in periods:
                proxy = proxy_rows.get((digest, period))
                golden = golden_rows.get((digest, period))
                record: dict[str, Any] = {
                    "pipeline": pipeline,
                    "candidate_sha256": digest,
                    "candidate_path": str(path),
                    "period_ns": period,
                    "is_baseline": int(path.name == "initial.sv" and "/VCS/" in str(path)),
                    "proxy_status": proxy["status"] if proxy else "missing",
                    "golden_status": golden["status"] if golden else "missing",
                    "proxy_score": proxy["score"] if proxy else None,
                    "golden_score": golden["score"] if golden else None,
                    "proxy_area_um2": proxy["area_um2"] if proxy else None,
                    "golden_area_um2": golden["area_um2"] if golden else None,
                    "proxy_delay_ns": proxy["delay_ns"] if proxy else None,
                    "golden_delay_ns": golden["delay_ns"] if golden else None,
                    "proxy_power_mw": proxy["power_mw"] if proxy else None,
                    "golden_power_mw": golden["power_mw"] if golden else None,
                    "area_relative_error": relative_error(
                        proxy["area_um2"] if proxy else None,
                        golden["area_um2"] if golden else None,
                    ),
                    "delay_relative_error": relative_error(
                        proxy["delay_ns"] if proxy else None,
                        golden["delay_ns"] if golden else None,
                    ),
                    "power_relative_error": relative_error(
                        proxy["power_mw"] if proxy else None,
                        golden["power_mw"] if golden else None,
                    ),
                }
                records.append(record)

    # Add ranks independently for each pipeline and clock.  A point can have
    # metrics but no score (for example, a predicted timing violation), so
    # ranking deliberately follows the score used by the search.
    for pipeline in candidates:
        for period in periods:
            group = [
                row for row in records
                if row["pipeline"] == pipeline and row["period_ns"] == period
            ]
            proxy_rank = rank_desc([
                (row["candidate_sha256"], float(row["proxy_score"]))
                for row in group if as_float(row["proxy_score"]) is not None
            ])
            golden_rank = rank_desc([
                (row["candidate_sha256"], float(row["golden_score"]))
                for row in group if as_float(row["golden_score"]) is not None
            ])
            for row in group:
                row["proxy_rank"] = proxy_rank.get(row["candidate_sha256"])
                row["golden_rank"] = golden_rank.get(row["candidate_sha256"])
    return records


def summary_rows(
    records: list[dict[str, Any]],
    pipelines: Iterable[str],
    periods: Iterable[float],
    top_k: int,
    margins: Iterable[float],
) -> list[dict[str, Any]]:
    margins = list(margins)
    summaries: list[dict[str, Any]] = []
    for pipeline in pipelines:
        for period in periods:
            group = [
                row for row in records
                if row["pipeline"] == pipeline and row["period_ns"] == period
            ]
            summary: dict[str, Any] = {
                "pipeline": pipeline,
                "period_ns": period,
                "candidates": len(group),
                "golden_points": sum(row["golden_score"] is not None for row in group),
                "proxy_points": sum(row["proxy_score"] is not None for row in group),
            }
            metric_errors: dict[str, list[float]] = {}
            for field, name in METRICS.items():
                errors = [
                    float(row[f"{name}_relative_error"])
                    for row in group if row[f"{name}_relative_error"] is not None
                ]
                metric_errors[name] = errors
                summary[f"{name}_n"] = len(errors)
                summary[f"{name}_mape_percent"] = (
                    100.0 * statistics.mean(abs(value) for value in errors)
                    if errors else None
                )
                summary[f"{name}_median_ape_percent"] = (
                    100.0 * statistics.median(abs(value) for value in errors)
                    if errors else None
                )
                summary[f"{name}_bias_percent"] = (
                    100.0 * statistics.mean(errors) if errors else None
                )
            all_metric_errors = [
                [float(row[f"{name}_relative_error"]) for name in METRICS.values()]
                for row in group
                if all(row[f"{name}_relative_error"] is not None for name in METRICS.values())
            ]
            for margin in margins:
                tag = margin_tag(margin)
                for name, errors in metric_errors.items():
                    summary[f"{name}_acc_at_{tag}_percent"] = accuracy_at_margin(errors, margin)
                summary[f"acc_at_{tag}_percent"] = (
                    100.0 * sum(
                        all(abs(error) <= margin for error in errors)
                        for errors in all_metric_errors
                    ) / len(all_metric_errors)
                    if all_metric_errors else None
                )

            clean_pairs = [
                row for row in group
                if row["proxy_status"] != "missing" and row["golden_status"] != "missing"
            ]
            if clean_pairs:
                correct = sum(
                    (row["proxy_status"] == "success") == (row["golden_status"] == "success")
                    for row in clean_pairs
                )
                summary["timing_clean_accuracy_percent"] = 100.0 * correct / len(clean_pairs)
            else:
                summary["timing_clean_accuracy_percent"] = None

            ranked = [
                row for row in group
                if as_float(row["proxy_score"]) is not None
                and as_float(row["golden_score"]) is not None
            ]
            summary["ranked_points"] = len(ranked)
            summary["spearman"] = spearman(
                [float(row["proxy_rank"]) for row in ranked],
                [float(row["golden_rank"]) for row in ranked],
            ) if ranked else None
            # Recall is measured against the complete golden ranking.  A
            # proxy-failed candidate remains in the golden denominator; it
            # must not disappear merely because the proxy could not score it.
            proxy_order = [row["candidate_sha256"] for row in sorted(
                [row for row in group if as_float(row["proxy_score"]) is not None],
                key=lambda row: float(row["proxy_score"]), reverse=True,
            )]
            golden_order = [row["candidate_sha256"] for row in sorted(
                [row for row in group if as_float(row["golden_score"]) is not None],
                key=lambda row: float(row["golden_score"]), reverse=True,
            )]
            proxy_top = set(proxy_order[:top_k])
            golden_top = set(golden_order[:top_k])
            summary["top_k"] = top_k
            summary["top_k_recall_percent"] = (
                100.0 * len(proxy_top & golden_top) / len(golden_top)
                if golden_top else None
            )
            summary["proxy_best_digest"] = proxy_order[0] if proxy_order else None
            summary["golden_best_digest"] = golden_order[0] if golden_order else None
            summary["best_match"] = bool(proxy_order and golden_order and proxy_order[0] == golden_order[0])
            summary["missed_golden_top_k"] = sum(
                row["golden_rank"] is not None
                and row["golden_rank"] <= top_k
                and (row["proxy_rank"] is None or row["proxy_rank"] > top_k)
                for row in group
            )
            summaries.append(summary)
    return summaries


def overall_summary_rows(
    records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    pipelines: Iterable[str],
    margins: Iterable[float],
) -> list[dict[str, Any]]:
    """Aggregate metric errors and ranking statistics across all periods."""
    margins = list(margins)
    result: list[dict[str, Any]] = []
    for pipeline in pipelines:
        group = [row for row in records if row["pipeline"] == pipeline]
        output: dict[str, Any] = {
            "pipeline": pipeline,
            "periods": len({row["period_ns"] for row in group}),
            "candidate_points": len(group),
            "golden_points": sum(row["golden_score"] is not None for row in group),
            "proxy_points": sum(row["proxy_score"] is not None for row in group),
        }
        metric_errors: dict[str, list[float]] = {}
        for field, name in METRICS.items():
            errors = [
                float(row[f"{name}_relative_error"])
                for row in group if row[f"{name}_relative_error"] is not None
            ]
            metric_errors[name] = errors
            output[f"{name}_n"] = len(errors)
            output[f"{name}_mape_percent"] = (
                100.0 * statistics.mean(abs(value) for value in errors)
                if errors else None
            )
            output[f"{name}_median_ape_percent"] = (
                100.0 * statistics.median(abs(value) for value in errors)
                if errors else None
            )
        all_metric_errors = [
            [float(row[f"{name}_relative_error"]) for name in METRICS.values()]
            for row in group
            if all(row[f"{name}_relative_error"] is not None for name in METRICS.values())
        ]
        for margin in margins:
            tag = margin_tag(margin)
            for name, errors in metric_errors.items():
                output[f"{name}_acc_at_{tag}_percent"] = accuracy_at_margin(errors, margin)
            output[f"acc_at_{tag}_percent"] = (
                100.0 * sum(
                    all(abs(error) <= margin for error in errors)
                    for errors in all_metric_errors
                ) / len(all_metric_errors)
                if all_metric_errors else None
            )
        pairs = [
            row for row in group
            if row["proxy_status"] != "missing" and row["golden_status"] != "missing"
        ]
        output["timing_clean_accuracy_percent"] = (
            100.0 * sum(
                (row["proxy_status"] == "success") == (row["golden_status"] == "success")
                for row in pairs
            ) / len(pairs)
            if pairs else None
        )
        pipeline_summaries = [row for row in summaries if row["pipeline"] == pipeline]
        spearmans = [
            float(row["spearman"])
            for row in pipeline_summaries if row["spearman"] is not None
        ]
        recalls = [
            float(row["top_k_recall_percent"])
            for row in pipeline_summaries if row["top_k_recall_percent"] is not None
        ]
        output["mean_spearman"] = statistics.mean(spearmans) if spearmans else None
        output["mean_top_k_recall_percent"] = statistics.mean(recalls) if recalls else None
        output["missed_golden_top_k"] = sum(
            int(row["missed_golden_top_k"] or 0) for row in pipeline_summaries
        )
        result.append(output)
    return result


def missed_best_rows(
    records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    top_k: int,
    selected: dict[str, set[str]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in records:
        if row.get("golden_rank") is None or row["golden_rank"] > top_k:
            continue
        proxy_rank = row.get("proxy_rank")
        if proxy_rank is not None and proxy_rank <= top_k:
            continue
        row = dict(row)
        row["selected_finalist"] = int(
            row["candidate_sha256"] in selected.get(row["pipeline"], set())
        )
        row["reason"] = "golden_top_k_missed_by_proxy"
        result.append(row)
    result.sort(key=lambda item: (item["pipeline"], item["period_ns"], item["golden_rank"]))
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


TIMING_PARENT_STAGES = {
    "pipeline_search", "vcs_functional", "pipeline_point", "golden_validation",
    "report_generation", "preflight", "finalist_selection",
}


def timing_category(pipeline: str, stage: str) -> str:
    if stage in {"idea_generation", "codex_root_generation", "codex_refinement"}:
        return "generation"
    if stage in {"vcs_compile", "vcs_simulation"}:
        return "simulation"
    if stage in {"model_checkpoint_load"}:
        return "model_startup"
    if stage in {"gate_vcd_compile", "gate_vcd_simulation", "pt_vcd"}:
        return "golden_validation"
    if pipeline == "yosys_openroad" and stage in {"yosys", "openroad"}:
        return "synthesis_proxy"
    if pipeline == "trained_model" and stage in {
        "model_input_prepare", "model_inference", "model_decode",
    }:
        return "synthesis_proxy"
    if pipeline == "yosys_trained_model" and stage in {
        "yosys", "model_input_prepare", "model_inference", "model_decode",
    }:
        return "synthesis_proxy"
    if pipeline == "dc_pt" and stage in {"dc", "pt_fixed"}:
        return "synthesis_proxy"
    return "other"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def write_timing_reports(db_path: Path, output_dir: Path) -> None:
    """Export stage-level and category-level wall-clock timing reports."""
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    events = [
        row for row in db.execute(
            "SELECT pipeline,stage,wall_seconds,status FROM timing_events "
            "WHERE wall_seconds IS NOT NULL"
        )
        if row["stage"] not in TIMING_PARENT_STAGES
    ]
    db.close()

    stage_groups: dict[tuple[str, str], list[float]] = {}
    category_groups: dict[tuple[str, str], list[float]] = {}
    for row in events:
        pipeline = row["pipeline"] or "shared"
        stage = str(row["stage"])
        seconds = float(row["wall_seconds"] or 0.0)
        stage_groups.setdefault((pipeline, stage), []).append(seconds)
        category_groups.setdefault((pipeline, timing_category(pipeline, stage)), []).append(seconds)

    stage_rows: list[dict[str, Any]] = []
    for (pipeline, stage), values in sorted(stage_groups.items()):
        stage_rows.append({
            "pipeline": pipeline,
            "stage": stage,
            "calls": len(values),
            "total_seconds": sum(values),
            "mean_seconds": statistics.mean(values),
            "median_seconds": statistics.median(values),
            "p95_seconds": percentile(values, 0.95),
            "max_seconds": max(values),
        })

    category_rows: list[dict[str, Any]] = []
    for (pipeline, category), values in sorted(category_groups.items()):
        category_rows.append({
            "pipeline": pipeline,
            "category": category,
            "calls": len(values),
            "total_seconds": sum(values),
        })
    totals: dict[str, float] = {}
    for row in category_rows:
        totals[row["pipeline"]] = totals.get(row["pipeline"], 0.0) + row["total_seconds"]
    for row in category_rows:
        total = totals[row["pipeline"]]
        row["share_percent"] = 100.0 * row["total_seconds"] / total if total else None

    write_csv(output_dir / "timing_stages.csv", stage_rows)
    write_csv(output_dir / "timing_breakdown.csv", category_rows)

    pipelines = sorted(totals)
    categories = ["generation", "simulation", "synthesis_proxy", "model_startup", "golden_validation", "other"]
    lines = [
        "# Wall-clock timing report",
        "",
        "Durations are derived from `timing_events`; parent wrapper events are "
        "excluded to avoid double-counting their child stages.",
        "",
        "| Pipeline | Generation | Simulation | Synthesis/proxy | Model startup | Golden validation | Other | Accounted total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_category = {(row["pipeline"], row["category"]): row for row in category_rows}
    for pipeline in pipelines:
        values = [by_category.get((pipeline, category), {}).get("total_seconds", 0.0) for category in categories]
        lines.append(
            "| {pipeline} | {generation} | {simulation} | {synthesis} | {startup} | "
            "{golden} | {other} | {total} |".format(
                pipeline=pipeline,
                generation=fmt(values[0]), simulation=fmt(values[1]),
                synthesis=fmt(values[2]), startup=fmt(values[3]),
                golden=fmt(values[4]), other=fmt(values[5]),
                total=fmt(sum(values)),
            )
        )
    lines.extend([
        "",
        "Stage-level distributions are in `timing_stages.csv`; category totals "
        "and shares are in `timing_breakdown.csv`.",
        "",
        "The 100% stacked chart is `timing_phase_distribution.svg`.",
    ])
    (output_dir / "timing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any, digits: int = 2) -> str:
    number = as_float(value)
    if number is None:
        return "—"
    return f"{number:.{digits}f}"


def margin_cell(row: dict[str, Any], margin: float) -> str:
    """Format area/delay/power/all-metric acc@margin for Markdown."""
    tag = margin_tag(margin)
    values = [
        row.get(f"{name}_acc_at_{tag}_percent")
        for name in METRICS.values()
    ]
    values.append(row.get(f"acc_at_{tag}_percent"))
    return "/".join(fmt(value) for value in values)


def write_markdown(
    path: Path,
    run_dir: Path,
    overall: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    missed: list[dict[str, Any]],
    unsynthesized: list[dict[str, Any]],
    top_k: int,
    margins: list[float],
) -> None:
    margin_headers = " | ".join(
        f"acc@{margin_label(margin)} (A/D/P/all)" for margin in margins
    )
    margin_separator = "|".join("---:" for _ in margins)
    lines = [
        "# Proxy PPA accuracy report",
        "",
        f"Run: `{run_dir}`",
        "",
        "DC/PrimeTime is treated as the golden evaluator. Errors are "
        "`(proxy - golden) / golden`; lower absolute error is better.",
        "`acc@margin` is the percentage of comparable predictions whose absolute "
        "relative error is within the margin for all three PPA metrics.",
        "",
        "## Summary",
        "",
        "### Overall",
        "",
        f"| Pipeline | Golden points | Area MAPE | Delay MAPE | Power MAPE | Timing classification | Mean Spearman | Mean top-k recall | {margin_headers} | Missed top-k |",
        f"|---|---:|---:|---:|---:|---:|---:|---:|{margin_separator}|---:|",
    ]
    for row in overall:
        margin_values = " | ".join(
            margin_cell(row, margin)
            for margin in margins
        )
        lines.append(
            "| {pipeline} | {golden_points} | {area} | {delay} | {power} | {timing} | "
            "{spearman} | {recall} | {margin_values} | {missed} |".format(
                pipeline=row["pipeline"], golden_points=row["golden_points"],
                area=fmt(row["area_mape_percent"]),
                delay=fmt(row["delay_mape_percent"]),
                power=fmt(row["power_mape_percent"]),
                timing=fmt(row["timing_clean_accuracy_percent"]),
                spearman=fmt(row["mean_spearman"], 3),
                recall=fmt(row["mean_top_k_recall_percent"]),
                margin_values=margin_values,
                missed=row["missed_golden_top_k"],
            )
        )
    lines.extend([
        "",
        "### Per period",
        "",
        f"| Pipeline | Period | Points | Area MAPE | Delay MAPE | Power MAPE | Timing classification | Spearman | Top-k recall | {margin_headers} | Best match |",
        f"|---|---:|---:|---:|---:|---:|---:|---:|---:|{margin_separator}|---:|",
    ])
    for row in summaries:
        margin_values = " | ".join(
            margin_cell(row, margin)
            for margin in margins
        )
        lines.append(
            "| {pipeline} | {period_ns:g} | {ranked_points} | {area} | {delay} | {power} | "
            "{timing} | {spearman} | {recall} | {margin_values} | {best} |".format(
                pipeline=row["pipeline"], period_ns=row["period_ns"],
                ranked_points=row["ranked_points"],
                area=fmt(row["area_mape_percent"]),
                delay=fmt(row["delay_mape_percent"]),
                power=fmt(row["power_mape_percent"]),
                timing=fmt(row["timing_clean_accuracy_percent"]),
                spearman=fmt(row["spearman"], 3),
                recall=fmt(row["top_k_recall_percent"]),
                margin_values=margin_values,
                best="yes" if row["best_match"] else "no",
            )
        )

    lines.extend([
        "",
        f"## Missing golden best candidates (proxy top-{top_k})",
        "",
    ])
    if missed:
        lines.extend([
            "| Pipeline | Period | Golden rank | Proxy rank | Golden score | Proxy score | Candidate | Selected finalist |",
            "|---|---:|---:|---:|---:|---:|---|---:|",
        ])
        for row in missed:
            lines.append(
                f"| {row['pipeline']} | {row['period_ns']:g} | {fmt(row['golden_rank'], 1)} | "
                f"{fmt(row.get('proxy_rank'), 1)} | {fmt(row.get('golden_score'), 3)} | "
                f"{fmt(row.get('proxy_score'), 3)} | `{row['candidate_sha256'][:12]}` | "
                f"{row['selected_finalist']} |"
            )
    else:
        lines.append("No golden top-k candidates were missed by the proxy ranking.")

    lines.extend(["", "## Golden coverage", ""])
    if unsynthesized:
        lines.append(
            f"{len(unsynthesized)} candidate-period points have no DC/PT result. "
            "The missed-best analysis is incomplete until they are synthesized."
        )
    else:
        lines.append("Every candidate-period point has a DC/PT result.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pipelines", type=parse_pipelines, default=list(DEFAULT_PIPELINES))
    parser.add_argument("--periods", type=parse_periods, default=None)
    parser.add_argument("--baseline", type=Path, default=Path("VCS/Q3_HC/initial.sv"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--synthesize-missing", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--margins", type=parse_margins, default=list(DEFAULT_MARGINS),
        help="relative-error margins in percent, e.g. 5,10,20 (default: 5,10,20)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    run_dir = args.run_dir.resolve()
    db_path = run_dir / "state_v2.sqlite3"
    if not db_path.is_file():
        raise SystemExit(f"missing run database: {db_path}")
    output_dir = (args.output_dir or run_dir / "proxy_accuracy").resolve()

    baseline = args.baseline
    if not baseline.is_absolute():
        candidates_for_baseline = [
            (Path.cwd() / baseline).resolve(),
            (Path(__file__).resolve().parent.parent / baseline).resolve(),
        ]
        baseline = next(
            (path for path in candidates_for_baseline if path.is_file()),
            candidates_for_baseline[0],
        )

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    periods = load_periods(db, args.periods)
    candidates = candidate_catalog(db, args.pipelines, baseline)
    total_candidates = len({digest for values in candidates.values() for digest in values})
    print(
        f"[report] pipelines={','.join(args.pipelines)} candidates={total_candidates} "
        f"periods={','.join(f'{p:g}' for p in periods)}",
        flush=True,
    )
    db.close()

    if args.synthesize_missing:
        synthesize_missing(run_dir, db_path, candidates, periods, args.device)

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    records = make_records(db, candidates, periods)
    db.close()
    summaries = summary_rows(records, args.pipelines, periods, args.top_k, args.margins)
    overall = overall_summary_rows(records, summaries, args.pipelines, args.margins)
    selected = {pipeline: selected_digests(run_dir, pipeline) for pipeline in args.pipelines}
    missed = missed_best_rows(records, summaries, args.top_k, selected)
    unsynthesized = [
        row for row in records if row["golden_status"] == "missing"
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_timing_reports(db_path, output_dir)
    write_csv(output_dir / "candidate_points.csv", records)
    write_csv(output_dir / "accuracy_summary.csv", summaries)
    write_csv(output_dir / "overall_summary.csv", overall)
    write_csv(output_dir / "missed_best.csv", missed)
    write_csv(output_dir / "unsynthesized_points.csv", unsynthesized)
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "pipelines": args.pipelines,
        "periods": periods,
        "margins": args.margins,
        "top_k": args.top_k,
        "candidate_count": total_candidates,
        "candidate_point_count": len(records),
        "unsynthesized_point_count": len(unsynthesized),
        "summaries": summaries,
        "overall": overall,
        "missed_best_count": len(missed),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(
        output_dir / "report.md", run_dir, overall, summaries, missed,
        unsynthesized, args.top_k, args.margins,
    )

    print(f"[report] wrote {output_dir / 'report.md'}", flush=True)
    print(f"[report] missing golden points: {len(unsynthesized)}", flush=True)
    print(f"[report] missed golden top-{args.top_k} rows: {len(missed)}", flush=True)
    for row in summaries:
        margin_text = " ".join(
            f"acc@{margin_label(margin)}={fmt(row.get(f'acc_at_{margin_tag(margin)}_percent'))}%"
            for margin in args.margins
        )
        print(
            f"[{row['pipeline']} {row['period_ns']:g}ns] "
            f"n={row['ranked_points']} area_mape={fmt(row['area_mape_percent'])}% "
            f"delay_mape={fmt(row['delay_mape_percent'])}% "
            f"power_mape={fmt(row['power_mape_percent'])}% "
            f"spearman={fmt(row['spearman'], 3)} "
            f"top{args.top_k}_recall={fmt(row['top_k_recall_percent'])}% "
            f"{margin_text}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
