import argparse
import csv
import json
import math
import pathlib
import random
import re
from dataclasses import dataclass
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt


PPA_KEYS = ("area", "power", "eff_clk_period")
CLOCKS = (1.0, 3.0, 5.0, 8.0, 10.0, 15.0)
PROGRESS_TARGET_CLOCK = 1.0
AGGREGATED_PROGRESS_YLIM = (-0.1, 0.3)


@dataclass
class CandidatePoint:
    key: str
    problem: str
    generation: int | None
    source_metrics: dict[str, float | None] | None = None
    source_score: float | None = None
    source_logged_score: float | None = None
    target_metrics: dict[str, float | None] | None = None
    target_score: float | None = None
    target_clock: float | None = None
    target_by_clock: dict[float, dict[str, Any]] | None = None
    target_syn_points: int = 0


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


def normalize_metrics(metrics: dict[str, Any] | None) -> dict[str, float | None]:
    metrics = metrics or {}
    eff_clk_period = metrics.get("eff_clk_period")
    if eff_clk_period is None:
        eff_clk_period = metrics.get("delay")
    return {
        "area": as_float(metrics.get("area")),
        "power": as_float(metrics.get("power")),
        "eff_clk_period": as_float(eff_clk_period),
        "tns": as_float(metrics.get("tns")),
        "wns": as_float(metrics.get("wns")),
    }


def normalize_syn_metrics(syn: dict[str, Any]) -> dict[str, float | None]:
    return {
        "area": as_float(syn.get("Total Area")),
        "power": as_float(syn.get("Total Power")),
        "eff_clk_period": as_float(syn.get("Delay")),
        "tns": as_float(syn.get("TNS")),
        "wns": as_float(syn.get("WNS")),
        "clock": as_float(syn.get("Clock")),
    }


def load_reference_ppa(benchmark_path: pathlib.Path, problem_name: str) -> dict[str, float | None]:
    ppa_path = benchmark_path / f"{problem_name}_ppa.txt"
    if not ppa_path.is_file():
        return {}

    with open(ppa_path, "r", encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f), None)

    if not row:
        return {}

    return normalize_metrics(row)


def fitness_score(metrics: dict[str, float | None], ref: dict[str, float | None]) -> float | None:
    """PPA fitness used by the evolutionary loop: higher is better."""
    p_gen = as_float(metrics.get("power"))
    a_gen = as_float(metrics.get("area"))
    t_gen = as_float(metrics.get("eff_clk_period"))
    p_ref = as_float(ref.get("power"))
    a_ref = as_float(ref.get("area"))
    t_ref = as_float(ref.get("eff_clk_period"))

    if any(v is None for v in (p_gen, a_gen, t_gen, p_ref, a_ref, t_ref)):
        return None

    p_ref = p_ref or 1.0
    a_ref = a_ref or 1.0
    power_delta = (p_gen - p_ref) / p_ref
    area_delta = (a_gen - a_ref) / a_ref

    if t_ref == 0.0:
        total_delta = (power_delta + area_delta) / 2
    else:
        t_ref = t_ref or 1.0
        t_gen = t_gen or 1.0
        timing_delta = (t_gen - t_ref) / t_ref
        total_delta = (power_delta + area_delta + timing_delta) / 3

    return -total_delta


def metric_improvement(new_value: float | None, ref_value: float | None) -> float | None:
    """Lower-is-better percentage improvement."""
    if new_value is None or ref_value in (None, 0):
        return None
    return ((ref_value - new_value) / abs(ref_value)) * 100.0


def average_improvement(metrics: dict[str, float | None], ref: dict[str, float | None]) -> float | None:
    values: list[float] = []
    for key in PPA_KEYS:
        value = metric_improvement(as_float(metrics.get(key)), as_float(ref.get(key)))
        if value is not None:
            values.append(value)
    return mean(values) if values else None


def report_path_to_candidate_key(
    problem: str,
    report_path: str | None,
    suffixes: tuple[str, ...] = (".proxy.json", "_synthesis_report.ppa"),
) -> str | None:
    if not report_path:
        return None

    path = pathlib.PurePosixPath(str(report_path))
    stem = path.name
    for suffix in suffixes:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    else:
        stem = pathlib.PurePosixPath(stem).stem

    generation = next((part for part in path.parts if re.fullmatch(r"Gen\d+", part)), None)
    if generation is None:
        return None
    return f"{problem}__{generation}__{stem}"


def parse_generation_from_key(candidate_key: str) -> int | None:
    match = re.search(r"__Gen(\d+)__", candidate_key)
    return int(match.group(1)) if match else None


def simulation_log_passed(log_text: str) -> bool:
    """Match the pass criteria used by the Verilog evaluator."""
    mismatches = re.search(r"^Mismatches:\s*(\d+)\b", log_text, re.M)
    return (
        mismatches is not None and int(mismatches.group(1)) == 0
    ) or "===========Your Design Passed===========" in log_text


def candidate_key_from_simulation_log(problem: str, log_path: pathlib.Path) -> str | None:
    generation = log_path.parent.name
    if not re.fullmatch(r"Gen\d+", generation):
        return None
    suffix = "_simulation.log"
    if not log_path.name.endswith(suffix):
        return None
    return f"{problem}__{generation}__{log_path.name[:-len(suffix)]}"


def collect_last_round_passing_candidates(
        problem_dir: pathlib.Path,
        feedback_by_key: dict[str, dict[str, Any]],
        sample_count: int = 3,
    ) -> list[CandidatePoint]:
    generation_dirs = [
        path
        for path in problem_dir.iterdir()
        if path.is_dir() and re.fullmatch(r"Gen\d+", path.name)
    ]
    if not generation_dirs:
        return []

    last_generation_dir = max(
        generation_dirs,
        key=lambda path: int(path.name.removeprefix("Gen")),
    )
    passing: list[CandidatePoint] = []
    for log_path in sorted(last_generation_dir.glob("*_simulation.log")):
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not simulation_log_passed(log_text):
            continue

        key = candidate_key_from_simulation_log(problem_dir.name, log_path)
        if key is None:
            continue

        point = CandidatePoint(
            key=key,
            problem=problem_dir.name,
            generation=parse_generation_from_key(key),
        )
        feedback = feedback_by_key.get(key)
        if feedback:
            attach_target_feedback(point, feedback)
        passing.append(point)

    rng = random.Random(f"{problem_dir.name}:{last_generation_dir.name}:passing")
    rng.shuffle(passing)
    return passing[:sample_count]


def load_feedback(
        feedback_path: pathlib.Path,
        refs: dict[str, dict[str, float | None]],
        allow_problem_from_key: bool = False,
    ) -> dict[str, dict[str, Any]]:
    with open(feedback_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    feedback_by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("candidate_id") or row.get("Name")
        problem = row.get("problem")
        if allow_problem_from_key and not problem and key:
            problem = str(key).split("__", 1)[0]
        if not key or not problem or problem not in refs:
            continue

        scored_points: list[tuple[float, dict[str, float | None]]] = []
        scored_by_clock: dict[float, tuple[float, dict[str, float | None]]] = {}
        for syn in row.get("Syn") or []:
            metrics = normalize_syn_metrics(syn)
            score = fitness_score(metrics, refs[problem])
            if score is None:
                continue
            scored_points.append((score, metrics))
            clock = as_float(metrics.get("clock"))
            if clock is not None and (clock not in scored_by_clock or score > scored_by_clock[clock][0]):
                scored_by_clock[clock] = (score, metrics)

        if not scored_points:
            continue

        best_score, best_metrics = max(scored_points, key=lambda item: item[0])
        feedback_by_key[key] = {
            "problem": problem,
            "metrics": best_metrics,
            "score": best_score,
            "clock": best_metrics.get("clock"),
            "by_clock": {
                clock: {"metrics": metrics, "score": score}
                for clock, (score, metrics) in scored_by_clock.items()
            },
            "syn_points": len(scored_points),
        }

    return feedback_by_key


def attach_target_feedback(point: CandidatePoint, feedback: dict[str, Any]) -> None:
    point.target_metrics = feedback["metrics"]
    point.target_score = feedback["score"]
    point.target_clock = feedback["clock"]
    point.target_by_clock = feedback.get("by_clock")
    point.target_syn_points = feedback["syn_points"]


def target_score_at_clock(point: CandidatePoint, clock: float) -> float | None:
    if not point.target_by_clock:
        return None
    clock_entry = point.target_by_clock.get(clock)
    if not clock_entry:
        return None
    return as_float(clock_entry.get("score"))


def target_average_improvement_at_clock(
    point: CandidatePoint,
    clock: float,
    ref: dict[str, float | None],
) -> float | None:
    if not point.target_by_clock:
        return None
    clock_entry = point.target_by_clock.get(clock)
    if not clock_entry:
        return None
    improvement_pct = average_improvement(clock_entry["metrics"], ref)
    if improvement_pct is None:
        return None
    return improvement_pct / 100.0


def feedback_point(candidate_key: str, feedback: dict[str, Any]) -> CandidatePoint:
    point = CandidatePoint(
        key=candidate_key,
        problem=feedback["problem"],
        generation=parse_generation_from_key(candidate_key),
    )
    attach_target_feedback(point, feedback)
    return point


def collect_candidates(
        problem_dir: pathlib.Path,
        ref: dict[str, float | None],
        feedback_by_key: dict[str, dict[str, Any]],
        report_path_suffixes: tuple[str, ...],
        target_ref: dict[str, float | None] | None = None,
    ) -> tuple[dict[str, CandidatePoint], list[dict[str, float | int | None]]]:
    candidates: dict[str, CandidatePoint] = {}
    progress: list[dict[str, float | int | None]] = []
    generation_log = problem_dir / "generation_log.jsonl"
    if not generation_log.is_file():
        return candidates, progress

    target_ref = target_ref or ref
    best_source_score_so_far: float | None = None
    target_score_for_best_source_so_far: float | None = None
    best_synthesized_target_score_so_far: float | None = None

    with open(generation_log, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            generation = int(entry["generation"])
            source_scores: list[float] = []
            generation_best_source_score: float | None = None
            generation_best_target_score: float | None = None

            for detail in entry.get("population_ppa_details") or []:
                ppa_metrics = detail.get("ppa_metrics") or {}
                key = report_path_to_candidate_key(
                    problem_dir.name,
                    ppa_metrics.get("report_path"),
                    suffixes=report_path_suffixes,
                )
                if key is None:
                    continue

                source_metrics = normalize_metrics(ppa_metrics)
                point = CandidatePoint(
                    key=key,
                    problem=problem_dir.name,
                    generation=generation,
                    source_metrics=source_metrics,
                    source_score=fitness_score(source_metrics, ref),
                    source_logged_score=as_float(detail.get("score")),
                )

                feedback = feedback_by_key.get(key)
                if feedback:
                    attach_target_feedback(point, feedback)

                candidates[key] = point
                if point.source_score is not None:
                    source_scores.append(point.source_score)
                selection_score = point.source_logged_score
                if selection_score is None:
                    selection_score = point.source_score
                if selection_score is not None:
                    if (
                        generation_best_source_score is None
                        or selection_score > generation_best_source_score
                    ):
                        generation_best_source_score = selection_score
                        generation_best_target_score = target_average_improvement_at_clock(
                            point, PROGRESS_TARGET_CLOCK, target_ref
                        )
                target_score = target_average_improvement_at_clock(
                    point, PROGRESS_TARGET_CLOCK, target_ref
                )
                if (
                    target_score is not None
                    and (
                        best_synthesized_target_score_so_far is None
                        or target_score > best_synthesized_target_score_so_far
                    )
                ):
                    best_synthesized_target_score_so_far = target_score

            if (
                generation_best_source_score is not None
                and (
                    best_source_score_so_far is None
                    or generation_best_source_score > best_source_score_so_far
                )
            ):
                best_source_score_so_far = generation_best_source_score
                target_score_for_best_source_so_far = generation_best_target_score

            progress.append(
                {
                    "generation": generation,
                    "source_best_score": max(source_scores) if source_scores else None,
                    "target_best_score": target_score_for_best_source_so_far,
                    "target_topline_score": best_synthesized_target_score_so_far,
                }
            )

    return candidates, progress


def running_best(values: list[float | None]) -> list[float | None]:
    best: float | None = None
    result: list[float | None] = []
    for value in values:
        if value is not None:
            best = value if best is None else max(best, value)
        result.append(best)
    return result


def fmt_score(score: float | None) -> str:
    return "N/A" if score is None else f"{score * 100:+.2f}%"


def fmt_metric(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}g}"


def fmt_metrics(metrics: dict[str, float | None] | None) -> str:
    if not metrics:
        return "N/A"
    return (
        f"A={fmt_metric(metrics.get('area'))}, "
        f"P={fmt_metric(metrics.get('power'))}, "
        f"T={fmt_metric(metrics.get('eff_clk_period'))}"
    )


def new_optimization_summary() -> dict[str, Any]:
    return {"known": 0, "improved": 0, "area": [], "power": [], "period": [], "overall": []}


def add_optimization_summary(
    summary: dict[str, Any],
    metrics: dict[str, float | None] | None,
    ref: dict[str, float | None],
) -> None:
    if not metrics:
        return

    metric_values = {
        "area": metric_improvement(as_float(metrics.get("area")), as_float(ref.get("area"))),
        "power": metric_improvement(as_float(metrics.get("power")), as_float(ref.get("power"))),
        "period": metric_improvement(
            as_float(metrics.get("eff_clk_period")),
            as_float(ref.get("eff_clk_period")),
        ),
    }
    valid_values = [value for value in metric_values.values() if value is not None]
    if not valid_values:
        return

    overall = mean(valid_values)
    summary["known"] += 1
    summary["overall"].append(overall)
    if overall > 0:
        summary["improved"] += 1

    for key, value in metric_values.items():
        if value is not None:
            summary[key].append(value)


def fmt_percent_list(values: list[float]) -> str:
    return "N/A" if not values else f"{mean(values):+.2f}%"


def fmt_improved(summary: dict[str, Any]) -> str:
    known = summary["known"]
    if known == 0:
        return "N/A"
    rate = summary["improved"] / known * 100.0
    return f"{summary['improved']} / {known} ({rate:.1f}%)"


def best_clock_entry(candidates: list[CandidatePoint], clock: float) -> dict[str, Any] | None:
    return max(
        (
            (c.target_by_clock or {}).get(clock) | {"key": c.key}
            for c in candidates
            if (c.target_by_clock or {}).get(clock)
        ),
        key=lambda item: item["score"],
        default=None,
    )


def plot_progress(
        output_path: pathlib.Path,
        per_problem_progress: dict[str, list[dict[str, float | int | None]]],
        source_label: str,
        target_label: str,
        title: str,
        ylabel: str,
    ) -> None:
    problems = sorted(per_problem_progress)
    if not problems:
        return

    cols = 3
    rows = math.ceil(len(problems) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.4, rows * 3.45), squeeze=False)

    source_color = "#1f77b4"
    target_color = "#d9483b"

    for ax, problem in zip(axes.flat, problems):
        progress = sorted(per_problem_progress[problem], key=lambda item: int(item["generation"]))
        generations = [int(item["generation"]) for item in progress]
        source = [as_float(item["source_best_score"]) for item in progress]
        target = [as_float(item["target_best_score"]) for item in progress]

        ax.plot(
            generations,
            running_best(source),
            color=source_color,
            linewidth=2.0,
            label=f"{source_label} best-so-far",
        )
        ax.plot(
            generations,
            running_best(target),
            color=target_color,
            linewidth=2.0,
            label=f"{target_label} best-so-far",
        )
        ax.scatter(
            generations,
            source,
            color=source_color,
            s=16,
            alpha=0.5,
            label=f"{source_label} per-round best",
        )
        ax.scatter(
            generations,
            target,
            color=target_color,
            s=16,
            alpha=0.5,
            label=f"{target_label} per-round best",
        )
        ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.5)
        ax.set_title(problem, fontsize=10)
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.5, alpha=0.3)

    for ax in axes.flat[len(problems) :]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(title, y=0.992)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=2,
        frameon=True,
        fontsize=11,
        borderaxespad=0.6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_aggregated_target_progress(
        output_path: pathlib.Path,
        per_problem_progress: dict[str, list[dict[str, float | int | None]]],
        target_label: str,
        topline_label: str,
        title: str,
        ylabel: str,
    ) -> None:
    problem_curves: dict[str, dict[int, float | None]] = {}
    topline_curves: dict[str, dict[int, float | None]] = {}
    generations: set[int] = set()

    for problem, progress_items in sorted(per_problem_progress.items()):
        progress = sorted(progress_items, key=lambda item: int(item["generation"]))
        problem_generations = [int(item["generation"]) for item in progress]
        target_scores = [as_float(item["target_best_score"]) for item in progress]
        topline_scores = [as_float(item.get("target_topline_score")) for item in progress]
        problem_curves[problem] = dict(zip(problem_generations, target_scores))
        topline_curves[problem] = dict(zip(problem_generations, topline_scores))
        generations.update(problem_generations)

    if not generations:
        return

    x_values = sorted(generations)
    mean_values: list[float | None] = []
    for generation in x_values:
        values = [
            score_by_generation[generation] * 100.0
            for score_by_generation in problem_curves.values()
            if score_by_generation.get(generation) is not None
        ]
        mean_values.append(mean(values) if values else None)
    topline_mean_values: list[float | None] = []
    for generation in x_values:
        values = [
            score_by_generation[generation] * 100.0
            for score_by_generation in topline_curves.values()
            if score_by_generation.get(generation) is not None
        ]
        topline_mean_values.append(mean(values) if values else None)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfb")
    background_color = "#e57368"
    mean_color = "#17e3ba"
    topline_color = "#3E87E7CA"

    for problem, score_by_generation in problem_curves.items():
        y_values = [
            score_by_generation.get(generation) * 100.0
            if score_by_generation.get(generation) is not None
            else None
            for generation in x_values
        ]
        ax.plot(
            x_values,
            y_values,
            color=background_color,
            linewidth=1.3,
            alpha=0.20,
            label=f"{target_label} per-problem" if problem == next(iter(problem_curves)) else None,
        )

    ax.plot(
        x_values,
        mean_values,
        color=mean_color,
        linewidth=2.4,
        marker="o",
        markersize=2.5,
        label=f"Mean of {target_label}",
    )
    ax.plot(
        x_values,
        topline_mean_values,
        color=topline_color,
        linewidth=2.0,
        linestyle=":",
        label=f"{topline_label}",
    )
    ax.axhline(0, color="#7a7a7a", linewidth=1.0, alpha=0.65)
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylim(-30, 30)
    ax.set_xticks(x_values)
    ax.set_xlim(left=0)
    ax.grid(True, color="#d2d2d2", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    legend = ax.legend(loc="lower right", frameon=True, fontsize=11)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#cfcfcf")
    legend.get_frame().set_alpha(0.92)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyze_proxy_real(
        experiment_path: pathlib.Path,
        reference_benchmark_path: pathlib.Path,
        real_feedback_path: pathlib.Path,
        output_prefix: str,
    ) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    summary_files = sorted(experiment_path.glob("*/*_summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"No summary files found under {experiment_path}")

    refs: dict[str, dict[str, float | None]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for summary_path in summary_files:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        problem = summary["problem_name"]
        summaries[problem] = summary
        refs[problem] = load_reference_ppa(reference_benchmark_path, problem)

    real_by_key = load_feedback(real_feedback_path, refs)

    rows: list[dict[str, Any]] = []
    per_problem_progress: dict[str, list[dict[str, float | int | None]]] = {}
    optimization_summary = {
        "imagined_selected": new_optimization_summary(),
        "real_selected": new_optimization_summary(),
        "real_top3": new_optimization_summary(),
        "real_best": new_optimization_summary(),
    }
    clock_optimization_summary = {
        clock: {
            "real_selected": new_optimization_summary(),
            "real_top3": new_optimization_summary(),
            "real_best": new_optimization_summary(),
        }
        for clock in CLOCKS
    }

    for problem, summary in sorted(summaries.items()):
        problem_dir = experiment_path / problem
        ref = refs[problem]
        candidates, progress = collect_candidates(
            problem_dir,
            ref,
            real_by_key,
            report_path_suffixes=(".proxy.json",),
        )
        per_problem_progress[problem] = progress

        final_ppa = summary.get("final_population_ppa") or {}
        final_metrics = normalize_metrics(final_ppa.get("best_metrics"))
        selected_key = report_path_to_candidate_key(
            problem,
            (final_ppa.get("best_metrics") or {}).get("report_path"),
            suffixes=(".proxy.json",),
        )
        selected = candidates.get(selected_key or "")
        if selected is None and selected_key:
            selected = CandidatePoint(
                key=selected_key,
                problem=problem,
                generation=parse_generation_from_key(selected_key),
                source_metrics=final_metrics,
                source_logged_score=as_float(final_ppa.get("best_score")),
                source_score=fitness_score(final_metrics, ref),
            )
            real = real_by_key.get(selected_key)
            if real:
                attach_target_feedback(selected, real)

        last_round_passing_candidates = collect_last_round_passing_candidates(
            problem_dir,
            real_by_key,
            sample_count=3,
        )
        selected_source = "recorded_best"
        if selected is None and last_round_passing_candidates:
            selected = last_round_passing_candidates[0]
            selected_source = "last_round_random_pass"

        method_candidates_by_key = dict(candidates)
        if selected:
            method_candidates_by_key[selected.key] = selected
        top3_proxy_candidates = sorted(
            (c for c in method_candidates_by_key.values() if c.source_score is not None),
            key=lambda c: c.source_score,
            reverse=True,
        )[:3]
        top3_source = "proxy_score"
        if not top3_proxy_candidates and last_round_passing_candidates:
            top3_proxy_candidates = last_round_passing_candidates
            top3_source = "last_round_random_pass"
        real_top3 = max(
            (c for c in top3_proxy_candidates if c.target_score is not None),
            key=lambda c: c.target_score,
            default=None,
        )

        real_candidates_by_key = {
            c.key: c for c in candidates.values() if c.target_score is not None
        }
        for key, real in real_by_key.items():
            if real["problem"] == problem and key not in real_candidates_by_key:
                real_candidates_by_key[key] = feedback_point(key, real)
        real_candidates = list(real_candidates_by_key.values())
        real_best = max(real_candidates, key=lambda c: c.target_score) if real_candidates else None
        proxy_best_under_resynth = max(
            (c for c in candidates.values() if c.source_score is not None),
            key=lambda c: c.source_score,
            default=None,
        )

        selected_real_score = selected.target_score if selected else None
        real_best_score = real_best.target_score if real_best else None
        miss_gap = (
            real_best_score - selected_real_score
            if real_best_score is not None and selected_real_score is not None
            else None
        )
        miss_status = "unknown" if miss_gap is None else "yes" if miss_gap > 1e-9 else "no"

        if selected:
            add_optimization_summary(
                optimization_summary["imagined_selected"], selected.source_metrics, ref
            )
            add_optimization_summary(
                optimization_summary["real_selected"], selected.target_metrics, ref
            )
        if real_top3:
            add_optimization_summary(
                optimization_summary["real_top3"], real_top3.target_metrics, ref
            )
        if real_best:
            add_optimization_summary(
                optimization_summary["real_best"], real_best.target_metrics, ref
            )

        row = {
            "problem": problem,
            "reference": fmt_metrics(ref),
            "selected_key": selected.key if selected else selected_key,
            "selected_source": selected_source if selected else "missing",
            "selected_generation": selected.generation if selected else None,
            "imagined_original_score_pct": fmt_score(
                selected.source_logged_score if selected else as_float(final_ppa.get("best_score"))
            ),
            "imagined_resynth_score_pct": fmt_score(
                selected.source_score if selected else None
            ),
            "imagined_metrics": fmt_metrics(selected.source_metrics if selected else final_metrics),
            "real_selected_score_pct": fmt_score(selected_real_score),
            "real_selected_metrics": fmt_metrics(selected.target_metrics if selected else None),
            "real_selected_clock": fmt_metric(selected.target_clock if selected else None),
            "real_top3_key": real_top3.key if real_top3 else "N/A",
            "real_top3_generation": real_top3.generation if real_top3 else None,
            "real_top3_score_pct": fmt_score(real_top3.target_score if real_top3 else None),
            "real_top3_metrics": fmt_metrics(real_top3.target_metrics if real_top3 else None),
            "real_top3_clock": fmt_metric(real_top3.target_clock if real_top3 else None),
            "top3_proxy_keys": ";".join(c.key for c in top3_proxy_candidates),
            "top3_source": top3_source,
            "real_best_key": real_best.key if real_best else "N/A",
            "real_best_generation": real_best.generation if real_best else None,
            "real_best_score_pct": fmt_score(real_best_score),
            "real_best_metrics": fmt_metrics(real_best.target_metrics if real_best else None),
            "real_best_clock": fmt_metric(real_best.target_clock if real_best else None),
            "missed_better_real_solution": miss_status,
            "miss_gap_pct_points": "N/A" if miss_gap is None else f"{miss_gap * 100:+.2f}",
            "real_synthesized_candidates": len(real_candidates),
            "proxy_best_resynth_key": proxy_best_under_resynth.key if proxy_best_under_resynth else "N/A",
            "proxy_best_resynth_score_pct": fmt_score(
                proxy_best_under_resynth.source_score if proxy_best_under_resynth else None
            ),
        }

        for clock in CLOCKS:
            selected_clock = (selected.target_by_clock or {}).get(clock) if selected else None
            top3_clock = best_clock_entry(top3_proxy_candidates, clock)
            best_clock = best_clock_entry(real_candidates, clock)
            clock_label = f"clk_{clock:g}".replace(".", "p")
            if selected_clock:
                add_optimization_summary(
                    clock_optimization_summary[clock]["real_selected"],
                    selected_clock["metrics"],
                    ref,
                )
            if top3_clock:
                add_optimization_summary(
                    clock_optimization_summary[clock]["real_top3"],
                    top3_clock["metrics"],
                    ref,
                )
            if best_clock:
                add_optimization_summary(
                    clock_optimization_summary[clock]["real_best"],
                    best_clock["metrics"],
                    ref,
                )
            row[f"real_selected_{clock_label}_score_pct"] = fmt_score(
                selected_clock["score"] if selected_clock else None
            )
            row[f"real_top3_{clock_label}_score_pct"] = fmt_score(
                top3_clock["score"] if top3_clock else None
            )
            row[f"real_top3_{clock_label}_key"] = top3_clock["key"] if top3_clock else "N/A"
            row[f"real_best_{clock_label}_score_pct"] = fmt_score(
                best_clock["score"] if best_clock else None
            )
            row[f"real_best_{clock_label}_key"] = best_clock["key"] if best_clock else "N/A"

        rows.append(row)

    if not rows:
        raise RuntimeError("No overlapping problems found between experiment and real feedback.")

    output_md = experiment_path / f"{output_prefix}.md"
    output_csv = experiment_path / f"{output_prefix}.csv"
    output_png = experiment_path / f"{output_prefix}_progress.png"
    output_aggregated_png = experiment_path / f"{output_prefix}_progress_aggregated.png"

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_progress(
        output_png,
        per_problem_progress,
        source_label="Imagined",
        target_label="Real",
        title=f"Proxy prediction vs real synthesis progress (clock={PROGRESS_TARGET_CLOCK:.1f}ns)",
        ylabel="Fitness score",
    )
    plot_aggregated_target_progress(
        output_aggregated_png,
        per_problem_progress,
        target_label="PPA Model Recorded-Best",
        topline_label="DC Best Synthesized",
        title=f"PPA Model Recorded-Best Progress (clock={PROGRESS_TARGET_CLOCK:.1f}ns)",
        ylabel="Average PPA improvement (%)",
    )

    def numeric(row: dict[str, Any], key: str) -> float | None:
        value = row[key]
        if value == "N/A":
            return None
        return float(value.rstrip("%"))

    real_selected_scores = [numeric(row, "real_selected_score_pct") for row in rows]
    real_top3_scores = [numeric(row, "real_top3_score_pct") for row in rows]
    real_best_scores = [numeric(row, "real_best_score_pct") for row in rows]
    imagined_scores = [numeric(row, "imagined_resynth_score_pct") for row in rows]
    missed_count = sum(1 for row in rows if row["missed_better_real_solution"] == "yes")
    known_miss_rows = [
        row for row in rows if row["missed_better_real_solution"] in {"yes", "no"}
    ]
    unknown_count = len(rows) - len(known_miss_rows)

    def avg(values: list[float | None]) -> str:
        valid = [value for value in values if value is not None]
        return "N/A" if not valid else f"{mean(valid):+.2f}%"

    md_lines = [
        f"# Proxy-vs-Real Evolutionary Report: {experiment_path.name}",
        "",
        "This report uses the corrected reference PPA files from "
        f"`{reference_benchmark_path}` and real synthesis feedback from `{real_feedback_path}`.",
        "Fitness is recomputed with the PPA formula used by `src/revolution/algorithm.py`: "
        "higher is better, and positive values mean improvement over the reference.",
        "When a run has no proxy-held PPA samples, the comparison set is three randomly sampled "
        "final-generation candidates whose simulation logs report zero mismatches.",
        "",
        "## Overall",
        f"- Problems analyzed: {len(rows)}",
        f"- Average imagined selected improvement: {avg(imagined_scores)}",
        f"- Average real improvement of recorded best sample: {avg(real_selected_scores)}",
        f"- Average real improvement if holding top-3 proxy samples: {avg(real_top3_scores)}",
        f"- Average real improvement of best synthesized sample: {avg(real_best_scores)}",
        "- Tasks where the PPA estimator missed a better real synthesized solution: "
        f"{missed_count}/{len(known_miss_rows)} known comparisons",
        f"- Tasks with missing real synthesis for the recorded best sample: {unknown_count}/{len(rows)}",
        f"- Progress figure: `{output_png.name}`",
        f"- Aggregated progress figure: `{output_aggregated_png.name}`",
        f"- Machine-readable details: `{output_csv.name}`",
        "",
        "## PPA Optimization Summary",
        "| PPA Metric | Imagined Recorded Best | Real Recorded Best | Real Top-3 Held | Real Best Synthesized |",
        "|:---|:---:|:---:|:---:|:---:|",
        "| **Problems Improved** | "
        f"{fmt_improved(optimization_summary['imagined_selected'])} | "
        f"{fmt_improved(optimization_summary['real_selected'])} | "
        f"{fmt_improved(optimization_summary['real_top3'])} | "
        f"{fmt_improved(optimization_summary['real_best'])} |",
        "| **Area** | "
        f"{fmt_percent_list(optimization_summary['imagined_selected']['area'])} | "
        f"{fmt_percent_list(optimization_summary['real_selected']['area'])} | "
        f"{fmt_percent_list(optimization_summary['real_top3']['area'])} | "
        f"{fmt_percent_list(optimization_summary['real_best']['area'])} |",
        "| **Power** | "
        f"{fmt_percent_list(optimization_summary['imagined_selected']['power'])} | "
        f"{fmt_percent_list(optimization_summary['real_selected']['power'])} | "
        f"{fmt_percent_list(optimization_summary['real_top3']['power'])} | "
        f"{fmt_percent_list(optimization_summary['real_best']['power'])} |",
        "| **Performance (Period)** | "
        f"{fmt_percent_list(optimization_summary['imagined_selected']['period'])} | "
        f"{fmt_percent_list(optimization_summary['real_selected']['period'])} | "
        f"{fmt_percent_list(optimization_summary['real_top3']['period'])} | "
        f"{fmt_percent_list(optimization_summary['real_best']['period'])} |",
        "| **Average PPA** | "
        f"{fmt_percent_list(optimization_summary['imagined_selected']['overall'])} | "
        f"{fmt_percent_list(optimization_summary['real_selected']['overall'])} | "
        f"{fmt_percent_list(optimization_summary['real_top3']['overall'])} | "
        f"{fmt_percent_list(optimization_summary['real_best']['overall'])} |",
        "",
        "## Clock-by-Clock PPA Optimization Summary",
    ]

    for clock in CLOCKS:
        clock_summary = clock_optimization_summary[clock]
        md_lines.extend(
            [
                "",
                f"### Clock {clock:g}",
                "| PPA Metric | Real Recorded Best | Real Top-3 Held | Real Best Synthesized |",
                "|:---|:---:|:---:|:---:|",
                "| **Problems Improved** | "
                f"{fmt_improved(clock_summary['real_selected'])} | "
                f"{fmt_improved(clock_summary['real_top3'])} | "
                f"{fmt_improved(clock_summary['real_best'])} |",
                "| **Area** | "
                f"{fmt_percent_list(clock_summary['real_selected']['area'])} | "
                f"{fmt_percent_list(clock_summary['real_top3']['area'])} | "
                f"{fmt_percent_list(clock_summary['real_best']['area'])} |",
                "| **Power** | "
                f"{fmt_percent_list(clock_summary['real_selected']['power'])} | "
                f"{fmt_percent_list(clock_summary['real_top3']['power'])} | "
                f"{fmt_percent_list(clock_summary['real_best']['power'])} |",
                "| **Performance (Period)** | "
                f"{fmt_percent_list(clock_summary['real_selected']['period'])} | "
                f"{fmt_percent_list(clock_summary['real_top3']['period'])} | "
                f"{fmt_percent_list(clock_summary['real_best']['period'])} |",
                "| **Average PPA** | "
                f"{fmt_percent_list(clock_summary['real_selected']['overall'])} | "
                f"{fmt_percent_list(clock_summary['real_top3']['overall'])} | "
                f"{fmt_percent_list(clock_summary['real_best']['overall'])} |",
            ]
        )

    md_lines.extend(
        [
            "",
            "## Problem Details",
            "| Problem | Imagined selected | Real selected | Real top-3 held | Real best synthesized | Missed? | Gap (pct pts) |",
            "|:---|:---|:---|:---|:---|:---:|---:|",
        ]
    )

    for row in rows:
        imagined = (
            f"{row['imagined_resynth_score_pct']}<br>`{row['selected_key']}`<br>"
            f"source={row['selected_source']}<br>"
            f"{row['imagined_metrics']}"
        )
        real_selected = (
            f"{row['real_selected_score_pct']}<br>clk={row['real_selected_clock']}<br>"
            f"{row['real_selected_metrics']}"
        )
        real_top3 = (
            f"{row['real_top3_score_pct']}<br>`{row['real_top3_key']}`<br>"
            f"source={row['top3_source']}<br>"
            f"clk={row['real_top3_clock']}<br>{row['real_top3_metrics']}"
        )
        real_best = (
            f"{row['real_best_score_pct']}<br>`{row['real_best_key']}`<br>"
            f"clk={row['real_best_clock']}<br>{row['real_best_metrics']}"
        )
        md_lines.append(
            "| "
            + " | ".join(
                [
                    row["problem"],
                    imagined,
                    real_selected,
                    real_top3,
                    real_best,
                    row["missed_better_real_solution"],
                    row["miss_gap_pct_points"],
                ]
            )
            + " |"
        )

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        f.write("\n")

    return output_md, output_csv, output_png, output_aggregated_png


def analyze_yosys_dc(
        original_results_path: pathlib.Path,
        yosys_reference_benchmark_path: pathlib.Path,
        dc_reference_benchmark_path: pathlib.Path,
        dc_feedback_path: pathlib.Path,
        output_dir: pathlib.Path,
        output_prefix: str,
    ) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    summary_files = sorted(original_results_path.glob("*/*_summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"No summary files found under {original_results_path}")

    dc_rows = json.load(dc_feedback_path.open("r", encoding="utf-8"))
    target_problems = {
        row.get("problem") or str(row.get("Name", "")).split("__", 1)[0]
        for row in dc_rows
        if row.get("problem") or row.get("Name")
    }

    yosys_refs: dict[str, dict[str, float | None]] = {}
    dc_refs: dict[str, dict[str, float | None]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for summary_path in summary_files:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        problem = summary["problem_name"]
        if target_problems and problem not in target_problems:
            continue
        summaries[problem] = summary
        yosys_refs[problem] = load_reference_ppa(
            yosys_reference_benchmark_path, problem
        ) or normalize_metrics(summary.get("ref_ppa_metric"))
        dc_refs[problem] = load_reference_ppa(dc_reference_benchmark_path, problem)

    missing_dc_refs = [problem for problem, ref in dc_refs.items() if not ref]
    if missing_dc_refs:
        missing = ", ".join(sorted(missing_dc_refs)[:8])
        raise FileNotFoundError(
            f"Missing DC reference PPA files under {dc_reference_benchmark_path}: {missing}"
        )

    dc_by_key = load_feedback(dc_feedback_path, dc_refs, allow_problem_from_key=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    per_problem_progress: dict[str, list[dict[str, float | int | None]]] = {}
    optimization_summary = {
        "yosys_recorded": new_optimization_summary(),
        "dc_recorded": new_optimization_summary(),
        "dc_top3": new_optimization_summary(),
        "dc_best": new_optimization_summary(),
    }
    clock_optimization_summary = {
        clock: {
            "dc_recorded": new_optimization_summary(),
            "dc_top3": new_optimization_summary(),
            "dc_best": new_optimization_summary(),
        }
        for clock in CLOCKS
    }

    for problem, summary in sorted(summaries.items()):
        problem_dir = original_results_path / problem
        yosys_ref = yosys_refs[problem]
        dc_ref = dc_refs[problem]
        candidates, progress = collect_candidates(
            problem_dir,
            yosys_ref,
            dc_by_key,
            report_path_suffixes=(".proxy.json", "_synthesis_report.ppa"),
            target_ref=dc_ref,
        )
        per_problem_progress[problem] = progress

        final_ppa = summary.get("final_population_ppa") or {}
        final_metrics = normalize_metrics(final_ppa.get("best_metrics"))
        recorded_key = report_path_to_candidate_key(
            problem,
            (final_ppa.get("best_metrics") or {}).get("report_path"),
            suffixes=(".proxy.json", "_synthesis_report.ppa"),
        )
        recorded = candidates.get(recorded_key or "")
        if recorded is None and recorded_key:
            recorded = CandidatePoint(
                key=recorded_key,
                problem=problem,
                generation=parse_generation_from_key(recorded_key),
                source_metrics=final_metrics,
                source_logged_score=as_float(final_ppa.get("best_score")),
                source_score=fitness_score(final_metrics, yosys_ref),
            )
            dc = dc_by_key.get(recorded_key)
            if dc:
                attach_target_feedback(recorded, dc)

        method_candidates_by_key = dict(candidates)
        if recorded:
            method_candidates_by_key[recorded.key] = recorded
        top3_yosys_candidates = sorted(
            (c for c in method_candidates_by_key.values() if c.source_score is not None),
            key=lambda c: c.source_score,
            reverse=True,
        )[:3]
        dc_top3 = max(
            (c for c in top3_yosys_candidates if c.target_score is not None),
            key=lambda c: c.target_score,
            default=None,
        )

        dc_candidates_by_key = {
            c.key: c for c in candidates.values() if c.target_score is not None
        }
        for key, dc in dc_by_key.items():
            if dc["problem"] == problem and key not in dc_candidates_by_key:
                dc_candidates_by_key[key] = feedback_point(key, dc)
        dc_candidates = list(dc_candidates_by_key.values())
        dc_best = max(dc_candidates, key=lambda c: c.target_score) if dc_candidates else None

        recorded_dc_score = recorded.target_score if recorded else None
        dc_best_score = dc_best.target_score if dc_best else None
        miss_gap = (
            dc_best_score - recorded_dc_score
            if dc_best_score is not None and recorded_dc_score is not None
            else None
        )
        miss_status = "unknown" if miss_gap is None else "yes" if miss_gap > 1e-9 else "no"

        if recorded:
            add_optimization_summary(
                optimization_summary["yosys_recorded"], recorded.source_metrics, yosys_ref
            )
            add_optimization_summary(
                optimization_summary["dc_recorded"], recorded.target_metrics, dc_ref
            )
        if dc_top3:
            add_optimization_summary(optimization_summary["dc_top3"], dc_top3.target_metrics, dc_ref)
        if dc_best:
            add_optimization_summary(optimization_summary["dc_best"], dc_best.target_metrics, dc_ref)

        row = {
            "problem": problem,
            "yosys_reference": fmt_metrics(yosys_ref),
            "dc_reference": fmt_metrics(dc_ref),
            "recorded_key": recorded.key if recorded else recorded_key,
            "recorded_generation": recorded.generation if recorded else None,
            "yosys_recorded_score_pct": fmt_score(recorded.source_score if recorded else None),
            "yosys_recorded_metrics": fmt_metrics(recorded.source_metrics if recorded else final_metrics),
            "dc_recorded_score_pct": fmt_score(recorded_dc_score),
            "dc_recorded_metrics": fmt_metrics(recorded.target_metrics if recorded else None),
            "dc_recorded_clock": fmt_metric(recorded.target_clock if recorded else None),
            "dc_top3_key": dc_top3.key if dc_top3 else "N/A",
            "dc_top3_generation": dc_top3.generation if dc_top3 else None,
            "dc_top3_score_pct": fmt_score(dc_top3.target_score if dc_top3 else None),
            "dc_top3_metrics": fmt_metrics(dc_top3.target_metrics if dc_top3 else None),
            "dc_top3_clock": fmt_metric(dc_top3.target_clock if dc_top3 else None),
            "top3_yosys_keys": ";".join(c.key for c in top3_yosys_candidates),
            "dc_best_key": dc_best.key if dc_best else "N/A",
            "dc_best_generation": dc_best.generation if dc_best else None,
            "dc_best_score_pct": fmt_score(dc_best_score),
            "dc_best_metrics": fmt_metrics(dc_best.target_metrics if dc_best else None),
            "dc_best_clock": fmt_metric(dc_best.target_clock if dc_best else None),
            "missed_better_dc_solution": miss_status,
            "miss_gap_pct_points": "N/A" if miss_gap is None else f"{miss_gap * 100:+.2f}",
            "dc_synthesized_candidates": len(dc_candidates),
        }

        for clock in CLOCKS:
            recorded_clock = (recorded.target_by_clock or {}).get(clock) if recorded else None
            top3_clock = best_clock_entry(top3_yosys_candidates, clock)
            best_clock = best_clock_entry(dc_candidates, clock)
            clock_label = f"clk_{clock:g}".replace(".", "p")
            if recorded_clock:
                add_optimization_summary(
                    clock_optimization_summary[clock]["dc_recorded"],
                    recorded_clock["metrics"],
                    dc_ref,
                )
            if top3_clock:
                add_optimization_summary(
                    clock_optimization_summary[clock]["dc_top3"],
                    top3_clock["metrics"],
                    dc_ref,
                )
            if best_clock:
                add_optimization_summary(
                    clock_optimization_summary[clock]["dc_best"],
                    best_clock["metrics"],
                    dc_ref,
                )
            row[f"dc_recorded_{clock_label}_score_pct"] = fmt_score(
                recorded_clock["score"] if recorded_clock else None
            )
            row[f"dc_top3_{clock_label}_score_pct"] = fmt_score(
                top3_clock["score"] if top3_clock else None
            )
            row[f"dc_top3_{clock_label}_key"] = top3_clock["key"] if top3_clock else "N/A"
            row[f"dc_best_{clock_label}_score_pct"] = fmt_score(
                best_clock["score"] if best_clock else None
            )
            row[f"dc_best_{clock_label}_key"] = best_clock["key"] if best_clock else "N/A"

        rows.append(row)

    if not rows:
        raise RuntimeError("No overlapping problems found between original results and DC feedback.")

    output_md = output_dir / f"{output_prefix}.md"
    output_csv = output_dir / f"{output_prefix}.csv"
    output_png = output_dir / f"{output_prefix}_progress.png"
    output_aggregated_png = output_dir / f"{output_prefix}_progress_aggregated.png"

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_progress(
        output_png,
        per_problem_progress,
        source_label="Yosys/OpenROAD",
        target_label="DC/PT",
        title=f"Yosys/OpenROAD vs Design Compiler/PrimeTime progress by generation (clock={PROGRESS_TARGET_CLOCK:.1f}ns)",
        ylabel="Fitness score",
    )
    plot_aggregated_target_progress(
        output_aggregated_png,
        per_problem_progress,
        target_label="Yosys Recorded-Best",
        topline_label="DC Best Synthesized",
        title=f"Yosys Recorded-Best Progress (clock={PROGRESS_TARGET_CLOCK:.1f}ns)",
        ylabel="Average PPA improvement (%)",
    )

    def numeric(row: dict[str, Any], key: str) -> float | None:
        value = row[key]
        if value == "N/A":
            return None
        return float(value.rstrip("%"))

    yosys_scores = [numeric(row, "yosys_recorded_score_pct") for row in rows]
    dc_recorded_scores = [numeric(row, "dc_recorded_score_pct") for row in rows]
    dc_top3_scores = [numeric(row, "dc_top3_score_pct") for row in rows]
    dc_best_scores = [numeric(row, "dc_best_score_pct") for row in rows]
    missed_count = sum(1 for row in rows if row["missed_better_dc_solution"] == "yes")
    known_miss_rows = [row for row in rows if row["missed_better_dc_solution"] in {"yes", "no"}]
    unknown_count = len(rows) - len(known_miss_rows)

    def avg(values: list[float | None]) -> str:
        valid = [value for value in values if value is not None]
        return "N/A" if not valid else f"{mean(valid):+.2f}%"

    md_lines = [
        f"# Yosys/OpenROAD vs DC/PT Report: {original_results_path.name}",
        "",
        f"Original DeepSeek Yosys/OpenROAD results: `{original_results_path}`.",
        f"Design Compiler/PrimeTime feedback: `{dc_feedback_path}`.",
        f"Yosys/OpenROAD recorded-best improvements use `{yosys_reference_benchmark_path}` as baseline.",
        f"DC/PT recorded-best and best-synthesized improvements use `{dc_reference_benchmark_path}` as baseline.",
        "Fitness is recomputed with the same lower-is-better PPA formula used by the evolutionary loop. Higher is better; positive values mean improvement over each toolchain's own reference.",
        "Because the Yosys/OpenROAD and DC/PT columns use different baselines, their percentages should not be interpreted as directly comparable across toolchains.",
        "",
        "## Overall",
        f"- Problems analyzed: {len(rows)}",
        f"- Average Yosys/OpenROAD recorded-best improvement: {avg(yosys_scores)}",
        f"- Average DC/PT improvement of Yosys/OpenROAD recorded best: {avg(dc_recorded_scores)}",
        f"- Average DC/PT improvement if holding top-3 Yosys/OpenROAD samples: {avg(dc_top3_scores)}",
        f"- Average DC/PT improvement of best synthesized sample: {avg(dc_best_scores)}",
        f"- Tasks where Yosys/OpenROAD recorded best missed a better DC/PT synthesized solution: {missed_count}/{len(known_miss_rows)} known comparisons",
        f"- Tasks with missing DC/PT synthesis for the Yosys/OpenROAD recorded best: {unknown_count}/{len(rows)}",
        f"- Progress figure: `{output_png.name}`",
        f"- Aggregated progress figure: `{output_aggregated_png.name}`",
        f"- Machine-readable details: `{output_csv.name}`",
        "",
        "## PPA Optimization Summary",
        "| PPA Metric | Yosys Recorded Best | DC Recorded Best | DC Top-3 Held | DC Best Synthesized |",
        "|:---|:---:|:---:|:---:|:---:|",
        "| **Problems Improved** | "
        f"{fmt_improved(optimization_summary['yosys_recorded'])} | "
        f"{fmt_improved(optimization_summary['dc_recorded'])} | "
        f"{fmt_improved(optimization_summary['dc_top3'])} | "
        f"{fmt_improved(optimization_summary['dc_best'])} |",
        "| **Area** | "
        f"{fmt_percent_list(optimization_summary['yosys_recorded']['area'])} | "
        f"{fmt_percent_list(optimization_summary['dc_recorded']['area'])} | "
        f"{fmt_percent_list(optimization_summary['dc_top3']['area'])} | "
        f"{fmt_percent_list(optimization_summary['dc_best']['area'])} |",
        "| **Power** | "
        f"{fmt_percent_list(optimization_summary['yosys_recorded']['power'])} | "
        f"{fmt_percent_list(optimization_summary['dc_recorded']['power'])} | "
        f"{fmt_percent_list(optimization_summary['dc_top3']['power'])} | "
        f"{fmt_percent_list(optimization_summary['dc_best']['power'])} |",
        "| **Performance (Period)** | "
        f"{fmt_percent_list(optimization_summary['yosys_recorded']['period'])} | "
        f"{fmt_percent_list(optimization_summary['dc_recorded']['period'])} | "
        f"{fmt_percent_list(optimization_summary['dc_top3']['period'])} | "
        f"{fmt_percent_list(optimization_summary['dc_best']['period'])} |",
        "| **Average PPA** | "
        f"{fmt_percent_list(optimization_summary['yosys_recorded']['overall'])} | "
        f"{fmt_percent_list(optimization_summary['dc_recorded']['overall'])} | "
        f"{fmt_percent_list(optimization_summary['dc_top3']['overall'])} | "
        f"{fmt_percent_list(optimization_summary['dc_best']['overall'])} |",
        "",
        "## Clock-by-Clock PPA Optimization Summary",
    ]

    for clock in CLOCKS:
        clock_summary = clock_optimization_summary[clock]
        md_lines.extend(
            [
                "",
                f"### Clock {clock:g}",
                "| PPA Metric | DC Recorded Best | DC Top-3 Held | DC Best Synthesized |",
                "|:---|:---:|:---:|:---:|",
                "| **Problems Improved** | "
                f"{fmt_improved(clock_summary['dc_recorded'])} | "
                f"{fmt_improved(clock_summary['dc_top3'])} | "
                f"{fmt_improved(clock_summary['dc_best'])} |",
                "| **Area** | "
                f"{fmt_percent_list(clock_summary['dc_recorded']['area'])} | "
                f"{fmt_percent_list(clock_summary['dc_top3']['area'])} | "
                f"{fmt_percent_list(clock_summary['dc_best']['area'])} |",
                "| **Power** | "
                f"{fmt_percent_list(clock_summary['dc_recorded']['power'])} | "
                f"{fmt_percent_list(clock_summary['dc_top3']['power'])} | "
                f"{fmt_percent_list(clock_summary['dc_best']['power'])} |",
                "| **Performance (Period)** | "
                f"{fmt_percent_list(clock_summary['dc_recorded']['period'])} | "
                f"{fmt_percent_list(clock_summary['dc_top3']['period'])} | "
                f"{fmt_percent_list(clock_summary['dc_best']['period'])} |",
                "| **Average PPA** | "
                f"{fmt_percent_list(clock_summary['dc_recorded']['overall'])} | "
                f"{fmt_percent_list(clock_summary['dc_top3']['overall'])} | "
                f"{fmt_percent_list(clock_summary['dc_best']['overall'])} |",
            ]
        )

    md_lines.extend(
        [
            "",
            "## Problem Details",
            "| Problem | Yosys recorded best | DC recorded best | DC top-3 held | DC best synthesized | Missed? | Gap (pct pts) |",
            "|:---|:---|:---|:---|:---|:---:|---:|",
        ]
    )

    for row in rows:
        yosys_recorded = (
            f"{row['yosys_recorded_score_pct']}<br>`{row['recorded_key']}`<br>"
            f"{row['yosys_recorded_metrics']}"
        )
        dc_recorded = (
            f"{row['dc_recorded_score_pct']}<br>clk={row['dc_recorded_clock']}<br>"
            f"{row['dc_recorded_metrics']}"
        )
        dc_top3 = (
            f"{row['dc_top3_score_pct']}<br>`{row['dc_top3_key']}`<br>"
            f"clk={row['dc_top3_clock']}<br>{row['dc_top3_metrics']}"
        )
        dc_best = (
            f"{row['dc_best_score_pct']}<br>`{row['dc_best_key']}`<br>"
            f"clk={row['dc_best_clock']}<br>{row['dc_best_metrics']}"
        )
        md_lines.append(
            "| "
            + " | ".join(
                [
                    row["problem"],
                    yosys_recorded,
                    dc_recorded,
                    dc_top3,
                    dc_best,
                    row["missed_better_dc_solution"],
                    row["miss_gap_pct_points"],
                ]
            )
            + " |"
        )

    with output_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        f.write("\n")

    return output_md, output_csv, output_png, output_aggregated_png


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate combined evolutionary PPA comparison reports."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    proxy_parser = subparsers.add_parser(
        "proxy-real",
        aliases=["proxy_real"],
        help="Compare proxy-imagined PPA progress with real synthesis feedback.",
    )
    proxy_parser.add_argument(
        "--experiment_path",
        type=pathlib.Path,
        default=pathlib.Path("src/exp/gpt-5.4-nano/VerilogEval-Spec-to-RTL-DC-opt"),
    )
    proxy_parser.add_argument(
        "--reference_benchmark_path",
        type=pathlib.Path,
        default=pathlib.Path("VerilogEval-Spec-to-RTL"),
    )
    proxy_parser.add_argument(
        "--real_feedback_path",
        type=pathlib.Path,
        default=pathlib.Path("data/synth/test_ppa_feedback.json"),
    )
    proxy_parser.add_argument(
        "--output_prefix",
        default="proxy_real_evolutionary_report",
        help="Output basename written under the experiment path.",
    )

    dc_parser = subparsers.add_parser(
        "yosys-dc",
        aliases=["yosys_dc"],
        help="Compare Yosys/OpenROAD results with Design Compiler/PrimeTime feedback.",
    )
    dc_parser.add_argument(
        "--original_results_path",
        type=pathlib.Path,
        default=pathlib.Path("exp/deepseek_clean_results/VerilogEval-Spec-to-RTL"),
    )
    dc_parser.add_argument(
        "--yosys_reference_benchmark_path",
        type=pathlib.Path,
        default=pathlib.Path("data/bench/VerilogEval-Spec-to-RTL"),
    )
    dc_parser.add_argument(
        "--dc_reference_benchmark_path",
        type=pathlib.Path,
        default=pathlib.Path("data/bench/VerilogEval-Spec-to-RTL-DC-opt_resynth"),
    )
    dc_parser.add_argument(
        "--dc_feedback_path",
        type=pathlib.Path,
        default=pathlib.Path("data/synth/test_deepseek_ref.json"),
    )
    dc_parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("src/outputs/revolution_reports"),
    )
    dc_parser.add_argument(
        "--output_prefix",
        default="deepseek_yosys_dc_real_report",
        help="Output basename written under the output directory.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode in {"proxy-real", "proxy_real"}:
        output_md, output_csv, output_png, output_aggregated_png = analyze_proxy_real(
            args.experiment_path,
            args.reference_benchmark_path,
            args.real_feedback_path,
            args.output_prefix,
        )
    elif args.mode in {"yosys-dc", "yosys_dc"}:
        output_md, output_csv, output_png, output_aggregated_png = analyze_yosys_dc(
            args.original_results_path,
            args.yosys_reference_benchmark_path,
            args.dc_reference_benchmark_path,
            args.dc_feedback_path,
            args.output_dir,
            args.output_prefix,
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    print(f"Markdown report: {output_md}")
    print(f"CSV details: {output_csv}")
    print(f"Progress figure: {output_png}")
    print(f"Aggregated progress figure: {output_aggregated_png}")


if __name__ == "__main__":
    main()
