import argparse
import json
import math
import pathlib
import re
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt


METRIC_LABELS = {
    "area": "Area",
    "delay": "Delay",
    "power": "Power",
}

PLOTS = (
    ("area", "delay", "Area-Delay"),
    ("area", "power", "Area-Power"),
    ("delay", "power", "Delay-Power"),
)


@dataclass(frozen=True)
class SynthPoint:
    problem: str
    method: str
    candidate: str
    area: float
    delay: float
    power: float
    clock: float | None


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


def method_name_from_path(path: pathlib.Path) -> str:
    stem = path.stem
    stem = re.sub(r"^test[_-]?", "", stem)
    return stem.replace("_", " ")


def problem_from_row(row: dict[str, Any]) -> str | None:
    problem = row.get("problem")
    if problem:
        return str(problem)

    name = row.get("candidate_id") or row.get("Name")
    if name:
        return str(name).split("__", 1)[0]
    return None


def load_points(path: pathlib.Path, method_name: str | None = None) -> list[SynthPoint]:
    method = method_name or method_name_from_path(path)
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    points: list[SynthPoint] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        problem = problem_from_row(row)
        candidate = row.get("candidate_id") or row.get("Name")
        if not problem or not candidate:
            continue

        for syn in row.get("Syn") or []:
            if not isinstance(syn, dict):
                continue
            area = as_float(syn.get("Total Area"))
            delay = as_float(syn.get("Delay"))
            power = as_float(syn.get("Total Power"))
            if area is None or delay is None or power is None:
                continue
            points.append(
                SynthPoint(
                    problem=problem,
                    method=method,
                    candidate=str(candidate),
                    area=area,
                    delay=delay,
                    power=power,
                    clock=as_float(syn.get("Clock")),
                )
            )

    return points


def pareto_front(points: list[SynthPoint], x_metric: str, y_metric: str) -> list[SynthPoint]:
    """Return nondominated points for a two-objective minimization plot."""
    best_by_xy: dict[tuple[float, float], SynthPoint] = {}
    for point in points:
        key = (getattr(point, x_metric), getattr(point, y_metric))
        best_by_xy.setdefault(key, point)

    ordered = sorted(
        best_by_xy.values(),
        key=lambda point: (getattr(point, x_metric), getattr(point, y_metric)),
    )
    front: list[SynthPoint] = []
    best_y = float("inf")
    for point in ordered:
        y_value = getattr(point, y_metric)
        if y_value < best_y:
            front.append(point)
            best_y = y_value
    return front


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def plot_problem(
        problem: str,
        points_by_method: dict[str, list[SynthPoint]],
        output_dir: pathlib.Path,
    ) -> pathlib.Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    colors = plt.get_cmap("tab10")

    for ax, (x_metric, y_metric, title) in zip(axes, PLOTS):
        for method_index, (method, points) in enumerate(sorted(points_by_method.items())):
            if not points:
                continue

            color = colors(method_index % 10)
            front = pareto_front(points, x_metric, y_metric)
            if front:
                ax.plot(
                    [getattr(point, x_metric) for point in front],
                    [getattr(point, y_metric) for point in front],
                    marker="o",
                    markersize=4,
                    linewidth=2.0,
                    color=color,
                    label=method,
                )

        ax.set_title(title)
        ax.set_xlabel(METRIC_LABELS[x_metric])
        ax.set_ylabel(METRIC_LABELS[y_metric])
        ax.grid(True, linewidth=0.5, alpha=0.35)

    handles, labels = axes[0].get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(4, max(1, len(unique))),
        fontsize=9,
        frameon=True,
    )
    fig.suptitle(f"Pareto Frontiers: {problem}", y=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    output_path = output_dir / f"{safe_filename(problem)}_pareto.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw per-problem synthesis Pareto frontiers from result JSON files."
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        nargs="*",
        default=None,
        help="Result JSON files. Defaults to data/synth/*.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("outputs/pareto"),
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels matching --input order.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_paths = args.input or sorted(pathlib.Path("data/synth").glob("*.json"))
    if not input_paths:
        raise FileNotFoundError("No synthesis JSON files found. Pass files with --input.")
    if args.labels and len(args.labels) != len(input_paths):
        raise ValueError("--labels must have the same count as --input files.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_points: list[SynthPoint] = []
    for index, path in enumerate(input_paths):
        label = args.labels[index] if args.labels else None
        points = load_points(path, label)
        print(f"Loaded {len(points)} synthesis points from {path}")
        all_points.extend(points)

    problems = sorted({point.problem for point in all_points})
    if not problems:
        raise RuntimeError("No valid synthesis points found in the input JSON files.")

    written: list[pathlib.Path] = []
    for problem in problems:
        points_by_method: dict[str, list[SynthPoint]] = {}
        for point in all_points:
            if point.problem == problem:
                points_by_method.setdefault(point.method, []).append(point)
        written.append(plot_problem(problem, points_by_method, args.output_dir))

    print(f"Wrote {len(written)} figures to {args.output_dir}")


if __name__ == "__main__":
    main()
