#!/usr/bin/env python3
"""Draw a raster-friendly generation-time chart from evolve timing logs.

The chart compares mean wall time per generation event for the shared idea and
root stages and for the Yosys/OpenROAD refinement stage.  Totals and samples
are written beside the image so the chart remains easy to audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from html import escape
from pathlib import Path


STAGES = (
    ("idea_generation", "Idea generation"),
    ("codex_root_generation", "Codex root"),
    ("codex_refinement", "Codex refinement"),
)
RUN_COLORS = {
    "Terra medium": "#4e79a7",
    "Luna max": "#e15759",
}
PHASE_COLORS = {
    "idea_generation": "#4e79a7",
    "codex_root_generation": "#f28e2b",
    "codex_refinement": "#e15759",
}


def load_stage_stats(run_root: Path) -> dict[str, dict[str, float | int]]:
    timings_path = run_root / "timings.csv"
    if not timings_path.is_file():
        raise SystemExit(f"timings.csv not found: {timings_path}")

    values: dict[str, list[float]] = {stage: [] for stage, _ in STAGES}
    with timings_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "success":
                continue
            stage = row.get("stage", "")
            if stage not in values:
                continue
            # Refine only the Yosys/OpenROAD pipeline.  Idea/root generation is
            # shared and has no pipeline value in the experiment log.
            if stage == "codex_refinement" and row.get("pipeline") != "yosys_openroad":
                continue
            try:
                value = float(row["wall_seconds"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0:
                values[stage].append(value)

    missing = [stage for stage, _ in STAGES if not values[stage]]
    if missing:
        raise SystemExit(
            f"{run_root}: missing successful generation stages: {', '.join(missing)}"
        )
    return {
        stage: {
            "calls": len(samples),
            "total_seconds": sum(samples),
            "mean_seconds": sum(samples) / len(samples),
        }
        for stage, samples in values.items()
    }


def text(x: float, y: float, value: str, size: int = 16, **attrs: str) -> str:
    attributes = " ".join(
        f'{key.replace("_", "-")}="{escape(str(val))}"'
        for key, val in attrs.items()
    )
    return f'<text x="{x:g}" y="{y:g}" font-size="{size}px" {attributes}>{escape(value)}</text>'


def draw_percent_stacked_svg(
    stats: dict[str, dict[str, dict[str, float | int]]]
) -> str:
    """Draw a horizontal 100%-stacked chart matching the evolve phase chart."""
    width = 1280
    height = max(520, 330 + 115 * len(stats))
    left, right, top, bottom = 250, 45, 95, 125
    plot_width = width - left - right
    bar_height, gap = 72, 35
    plot_bottom = top + len(stats) * bar_height + (len(stats) - 1) * gap
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(
            width / 2,
            42,
            "Generation-time phase distribution",
            25,
            text_anchor="middle",
            font_family="Arial, sans-serif",
            font_weight="bold",
        ),
        text(
            left,
            70,
            "Percentage of each run's accounted generation time",
            15,
            font_family="Arial, sans-serif",
            fill="#555",
        ),
    ]

    for tick in (0, 25, 50, 75, 100):
        x = left + plot_width * tick / 100.0
        parts.append(
            f'<line x1="{x:g}" y1="{top - 8:g}" x2="{x:g}" y2="{plot_bottom:g}" stroke="#d9d9d9" stroke-width="1"/>'
        )
        parts.append(
            text(
                x,
                plot_bottom + 25,
                f"{tick}%",
                13,
                text_anchor="middle",
                font_family="Arial, sans-serif",
                fill="#555",
            )
        )

    for index, (run, stages) in enumerate(stats.items()):
        y = top + index * (bar_height + gap)
        parts.append(
            text(
                left - 18,
                y + bar_height / 2 + 6,
                run,
                17,
                text_anchor="end",
                font_family="Arial, sans-serif",
                fill="#222",
            )
        )
        total = sum(
            float(stages[stage]["total_seconds"]) for stage, _ in STAGES
        )
        x = float(left)
        for stage, label in STAGES:
            share = 100.0 * float(stages[stage]["total_seconds"]) / total
            segment_width = plot_width * share / 100.0
            if segment_width <= 0:
                continue
            parts.append(
                f'<rect x="{x:g}" y="{y:g}" width="{segment_width:g}" height="{bar_height:g}" fill="{PHASE_COLORS[stage]}"/>'
            )
            if share >= 5.0:
                parts.append(
                    text(
                        x + segment_width / 2,
                        y + bar_height / 2 + 6,
                        f"{share:.1f}%",
                        14,
                        text_anchor="middle",
                        font_family="Arial, sans-serif",
                        fill="white",
                        font_weight="bold",
                    )
                )
            x += segment_width

    legend_y = height - 58
    legend_x = left
    for stage, label in STAGES:
        parts.append(
            f'<rect x="{legend_x:g}" y="{legend_y - 14:g}" width="18" height="18" fill="{PHASE_COLORS[stage]}"/>'
        )
        parts.append(
            text(
                legend_x + 25,
                legend_y,
                label,
                14,
                font_family="Arial, sans-serif",
                fill="#333",
            )
        )
        legend_x += 220
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def draw_svg(
    stats: dict[str, dict[str, dict[str, float | int]]],
    percent: bool = False,
) -> str:
    if percent:
        return draw_percent_stacked_svg(stats)

    width, height = 1280, 760
    left, right, top, bottom = 120, 55, 105, 210
    plot_width = width - left - right
    plot_height = height - top - bottom
    runs = list(stats)
    if percent:
        y_max = 100.0
    else:
        max_mean = max(
            float(stats[run][stage]["mean_seconds"])
            for run in runs
            for stage, _ in STAGES
        )
        y_max = max(100.0, math.ceil(max_mean / 100.0) * 100.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(
            width / 2,
            42,
            "Generation-time share" if percent else "Mean generation time per call",
            25,
            text_anchor="middle",
            font_family="Arial, sans-serif",
            font_weight="bold",
        ),
        text(
            width / 2,
            70,
            (
                "HC Yosys/OpenROAD pipeline; percentage of each run's total generation time"
                if percent
                else "HC Yosys/OpenROAD pipeline; refinement bars are filtered to yosys_openroad"
            ),
            15,
            text_anchor="middle",
            font_family="Arial, sans-serif",
            fill="#555",
        ),
    ]

    for tick in range(0, int(y_max) + 1, int(y_max / 5)):
        y = top + plot_height - plot_height * tick / y_max
        parts.append(
            f'<line x1="{left:g}" y1="{y:g}" x2="{width - right:g}" y2="{y:g}" stroke="#d9d9d9" stroke-width="1"/>'
        )
        parts.append(
            text(
                left - 14,
                y + 5,
                f"{tick:g}{'%' if percent else ''}",
                13,
                text_anchor="end",
                font_family="Arial, sans-serif",
                fill="#555",
            )
        )
    parts.append(
        text(
            35,
            top + plot_height / 2,
            "% of total generation time" if percent else "seconds per call",
            14,
            text_anchor="middle",
            font_family="Arial, sans-serif",
            fill="#555",
            transform=f"rotate(-90 35 {top + plot_height / 2:g})",
        )
    )

    group_width = plot_width / len(STAGES)
    bar_width = min(105.0, group_width * 0.28)
    bar_gap = 14.0
    for index, (stage, label) in enumerate(STAGES):
        center = left + group_width * (index + 0.5)
        parts.append(
            text(
                center,
                top + plot_height + 35,
                label,
                16,
                text_anchor="middle",
                font_family="Arial, sans-serif",
                fill="#222",
            )
        )
        for run_index, run in enumerate(runs):
            run_total = sum(
                float(stats[run][stage_name]["total_seconds"])
                for stage_name, _ in STAGES
            )
            value = (
                100.0 * float(stats[run][stage]["total_seconds"]) / run_total
                if percent
                else float(stats[run][stage]["mean_seconds"])
            )
            x = center + (run_index - (len(runs) - 1) / 2) * (bar_width + bar_gap) - bar_width / 2
            bar_h = plot_height * value / y_max
            y = top + plot_height - bar_h
            parts.append(
                f'<rect x="{x:g}" y="{y:g}" width="{bar_width:g}" height="{bar_h:g}" fill="{RUN_COLORS[run]}"/>'
            )
            parts.append(
                text(
                    x + bar_width / 2,
                    max(top - 4, y - 10),
                    f"{value:.1f}%" if percent else f"{value:.1f}s",
                    14,
                    text_anchor="middle",
                    font_family="Arial, sans-serif",
                    fill="#222",
                    font_weight="bold",
                )
            )

    legend_y = height - 152
    legend_x = left
    for run in runs:
        total = sum(float(stats[run][stage]["total_seconds"]) for stage, _ in STAGES)
        calls = sum(int(stats[run][stage]["calls"]) for stage, _ in STAGES)
        parts.append(
            f'<rect x="{legend_x:g}" y="{legend_y - 14:g}" width="18" height="18" fill="{RUN_COLORS[run]}"/>'
        )
        parts.append(
            text(
                legend_x + 25,
                legend_y,
                f"{run}: {total:,.1f}s total across {calls} events",
                14,
                font_family="Arial, sans-serif",
                fill="#333",
            )
        )
        legend_x += 480

    parts.append(
        text(
            left,
            height - 85,
            "Idea/root events are shared across pipelines; all values are measured wall-clock event durations.",
            13,
            font_family="Arial, sans-serif",
            fill="#555",
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--luna-run", type=Path, required=True)
    parser.add_argument(
        "--terra-run",
        type=Path,
        default=Path("evolve/runs/q3_hc_four_pipeline_ppa_60"),
        help="Optional Terra-medium reference run; omit with --no-terra.",
    )
    parser.add_argument("--no-terra", action="store_true")
    parser.add_argument(
        "--percent",
        action="store_true",
        help="Plot each stage as a percentage of that run's total generation time.",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    run_paths: list[tuple[str, Path]] = [("Luna max", args.luna_run)]
    if not args.no_terra:
        if not args.terra_run.is_file() and not (args.terra_run / "timings.csv").is_file():
            raise SystemExit(f"Terra run not found: {args.terra_run}")
        run_paths.insert(0, ("Terra medium", args.terra_run))

    stats = {label: load_stage_stats(path.resolve()) for label, path in run_paths}
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output_prefix.with_suffix(".svg")
    png_path = output_prefix.with_suffix(".png")
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    svg_path.write_text(draw_svg(stats, percent=args.percent), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("run", "stage", "calls", "total_seconds", "mean_seconds", "share_percent")
        )
        for run, stages in stats.items():
            run_total = sum(
                float(stages[stage]["total_seconds"]) for stage, _ in STAGES
            )
            for stage, _ in STAGES:
                row = stages[stage]
                writer.writerow(
                    (
                        run,
                        stage,
                        row["calls"],
                        row["total_seconds"],
                        row["mean_seconds"],
                        100.0 * float(row["total_seconds"]) / run_total,
                    )
                )
    json_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(svg_path),
                str(png_path),
            ],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"failed to create PNG with ffmpeg: {exc}") from exc

    print(json.dumps(stats, indent=2))
    print(svg_path)
    print(png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
