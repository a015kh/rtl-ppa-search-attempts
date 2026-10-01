#!/usr/bin/env python3
"""Draw a wall-clock phase chart for one evolve pipeline.

The chart combines the shared model-generation stages with the selected
pipeline's VCS and synthesis stages.  It reads ``timings.csv`` directly and
does not require sqlite, matplotlib, or an EDA environment.  Parent wrapper
events are excluded so their child events are not counted twice.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from html import escape
from pathlib import Path


PARENT_STAGES = {
    "pipeline_search",
    "vcs_functional",
    "pipeline_point",
    "golden_validation",
    "report_generation",
    "preflight",
    "finalist_selection",
}
GENERATION_STAGES = {
    "idea_generation",
    "codex_root_generation",
    "codex_refinement",
}
VCS_STAGES = {"vcs_compile", "vcs_simulation"}
SYNTHESIS_STAGES = {
    "yosys",
    "openroad",
    "model_input_prepare",
    "model_inference",
    "model_decode",
    "dc",
    "pt_fixed",
}
PHASES = (
    ("generation", "Generation", "#4e79a7"),
    ("vcs_simulation", "VCS simulation", "#f28e2b"),
    ("synthesis", "Synthesis / proxy", "#e15759"),
    ("other", "Other", "#bab0ab"),
)


def load_phase_stats(run_dir: Path, pipeline: str) -> dict[str, dict[str, float | int]]:
    """Aggregate event durations without double-counting wrapper events.

    Generation events have an empty pipeline field because they are shared by
    all pipelines.  They are included by default; events belonging to other
    pipelines (for example the DC/PT baseline) are ignored.
    """
    path = run_dir / "timings.csv"
    if not path.is_file():
        raise SystemExit(f"timings.csv not found: {path}")

    values: dict[str, list[float]] = {phase: [] for phase, _, _ in PHASES}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stage = row.get("stage", "")
            if stage in PARENT_STAGES:
                continue
            try:
                seconds = float(row.get("wall_seconds", ""))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(seconds) or seconds < 0:
                continue

            row_pipeline = row.get("pipeline", "")
            if row_pipeline in {"", "shared"}:
                # Only shared generation belongs in this pipeline's chart.
                if stage in GENERATION_STAGES:
                    values["generation"].append(seconds)
                continue
            if row_pipeline != pipeline:
                continue

            if stage in GENERATION_STAGES:
                values["generation"].append(seconds)
            elif stage in VCS_STAGES:
                values["vcs_simulation"].append(seconds)
            elif stage in SYNTHESIS_STAGES:
                values["synthesis"].append(seconds)
            else:
                values["other"].append(seconds)

    if not any(values.values()):
        raise SystemExit(f"no timing events found for pipeline {pipeline!r} in {path}")
    return {
        phase: {
            "calls": len(samples),
            "total_seconds": sum(samples),
            "mean_seconds": sum(samples) / len(samples) if samples else 0.0,
        }
        for phase, samples in values.items()
    }


def text(x: float, y: float, value: str, size: int = 16, **attrs: str) -> str:
    attributes = " ".join(
        f'{key.replace("_", "-")}="{escape(str(val))}"'
        for key, val in attrs.items()
    )
    return f'<text x="{x:g}" y="{y:g}" font-size="{size}px" {attributes}>{escape(value)}</text>'


def draw_svg(
    stats: dict[str, dict[str, float | int]],
    pipeline_label: str,
    include_shared_generation: bool,
) -> str:
    width, height = 1280, 610
    left, right, top = 250, 45, 105
    plot_width = width - left - right
    bar_height = 72
    plot_bottom = top + bar_height
    total = sum(float(stats[phase]["total_seconds"]) for phase, _, _ in PHASES)
    if total <= 0:
        raise SystemExit("phase total is zero")

    title = "Wall-clock timing distribution"
    subtitle = (
        f"Generation, VCS simulation, and synthesis/proxy · {pipeline_label}"
        + (" · shared generation included" if include_shared_generation else "")
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 42, title, 25, text_anchor="middle", font_family="Arial, sans-serif", font_weight="bold"),
        text(left, 70, subtitle, 15, font_family="Arial, sans-serif", fill="#555"),
    ]

    for tick in (0, 25, 50, 75, 100):
        x = left + plot_width * tick / 100.0
        parts.append(
            f'<line x1="{x:g}" y1="{top - 8:g}" x2="{x:g}" y2="{plot_bottom:g}" stroke="#d9d9d9" stroke-width="1"/>'
        )
        parts.append(text(x, plot_bottom + 25, f"{tick}%", 13, text_anchor="middle", font_family="Arial, sans-serif", fill="#555"))

    parts.append(text(left - 18, top + bar_height / 2 + 6, pipeline_label, 17, text_anchor="end", font_family="Arial, sans-serif", fill="#222"))
    x = float(left)
    shares: dict[str, float] = {}
    for phase, label, color in PHASES:
        share = 100.0 * float(stats[phase]["total_seconds"]) / total
        shares[phase] = share
        segment_width = plot_width * share / 100.0
        if segment_width <= 0:
            continue
        parts.append(f'<rect x="{x:g}" y="{top:g}" width="{segment_width:g}" height="{bar_height:g}" fill="{color}"/>')
        # Keep the bar readable while reporting small phases in the table below.
        if share >= 5.0:
            parts.append(text(x + segment_width / 2, top + bar_height / 2 + 6, f"{share:.1f}%", 14, text_anchor="middle", font_family="Arial, sans-serif", fill="white", font_weight="bold"))
        x += segment_width

    legend_y = 300
    legend_x = left
    for phase, label, color in PHASES:
        parts.append(f'<rect x="{legend_x:g}" y="{legend_y - 14:g}" width="18" height="18" fill="{color}"/>')
        parts.append(text(legend_x + 25, legend_y, label, 14, font_family="Arial, sans-serif", fill="#333"))
        legend_x += 235

    parts.append(text(left, 355, "Accounted event wall time", 16, font_family="Arial, sans-serif", font_weight="bold", fill="#222"))
    summary_y = 390
    for phase, label, _ in PHASES:
        seconds = float(stats[phase]["total_seconds"])
        calls = int(stats[phase]["calls"])
        parts.append(text(left, summary_y, f"{label}: {seconds:,.1f}s ({shares[phase]:.1f}%), {calls} events", 14, font_family="Arial, sans-serif", fill="#333"))
        summary_y += 28
    parts.append(text(left, height - 28, "Parent wrapper events are excluded to avoid double-counting; DC/PT events are not included in this Yosys/OpenROAD view.", 13, font_family="Arial, sans-serif", fill="#555"))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pipeline", default="yosys_openroad")
    parser.add_argument("--pipeline-label", default="Yosys/OpenROAD")
    parser.add_argument("--exclude-shared-generation", action="store_true")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    stats = load_phase_stats(args.run_dir.resolve(), args.pipeline)
    if args.exclude_shared_generation:
        # Re-read only the selected pipeline's generation rows by subtracting
        # the shared idea/root stages from the aggregate.
        path = args.run_dir.resolve() / "timings.csv"
        shared = {"idea_generation", "codex_root_generation"}
        shared_seconds = 0.0
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("pipeline", "") not in {"", "shared"} or row.get("stage") not in shared:
                    continue
                try:
                    value = float(row.get("wall_seconds", ""))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value >= 0:
                    shared_seconds += value
        stats["generation"]["total_seconds"] = max(0.0, float(stats["generation"]["total_seconds"]) - shared_seconds)

    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output_prefix.with_suffix(".svg")
    png_path = output_prefix.with_suffix(".png")
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    svg_path.write_text(draw_svg(stats, args.pipeline_label, not args.exclude_shared_generation), encoding="utf-8")

    total = sum(float(stats[phase]["total_seconds"]) for phase, _, _ in PHASES)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("pipeline", "phase", "calls", "total_seconds", "mean_seconds", "share_percent"))
        for phase, label, _ in PHASES:
            row = stats[phase]
            writer.writerow((args.pipeline, phase, row["calls"], row["total_seconds"], row["mean_seconds"], 100.0 * float(row["total_seconds"]) / total if total else 0.0))
    json_path.write_text(json.dumps({"pipeline": args.pipeline, "pipeline_label": args.pipeline_label, "include_shared_generation": not args.exclude_shared_generation, "phases": stats}, indent=2) + "\n", encoding="utf-8")

    try:
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(svg_path), str(png_path)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"failed to create PNG with ffmpeg: {exc}") from exc
    print(svg_path)
    print(png_path)
    print(csv_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
