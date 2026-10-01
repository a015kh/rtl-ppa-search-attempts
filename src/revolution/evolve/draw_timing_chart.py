#!/usr/bin/env python3
"""Draw a dependency-free 100% stacked timing-phase bar chart.

The input is the ``timing_breakdown.csv`` emitted by
``compare_proxy_accuracy.py``.  The shared row is intentionally excluded.
SVG is used so this works in the EDA Python environment without matplotlib.
"""

from __future__ import annotations

import argparse
import csv
from html import escape
from pathlib import Path
import subprocess


PIPELINE_ORDER = [
    ("yosys_openroad", "Yosys/OpenROAD"),
    ("trained_model", "Trained model"),
    ("yosys_trained_model", "Yosys + trained"),
    ("dc_pt", "DC + PrimeTime"),
]
PHASES = [
    ("generation", "Generation", "#4e79a7"),
    ("simulation", "Simulation", "#f28e2b"),
    ("synthesis_proxy", "Synthesis / proxy", "#e15759"),
    ("model_startup", "Model startup", "#76b7b2"),
    ("golden_validation", "Golden validation", "#59a14f"),
    ("other", "Other", "#bab0ab"),
]


def read_breakdown(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pipeline = row["pipeline"]
            if pipeline == "shared":
                continue
            rows[pipeline] = {
                phase: float(row.get("share_percent") or 0.0)
                for phase, _, _ in PHASES
                if row.get("category") == phase
            } | rows.get(pipeline, {})
    return rows


def text(x: float, y: float, value: str, size: int = 16, **attrs: str) -> str:
    attributes = " ".join(f'{key.replace("_", "-")}="{escape(str(val))}"' for key, val in attrs.items())
    return f'<text x="{x:g}" y="{y:g}" font-size="{size}px" {attributes}>{escape(value)}</text>'


def draw_svg(rows: dict[str, dict[str, float]]) -> str:
    width, height = 1280, 720
    left, right, top, bottom = 250, 45, 95, 135
    plot_width = width - left - right
    bar_height, gap = 72, 35
    plot_bottom = top + len(PIPELINE_ORDER) * bar_height + (len(PIPELINE_ORDER) - 1) * gap
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 42, "Wall-clock phase distribution (shared generation excluded)", 25,
             text_anchor="middle", font_family="Arial, sans-serif", font_weight="bold"),
        text(left, 70, "Percentage of each pipeline's accounted wall time", 15,
             font_family="Arial, sans-serif", fill="#555"),
    ]

    for tick in (0, 25, 50, 75, 100):
        x = left + plot_width * tick / 100.0
        parts.append(
            f'<line x1="{x:g}" y1="{top - 8:g}" x2="{x:g}" y2="{plot_bottom:g}" '
            'stroke="#d9d9d9" stroke-width="1"/>'
        )
        parts.append(text(x, plot_bottom + 25, f"{tick}%", 13,
                          text_anchor="middle", font_family="Arial, sans-serif", fill="#555"))

    for index, (pipeline, label) in enumerate(PIPELINE_ORDER):
        y = top + index * (bar_height + gap)
        parts.append(text(left - 18, y + bar_height / 2 + 6, label, 17,
                          text_anchor="end", font_family="Arial, sans-serif", fill="#222"))
        x = float(left)
        values = rows.get(pipeline, {})
        for phase, phase_label, color in PHASES:
            share = max(0.0, values.get(phase, 0.0))
            segment_width = plot_width * share / 100.0
            if segment_width <= 0:
                continue
            parts.append(
                f'<rect x="{x:g}" y="{y:g}" width="{segment_width:g}" height="{bar_height:g}" '
                f'fill="{color}"/>'
            )
            if share >= 5.0:
                parts.append(text(x + segment_width / 2, y + bar_height / 2 + 6,
                                  f"{share:.1f}%", 14, text_anchor="middle",
                                  font_family="Arial, sans-serif", fill="white", font_weight="bold"))
            x += segment_width

    legend_y = height - 65
    legend_x = left
    for phase, label, color in PHASES:
        parts.append(f'<rect x="{legend_x:g}" y="{legend_y - 14:g}" width="18" height="18" fill="{color}"/>')
        parts.append(text(legend_x + 25, legend_y, label, 14,
                          font_family="Arial, sans-serif", fill="#333"))
        legend_x += 165 if phase != "synthesis_proxy" else 195
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="timing_breakdown.csv")
    parser.add_argument("--output", type=Path, required=True, help="output SVG path")
    parser.add_argument("--png-output", type=Path, help="optional PNG output for slide software")
    args = parser.parse_args()
    rows = read_breakdown(args.input)
    missing = [pipeline for pipeline, _ in PIPELINE_ORDER if pipeline not in rows]
    if missing:
        raise SystemExit(f"missing pipelines in timing breakdown: {', '.join(missing)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(draw_svg(rows), encoding="utf-8")
    print(args.output)
    if args.png_output:
        args.png_output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(args.output), str(args.png_output)],
            check=True,
        )
        print(args.png_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
