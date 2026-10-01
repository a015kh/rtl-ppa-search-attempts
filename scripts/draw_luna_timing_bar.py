#!/usr/bin/env python3
"""Draw an evolve-style wall-clock phase bar for a Revolution Luna run.

Revolution records the run/generation wall time and the DC/PT job elapsed times,
but the current output format does not split API generation from simulation.
The unlabelled remainder is therefore reported explicitly as
``API/model + simulation + orchestration`` instead of being presented as a
measured generation or simulation duration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from html import escape
from pathlib import Path


PHASES = (
    ("unattributed", "API/model + simulation + orchestration", "#4e79a7"),
    ("dc", "Design Compiler", "#e15759"),
    ("pt", "PrimeTime", "#59a14f"),
)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object in {path}")
    return value


def elapsed(path: Path) -> float | None:
    if not path.is_file():
        return None
    data = read_json(path)
    value = data.get("elapsed")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def find_summary(problem_root: Path) -> Path:
    candidates = sorted(problem_root.glob("*_summary.json"))
    if not candidates:
        raise SystemExit(f"no *_summary.json found below {problem_root}")
    return candidates[0]


def collect(problem_root: Path) -> tuple[dict[str, float], dict[str, int], dict]:
    summary_path = find_summary(problem_root)
    summary = read_json(summary_path)
    total = summary.get("total_runtime_seconds")
    if not isinstance(total, (int, float)) or not math.isfinite(float(total)):
        raise SystemExit(f"{summary_path} has no finite total_runtime_seconds")
    total = float(total)

    dc_seconds = 0.0
    pt_seconds = 0.0
    dc_calls = 0
    pt_calls = 0
    cache = problem_root / ".dcpt_cache"
    for dc_path in sorted(cache.glob("jobs/*/*/dc_status.json")):
        value = elapsed(dc_path)
        if value is not None:
            dc_seconds += value
            dc_calls += 1
        pt_path = dc_path.parent / "pt_status.json"
        value = elapsed(pt_path)
        if value is not None:
            pt_seconds += value
            pt_calls += 1

    # The cache times are sums of recorded stage wall times.  Use the run's
    # wall time as the denominator and expose any remainder explicitly.  If a
    # future parallel run has overlapping stage sums, keep all measured time
    # visible by normalizing by the accounted total instead of going negative.
    remainder = total - dc_seconds - pt_seconds
    if remainder < 0:
        remainder = 0.0
    accounted = remainder + dc_seconds + pt_seconds
    if accounted <= 0:
        raise SystemExit("no positive timing measurements found")

    seconds = {
        "unattributed": remainder,
        "dc": dc_seconds,
        "pt": pt_seconds,
    }
    calls = {"unattributed": 0, "dc": dc_calls, "pt": pt_calls}
    metadata = {
        "problem_root": str(problem_root),
        "summary_path": str(summary_path),
        "model_name": summary.get("model_name"),
        "problem_name": summary.get("problem_name"),
        "total_runtime_seconds": total,
        "dc_status_files": dc_calls,
        "pt_status_files": pt_calls,
        "accounted_seconds": accounted,
        "remainder_definition": "total runtime minus summed DC/PT elapsed times",
        "remainder_note": (
            "API/model generation, simulation, and orchestration are not split "
            "in the current Revolution timing logs."
        ),
    }
    return seconds, calls, metadata


def svg_text(x: float, y: float, value: str, size: int = 16, **attrs: str) -> str:
    attributes = " ".join(
        f'{key.replace("_", "-")}="{escape(str(val))}"'
        for key, val in attrs.items()
    )
    return f'<text x="{x:g}" y="{y:g}" font-size="{size}px" {attributes}>{escape(value)}</text>'


def draw_svg(seconds: dict[str, float], metadata: dict) -> str:
    width, height = 1280, 520
    left, right, top, bottom = 260, 45, 105, 150
    plot_width = width - left - right
    bar_height = 90
    bar_y = top
    total = sum(seconds.values())
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(
            width / 2,
            42,
            "Luna wall-clock phase distribution",
            25,
            text_anchor="middle",
            font_family="Arial, sans-serif",
            font_weight="bold",
        ),
        svg_text(
            left,
            72,
            f"One {metadata.get('problem_name') or 'problem'} Revolution run; percentages use accounted wall time",
            15,
            font_family="Arial, sans-serif",
            fill="#555",
        ),
    ]

    for tick in (0, 25, 50, 75, 100):
        x = left + plot_width * tick / 100.0
        parts.append(
            f'<line x1="{x:g}" y1="{bar_y - 10:g}" x2="{x:g}" y2="{bar_y + bar_height + 8:g}" stroke="#d9d9d9" stroke-width="1"/>'
        )
        parts.append(
            svg_text(
                x,
                bar_y + bar_height + 34,
                f"{tick}%",
                13,
                text_anchor="middle",
                font_family="Arial, sans-serif",
                fill="#555",
            )
        )

    parts.append(
        svg_text(
            left - 18,
            bar_y + bar_height / 2 + 6,
            "gpt-5.6-luna",
            17,
            text_anchor="end",
            font_family="Arial, sans-serif",
            fill="#222",
        )
    )

    x = float(left)
    for phase, label, color in PHASES:
        share = 100.0 * seconds[phase] / total if total else 0.0
        segment_width = plot_width * share / 100.0
        if segment_width <= 0:
            continue
        parts.append(
            f'<rect x="{x:g}" y="{bar_y:g}" width="{segment_width:g}" height="{bar_height:g}" fill="{color}"/>'
        )
        if share >= 4.0:
            parts.append(
                svg_text(
                    x + segment_width / 2,
                    bar_y + bar_height / 2 + 6,
                    f"{share:.1f}%",
                    14,
                    text_anchor="middle",
                    font_family="Arial, sans-serif",
                    fill="white",
                    font_weight="bold",
                )
            )
        x += segment_width

    legend_y = height - 104
    legend_x = left
    for index, (phase, label, color) in enumerate(PHASES):
        share = 100.0 * seconds[phase] / total if total else 0.0
        row_y = legend_y + index * 25
        parts.append(
            f'<rect x="{legend_x:g}" y="{row_y - 14:g}" width="18" height="18" fill="{color}"/>'
        )
        parts.append(
            svg_text(
                legend_x + 25,
                row_y,
                f"{label} ({seconds[phase]:,.1f}s, {share:.1f}%)",
                14,
                font_family="Arial, sans-serif",
                fill="#333",
            )
        )

    parts.append(
        svg_text(
            left,
            bar_y + bar_height + 75,
            "Residual is not a direct API-generation measurement: current logs do not split API, simulation, and orchestration time.",
            13,
            font_family="Arial, sans-serif",
            fill="#555",
        )
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem-root",
        type=Path,
        required=True,
        help="Revolution problem directory containing *_summary.json and .dcpt_cache",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Output prefix; writes .svg, .png, .csv, and .json",
    )
    args = parser.parse_args()
    problem_root = args.problem_root.expanduser().resolve()
    if args.output_prefix is None:
        output_prefix = problem_root / "timing_phase_distribution"
    else:
        output_prefix = args.output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    seconds, calls, metadata = collect(problem_root)
    total = sum(seconds.values())
    svg_path = output_prefix.with_suffix(".svg")
    png_path = output_prefix.with_suffix(".png")
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    svg_path.write_text(draw_svg(seconds, metadata), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("phase", "calls", "seconds", "share_percent"))
        for phase, _, _ in PHASES:
            writer.writerow(
                (phase, calls[phase], seconds[phase], 100.0 * seconds[phase] / total)
            )
    result = {
        **metadata,
        "phases": {
            phase: {
                "calls": calls[phase],
                "seconds": seconds[phase],
                "share_percent": 100.0 * seconds[phase] / total,
            }
            for phase, _, _ in PHASES
        },
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

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

    print(json.dumps(result, indent=2))
    print(svg_path)
    print(png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
