#!/usr/bin/env python3
"""Compare the 10 ns speed cohort with elapsed times from an HC search."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SPEED_ROOT = None
DEFAULT_HC_ROOT = None


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values), quantile))


def load_speed_measurements(
    path: Path,
) -> tuple[list[float], list[float], dict[str, list[float]]]:
    stages: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    benchmark: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["role"] != "main"
                or row["toolchain"] != "dc_pt"
                or row["status"] != "success"
                or row["stage"] not in {"dc", "pt"}
            ):
                continue
            key = (row["case_id"], row["repetition"])
            stages[key][row["stage"]] = float(row["wall_seconds"])
            benchmark[row["case_id"]] = row["benchmark"]

    overall: list[float] = []
    icrtl: list[float] = []
    components: dict[str, list[float]] = {"DC": [], "PT": []}
    for (case_id, _), values in stages.items():
        if set(values) != {"dc", "pt"}:
            continue
        elapsed = values["dc"] + values["pt"]
        overall.append(elapsed)
        components["DC"].append(values["dc"])
        components["PT"].append(values["pt"])
        if benchmark[case_id] == "icrtl":
            icrtl.append(elapsed)
    return overall, icrtl, components


def read_elapsed(path: Path) -> tuple[bool, float] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    elapsed = data.get("elapsed")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed):
        return None
    return bool(data.get("ok")), float(elapsed)


def read_clock(job: Path) -> float | None:
    sdc = job / "dc" / "design.sdc"
    if not sdc.is_file():
        return None
    text = sdc.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"create_clock\b[^\n]*?-period\s+([0-9.]+)", text)
    if not match:
        match = re.search(r"set\s+cycle(?:_clk)?\s+([0-9.]+)", text)
    return float(match.group(1)) if match else None


def load_hc_measurements(
    cache: Path,
) -> tuple[list[float], list[float], dict[float, list[float]], int]:
    successful: list[float] = []
    attempted: list[float] = []
    by_clock: dict[float, list[float]] = defaultdict(list)
    failed = 0
    for dc_path in cache.glob("jobs/*/*/dc_status.json"):
        job = dc_path.parent
        dc = read_elapsed(dc_path)
        if dc is None:
            continue
        dc_ok, dc_elapsed = dc
        pt = read_elapsed(job / "pt_status.json")
        total = dc_elapsed
        if dc_ok and pt is not None and pt[0]:
            total += pt[1]
            successful.append(total)
            clock = read_clock(job)
            if clock is not None:
                by_clock[clock].append(total)
        else:
            failed += 1
        attempted.append(total)
    return successful, attempted, dict(by_clock), failed


def ecdf(ax: plt.Axes, values: list[float], label: str, color: str) -> None:
    ordered = np.sort(np.asarray(values))
    probability = np.arange(1, len(ordered) + 1) / len(ordered)
    ax.step(ordered, probability, where="post", label=f"{label} (n={len(values)})", color=color)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed-root", type=Path, required=True)
    parser.add_argument("--hc-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/figures/synthesis_time_distribution_comparison.png"),
    )
    args = parser.parse_args()

    speed_all, speed_icrtl, components = load_speed_measurements(
        args.speed_root / "measurements.csv"
    )
    hc_success, hc_attempted, hc_by_clock, hc_failed = load_hc_measurements(
        args.hc_root / ".dcpt_cache"
    )
    if not speed_all or not hc_success:
        raise SystemExit("No comparable measurements found")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    ecdf(ax, speed_all, "10 ns cohort", "#3178b5")
    ecdf(ax, speed_icrtl, "10 ns ICRTL subset", "#53a653")
    ecdf(ax, hc_success, "HC six-clock search", "#d24b40")
    ax.set_xscale("log")
    ax.set_xlabel("DC + PT wall time per synthesis point (s, log scale)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("A. Successful Evaluation Latency")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    clocks = sorted(hc_by_clock)
    distributions = [hc_by_clock[clock] for clock in clocks]
    box = ax.boxplot(
        distributions,
        tick_labels=[f"{clock:g}" for clock in clocks],
        showfliers=True,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#efaaa3")
        patch.set_alpha(0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Target clock (ns)")
    ax.set_ylabel("DC + PT wall time (s, log scale)")
    ax.set_title("B. HC Search Cost by Clock Constraint")
    ax.grid(True, axis="y", which="both", alpha=0.2)

    ax = axes[1, 0]
    labels = ["10 ns cohort", "10 ns ICRTL", "HC search"]
    groups = [speed_all, speed_icrtl, hc_success]
    colors = ["#3178b5", "#53a653", "#d24b40"]
    quantiles = [50, 90, 95, 99]
    x = np.arange(len(quantiles))
    width = 0.24
    for index, (label, values, color) in enumerate(zip(labels, groups, colors)):
        heights = [percentile(values, q) for q in quantiles]
        ax.bar(x + (index - 1) * width, heights, width, label=label, color=color)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"P{q}" for q in quantiles])
    ax.set_ylabel("Wall time (s, log scale)")
    ax.set_title("C. Tail Latency Percentiles")
    ax.grid(True, axis="y", which="both", alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    categories = ["Speed cohort\nsuccessful", "HC successful", "HC failed"]
    counts = [len(speed_all), len(hc_success), hc_failed]
    ax.bar(categories, counts, color=["#3178b5", "#d24b40", "#777777"])
    ax.set_ylabel("Synthesis points")
    ax.set_title("D. Coverage and Failure Visibility")
    ax.grid(True, axis="y", alpha=0.2)
    for index, count in enumerate(counts):
        ax.text(index, count, f"{count:,}", ha="center", va="bottom")
    note = (
        f"HC failure-inclusive elapsed: median {percentile(hc_attempted, 50):.1f}s, "
        f"P99 {percentile(hc_attempted, 99):.1f}s"
    )
    ax.text(0.02, 0.88, note, transform=ax.transAxes, va="top", fontsize=9)

    fig.suptitle(
        "Synthesis-Time Distributions: Fixed 10 ns Benchmark vs. Actual HC Search",
        fontsize=15,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    fig.savefig(args.output.with_suffix(".pdf"))

    summary = {
        "speed_cohort": {
            "count": len(speed_all),
            "median_s": percentile(speed_all, 50),
            "p90_s": percentile(speed_all, 90),
            "p99_s": percentile(speed_all, 99),
        },
        "speed_icrtl": {
            "count": len(speed_icrtl),
            "median_s": percentile(speed_icrtl, 50),
            "p90_s": percentile(speed_icrtl, 90),
            "p99_s": percentile(speed_icrtl, 99),
        },
        "hc_success": {
            "count": len(hc_success),
            "median_s": percentile(hc_success, 50),
            "p90_s": percentile(hc_success, 90),
            "p99_s": percentile(hc_success, 99),
        },
        "hc_attempted": {
            "count": len(hc_attempted),
            "failed": hc_failed,
            "median_s": percentile(hc_attempted, 50),
            "p90_s": percentile(hc_attempted, 90),
            "p99_s": percentile(hc_attempted, 99),
            "max_s": max(hc_attempted),
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
