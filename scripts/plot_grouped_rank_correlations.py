#!/usr/bin/env python3
"""Plot within-problem fixed-clock macro rank correlations from aggregate records."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


METRICS = ("area", "delay", "total_power")
LABELS = ("Area", "Delay", "Total power")
COLORS = ("#0072B2", "#E69F00", "#009E73")


def load_records(path: Path, minimum: int):
    with path.open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source) if int(row["min_group_size"]) == minimum]
    if not rows:
        raise ValueError("No aggregate records")
    splits = list(dict.fromkeys(row["split"] for row in rows))
    indexed = {}
    for row in rows:
        identity = (row["split"], row["metric"])
        if identity in indexed or row["metric"] not in METRICS:
            raise ValueError("Duplicate or unsupported aggregate metric")
        indexed[identity] = row
        for field in ("spearman_rho", "kendall_tau_b"):
            if row[field]:
                value = float(row[field])
                if not math.isfinite(value) or not -1.000000000001 <= value <= 1.000000000001:
                    raise ValueError("Correlation must be finite and between -1 and 1")
    if set(indexed) != {(split, metric) for split in splits for metric in METRICS}:
        raise ValueError("Each split must include all three PPA metrics")
    return splits, indexed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Aggregate rank-correlation CSV")
    parser.add_argument("--output", type=Path, required=True, help="New figure directory")
    parser.add_argument("--min-group-size", type=int, default=10, help="Published group-size policy to plot (default 10)")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; select a new directory")
    splits, indexed = load_records(args.input, args.min_group_size)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13.5, max(5.8, len(splits) * 0.67)), sharey=True)
    y = np.arange(len(splits))
    height = 0.24
    for axis, field, title in zip(axes, ("spearman_rho", "kendall_tau_b"),
                                 ("Spearman rho", "Kendall tau-b")):
        for index, (metric, label, color) in enumerate(zip(METRICS, LABELS, COLORS)):
            values = [float(indexed[(split, metric)][field]) if indexed[(split, metric)][field] else float("nan")
                      for split in splits]
            positions = y + (index - 1) * height
            axis.barh(positions, values, height=height * 0.9, label=label, color=color)
            for position, value in zip(positions, values):
                if math.isfinite(value):
                    axis.text(value + (0.012 if value >= 0 else -0.012), position, f"{value:.3f}",
                              va="center", ha="left" if value >= 0 else "right", fontsize=8)
        for position, split in zip(y, splits):
            if all(not indexed[(split, metric)][field] for metric in METRICS):
                message = "unverified groups" if indexed[(split, METRICS[0])]["status"] == "group_membership_unverified" else "no usable groups"
                axis.text(0.02, position, message, va="center", fontsize=9, color="#666666")
        axis.set_xlim(-1.05, 1.16)
        axis.set_xticks([-1, -0.5, 0, 0.5, 1])
        axis.axvline(0, color="#777777", linewidth=0.7)
        axis.grid(axis="x", alpha=0.18)
        axis.set_axisbelow(True)
        axis.set_title(title)
        axis.set_xlabel("Equal-clock mean of equal-group correlations")
    benchmark_labels = {"test_rtl_opt": "RTL-OPT (local subset)", "test_pluto": "Pluto (local subset)",
                        "test_realbench": "RealBench"}
    labels = []
    for split in splits:
        label = benchmark_labels.get(split, split.replace("test_", "", 1) if split.startswith("test_") else split)
        if any(indexed[(split, metric)]["status"] == "group_membership_unverified" for metric in METRICS):
            label += " (groups unverified)"
        else:
            counts = [indexed[(split, metric)]["usable_groups"] for metric in METRICS]
            label += " (groups=" + "/".join(counts) + ")"
        labels.append(label)
    axes[0].set_yticks(y, labels)
    axes[0].set_ylim(len(splits) - 0.35, -0.65)
    fig.suptitle(f"DC/PT v6.3: within-problem fixed-clock rank correlations (minimum {args.min_group_size} pairs)", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3, frameon=False)
    fig.text(0.5, 0.012, "Groups shown as area/delay/power counts. Undefined/undersized groups are excluded; saved row precision and ties retained.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 0.91))
    args.output.mkdir(parents=True, exist_ok=False)
    for suffix in ("png", "svg"):
        fig.savefig(args.output / f"v6.3-in-group-rank-min{args.min_group_size}.{suffix}", dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
