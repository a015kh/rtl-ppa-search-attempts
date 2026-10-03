"""Summarize in-group Spearman and Kendall from explicitly grouped CSV rows.

Family identities must be supplied by a CSV column; no design-name prefix or
functional-equivalence inference is performed. Output contains only aggregates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

from .grouped_rank_metrics import COUNT_FIELDS, summarize_grouped_pairs


METRICS = ("area", "delay", "total_power")
FIELDS = ("split", "metric", "min_group_size", *COUNT_FIELDS,
          "clocks_total", "usable_clocks", "spearman_rho", "kendall_tau_b", "status")


def summarize_grouped_csv(
    split: str, path: Path, *, min_group_sizes: tuple[int, ...] = (10,),
    group_column: str = "family_id", clock_column: str = "clock",
) -> dict[str, list[dict[str, Any]]]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", split):
        raise ValueError("Split must be a plain public label")
    if not group_column or not clock_column or group_column == clock_column:
        raise ValueError("Group and clock columns must be explicit distinct column names")
    if not min_group_sizes or any(isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in min_group_sizes):
        raise ValueError("Minimum group sizes must be integers of at least two")
    if len(set(min_group_sizes)) != len(min_group_sizes):
        raise ValueError("Minimum group sizes must be unique")
    pairs = {metric: ([], []) for metric in METRICS}
    groups, clocks = [], []
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {group_column, clock_column} | {f"{prefix}_{metric}" for metric in METRICS for prefix in ("gt", "pred")}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("CSV requires explicit group/clock columns and gt_/pred_ scalar PPA columns")
        if len(reader.fieldnames or ()) != len(set(reader.fieldnames or ())):
            raise ValueError("CSV column names must be unique")
        for line_number, record in enumerate(reader, 2):
            group = record.get(group_column)
            if group is None or not group.strip():
                raise ValueError(f"Missing explicit group ID at CSV line {line_number}; supply a verified mapping")
            groups.append(group)
            try:
                clocks.append(float(record.get(clock_column, "")))
                for metric in METRICS:
                    for prefix, values in zip(("gt", "pred"), pairs[metric]):
                        value = record.get(f"{prefix}_{metric}")
                        values.append(float(value) if value and value.strip() else float("nan"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Malformed numeric field at CSV line {line_number}") from exc
    summaries, per_clock = [], []
    for cutoff in min_group_sizes:
        for metric in METRICS:
            result = summarize_grouped_pairs(*pairs[metric], groups, clocks, min_group_size=cutoff)
            summaries.append({"split": split, "metric": metric, **result["summary"]})
            per_clock.extend({"split": split, "metric": metric, **row} for row in result["per_clock"])
    return {"summary": summaries, "per_clock": per_clock}


def write_grouped_summary(
    inputs: list[tuple[str, Path]], output: Path, *, min_group_sizes: tuple[int, ...] = (10,),
    group_column: str = "family_id", clock_column: str = "clock", include_per_clock: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    output = output.expanduser().resolve()
    if output.exists():
        raise ValueError("Output already exists; select a new directory")
    package = Path(__file__).resolve().parent
    checkout = package.parents[1]
    if output.is_relative_to(package) or ((checkout / "pyproject.toml").is_file() and output.is_relative_to(checkout)):
        raise ValueError("Summary output must be outside the source checkout")
    labels = [split for split, _ in inputs]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Input split labels must be nonempty and unique")
    collected = {"summary": [], "per_clock": []}
    for split, path in inputs:
        path = path.expanduser().resolve()
        if output == path or output in path.parents or path in output.parents:
            raise ValueError("Summary output must be separate from all prediction inputs")
        result = summarize_grouped_csv(split, path, min_group_sizes=min_group_sizes,
                                       group_column=group_column, clock_column=clock_column)
        collected["summary"].extend(result["summary"])
        if include_per_clock:
            collected["per_clock"].extend(result["per_clock"])
    output.mkdir(parents=True, exist_ok=False)
    for name, fields, rows in (
        ("grouped-rank-summary.csv", FIELDS, collected["summary"]),
        ("grouped-rank-per-clock.csv", ("split", "metric", "clock", *FIELDS[2:]), collected["per_clock"]),
    ):
        if name.endswith("per-clock.csv") and not include_per_clock:
            continue
        with (output / name).open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    metadata = {
        "scope": "within_explicit_group_and_fixed_clock",
        "method": {
            "grouping": "Explicit nonblank group IDs; numeric clock rounded to 9 decimal places",
            "cutoff": "Minimum finite prediction/ground-truth pairs separately per metric and group",
            "ties": "Exact average-rank Spearman and Kendall tau-b; no approximate constant threshold",
            "weighting": "Equal weight per defined group within each clock, then equal weight per defined clock",
            "undefined": "Constants and below-cutoff groups excluded from means; empty means are null with status",
            "zeros": "Finite zero targets retained",
        },
        "limitations": [
            "No automatic design-name parsing, functional verification, or group membership recovery",
            "Group labels are supplied evidence; their functional validity is not established by this tool",
            "Failed decodes absent from input CSVs are not reconstructed or penalized",
            "No per-group identities, raw prediction rows or input paths are written",
            "These recorded prediction ranks do not establish optimization-search improvement",
        ],
        "records": collected["summary"],
    }
    if include_per_clock:
        metadata["per_clock"] = collected["per_clock"]
    (output / "grouped-rank-summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return collected


def _input(value: str) -> tuple[str, Path]:
    split, separator, location = value.partition("=")
    if not separator or not split or not location:
        raise argparse.ArgumentTypeError("Use --input SPLIT=/path/to/grouped-predictions.csv")
    return split, Path(location)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, action="append", type=_input,
                        help="SPLIT=CSV, repeat for each explicitly grouped prediction input")
    parser.add_argument("--output", required=True, type=Path, help="New output directory outside the source checkout")
    parser.add_argument("--group-column", default="family_id", help="Explicit verified group ID column; no inference")
    parser.add_argument("--clock-column", default="clock", help="Numeric clock column")
    parser.add_argument("--min-group-size", action="append", type=int, help="Minimum finite pairs per group, repeat for multiple cutoffs; default 10")
    parser.add_argument("--per-clock", action="store_true", help="Also save aggregates per numeric clock")
    args = parser.parse_args(argv)
    try:
        result = write_grouped_summary(args.input, args.output,
                                       min_group_sizes=tuple(args.min_group_size or (10,)),
                                       group_column=args.group_column, clock_column=args.clock_column,
                                       include_per_clock=args.per_clock)
    except (OSError, ValueError, TypeError) as exc:
        print(f"Grouped rank summary error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "written", "splits": len(args.input), "summary_rows": len(result["summary"]), "per_clock_rows": len(result["per_clock"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
