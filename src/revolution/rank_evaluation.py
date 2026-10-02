"""Summarize rank correlations from explicitly supplied prediction CSVs.

Input files must contain successfully decoded scalar PPA prediction records.
No trained model is loaded and no design records are written to the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

from .rank_metrics import paired_rank_metrics


METRICS = ("area", "delay", "total_power")
FIELDS = ("split", "metric", "decoded_samples", "decode_fail", "finite_pairs", "excluded_nonfinite",
          "spearman_rho", "kendall_tau_b", "spearman_status", "kendall_status")


def summarize_csv(split: str, path: Path, decode_fail: int | None = None) -> list[dict[str, Any]]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", split):
        raise ValueError("Split must be a plain public label")
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {f"{prefix}_{metric}" for metric in METRICS for prefix in ("gt", "pred")}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Prediction CSV requires gt_/pred_ columns for area, delay and total_power")
        pairs = {metric: ([], []) for metric in METRICS}
        count = 0
        for line_number, record in enumerate(reader, 2):
            for metric in METRICS:
                for prefix, values in zip(("gt", "pred"), pairs[metric]):
                    value = record.get(f"{prefix}_{metric}")
                    try:
                        values.append(float(value) if value and value.strip() else float("nan"))
                    except ValueError as exc:
                        raise ValueError(f"Malformed numeric prediction field at CSV line {line_number}") from exc
            count += 1
    result = []
    for metric in METRICS:
        ranks = paired_rank_metrics(*pairs[metric])
        result.append({
            "split": split, "metric": metric, "decoded_samples": count, "decode_fail": decode_fail,
            "finite_pairs": ranks["n_pairs"], "excluded_nonfinite": ranks["excluded_nonfinite"],
            "spearman_rho": ranks["spearman_rho"], "kendall_tau_b": ranks["kendall_tau_b"],
            "spearman_status": ranks["status"], "kendall_status": ranks["status"],
        })
    return result


def write_summary(inputs: list[tuple[str, Path]], output: Path,
                  decode_failures: dict[str, int] | None = None) -> list[dict[str, Any]]:
    output = output.expanduser().resolve()
    if output.exists():
        raise ValueError("Output already exists; select a new directory")
    package = Path(__file__).resolve().parent
    checkout = package.parents[1]
    if output.is_relative_to(package) or ((checkout / "pyproject.toml").is_file() and output.is_relative_to(checkout)):
        raise ValueError("Summary output must be outside the source checkout")
    labels = [split for split, _ in inputs]
    if len(labels) != len(set(labels)):
        raise ValueError("Each split label must be unique")
    decode_failures = decode_failures or {}
    if set(decode_failures) - set(labels) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in decode_failures.values()
    ):
        raise ValueError("Decode failure counts must be nonnegative integers for input split labels")
    rows = []
    for split, path in inputs:
        path = path.expanduser().resolve()
        if output == path or output in path.parents or path in output.parents:
            raise ValueError("Summary output must be separate from all prediction inputs")
        rows.extend(summarize_csv(split, path, decode_failures.get(split)))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "rank-correlation-summary.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "scope": "pooled_split_per_metric",
        "method": {
            "spearman": "Pearson correlation of average ranks for tied values",
            "kendall": "tau-b, with tie correction",
            "pairs": "Successfully decoded CSV rows with finite ground truth and prediction per metric",
            "zeros": "retained", "weighting": "one equal weight per finite saved row",
            "undefined": "null with status for constants or fewer than two finite pairs",
        },
        "limitations": [
            "Input precision is preserved; no inference or target reload is performed",
            "Pooled split coefficients are not within-family candidate-ranking or search evidence",
            "Input files must already exclude failed decodes; omitted failure counts are unknown, not zero",
            "decode_fail is a split-level count repeated for each metric; do not sum its three repetitions",
        ],
        "records": rows,
    }
    (output / "rank-correlation-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return rows


def _input(value: str) -> tuple[str, Path]:
    split, separator, location = value.partition("=")
    if not separator or not split or not location:
        raise argparse.ArgumentTypeError("Use --input SPLIT=/path/to/predictions.csv")
    return split, Path(location)


def _failure_count(value: str) -> tuple[str, int]:
    split, separator, count = value.partition("=")
    if not separator or not split or not count.isdigit():
        raise argparse.ArgumentTypeError("Use --decode-fail SPLIT=NONNEGATIVE_INTEGER")
    return split, int(count)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=_input, required=True,
                        help="SPLIT=CSV, repeat for each successfully decoded prediction file")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--decode-fail", action="append", type=_failure_count, default=[],
                        help="SPLIT=COUNT from saved evaluation metadata; omitted counts remain unknown")
    args = parser.parse_args(argv)
    try:
        if len(args.decode_fail) != len(dict(args.decode_fail)):
            raise ValueError("Each decode-fail split label must be unique")
        rows = write_summary(args.input, args.output, dict(args.decode_fail))
    except (ValueError, OSError) as exc:
        print(f"Rank summary error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "written", "splits": len(args.input), "metric_rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
