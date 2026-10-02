"""Aggregate-only rank summaries must preserve pairing and exclude raw records."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from revolution.rank_evaluation import write_summary


ROOT = Path(__file__).resolve().parents[2]


def predictions(path):
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["design_name", *[
            f"{prefix}_{metric}" for metric in ("area", "delay", "total_power") for prefix in ("gt", "pred")]])
        writer.writeheader()
        for index, (truth, prediction) in enumerate(((1, 1), (1, 2), (3, 3), (4, float("inf")))):
            row = {"design_name": "private_design_marker"}
            for metric in ("area", "delay", "total_power"):
                row[f"gt_{metric}"] = truth
                row[f"pred_{metric}"] = prediction
            writer.writerow(row)


def test_summary_preserves_ties_pair_counts_and_omits_identifiers(tmp_path):
    source = tmp_path / "predictions.csv"
    predictions(source)
    output = tmp_path / "summary"
    rows = write_summary([("synthetic", source)], output)
    assert len(rows) == 3
    for row in rows:
        assert row["decoded_samples"] == 4
        assert row["finite_pairs"] == 3
        assert row["excluded_nonfinite"] == 1
        assert row["spearman_rho"] == pytest.approx(3**0.5 / 2)
        assert row["kendall_tau_b"] == pytest.approx((2/3)**0.5)
    for file in output.iterdir():
        assert "private_design_marker" not in file.read_text()
        assert str(source) not in file.read_text()
    assert json.loads((output / "rank-correlation-summary.json").read_text())["records"] == rows
    saved = (output / "rank-correlation-summary.csv").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        write_summary([("synthetic", source)], output)
    assert (output / "rank-correlation-summary.csv").read_bytes() == saved


def test_cli_runs_from_another_directory(tmp_path):
    source = tmp_path / "predictions.csv"
    predictions(source)
    output = tmp_path / "summary"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/summarize_rank_correlations.py"),
                             "--input", f"synthetic={source}", "--output", str(output)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["metric_rows"] == 3


def test_bad_columns_and_duplicate_split_fail_before_output(tmp_path):
    source = tmp_path / "predictions.csv"
    source.write_text("unrelated_column\n1\n")
    with pytest.raises(ValueError, match="requires gt_/pred_"):
        write_summary([("synthetic", source)], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    predictions(source)
    with pytest.raises(ValueError, match="unique"):
        write_summary([("synthetic", source), ("synthetic", source)], tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()
