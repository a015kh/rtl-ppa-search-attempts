"""Explicit in-group ordering, macro weights and aggregate-only CLI output."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from revolution.grouped_rank_evaluation import summarize_grouped_csv, write_grouped_summary
from revolution.grouped_rank_metrics import summarize_grouped_pairs
from revolution.rank_metrics import paired_rank_metrics


class GroupedMetricTests(unittest.TestCase):
    def test_high_pooled_correlation_can_hide_all_within_group_reversals(self):
        # Twenty two-item groups, widely separated in absolute magnitude.
        # Pooled rho=1-6/(40**2-1)>0.99, yet every in-group order is reversed.
        gt, pred, groups = [], [], []
        for index in range(20):
            gt.extend((index * 100, index * 100 + 1))
            pred.extend((index * 100 + 1, index * 100))
            groups.extend((str(index), str(index)))
        pooled = paired_rank_metrics(gt, pred)
        self.assertAlmostEqual(pooled["spearman_rho"], 1 - 6 / (40**2 - 1))
        self.assertGreater(pooled["spearman_rho"], 0.99)
        summary = summarize_grouped_pairs(gt, pred, groups, [1] * 40, min_group_size=2)["summary"]
        self.assertAlmostEqual(summary["spearman_rho"], -1)
        self.assertAlmostEqual(summary["kendall_tau_b"], -1)
        self.assertEqual((summary["usable_groups"], summary["usable_rows"]), (20, 40))

    def test_hierarchical_macro_differs_from_group_or_row_weighted_mean(self):
        # Clock 1 has one small correct group and one large reversed group.
        # Clock 2 has one correct group. Equal groups then equal clocks => 0.5.
        gt = [1, 2] + list(range(10)) + [1, 2, 3]
        pred = [1, 2] + list(reversed(range(10))) + [1, 2, 3]
        groups = ["small"] * 2 + ["large"] * 10 + ["other-clock"] * 3
        clocks = [1] * 12 + [2] * 3
        result = summarize_grouped_pairs(gt, pred, groups, clocks, min_group_size=2)
        self.assertAlmostEqual(result["summary"]["spearman_rho"], 0.5)
        self.assertAlmostEqual(result["summary"]["kendall_tau_b"], 0.5)
        self.assertAlmostEqual(result["per_clock"][0]["spearman_rho"], 0)
        self.assertAlmostEqual(result["per_clock"][1]["spearman_rho"], 1)
        self.assertNotAlmostEqual(result["summary"]["spearman_rho"], (1 - 1 + 1) / 3)
        self.assertEqual((result["summary"]["groups_total"], result["summary"]["usable_groups"], result["summary"]["usable_rows"]), (3, 3, 15))

    def test_cutoff_two_vs_three_and_default_ten(self):
        arrays = ([1, 2, 1, 2, 3], [1, 2, 3, 2, 1], ["pair"] * 2 + ["triple"] * 3, [1] * 5)
        two = summarize_grouped_pairs(*arrays, min_group_size=2)["summary"]
        three = summarize_grouped_pairs(*arrays, min_group_size=3)["summary"]
        ten = summarize_grouped_pairs(*arrays)["summary"]
        self.assertAlmostEqual(two["spearman_rho"], 0)
        self.assertAlmostEqual(three["spearman_rho"], -1)
        self.assertEqual((three["groups_below_min"], three["usable_groups"]), (1, 1))
        self.assertEqual((ten["groups_below_min"], ten["usable_groups"]), (2, 0))
        self.assertIsNone(ten["spearman_rho"])
        self.assertIsNone(ten["kendall_tau_b"])
        self.assertEqual(ten["status"], "no_usable_groups")

    def test_exact_ties_follow_shared_hand_calculation(self):
        summary = summarize_grouped_pairs([1, 1, 3], [1, 2, 3], ["tied"] * 3, [1] * 3, min_group_size=3)["summary"]
        self.assertAlmostEqual(summary["spearman_rho"], 3**0.5 / 2)
        self.assertAlmostEqual(summary["kendall_tau_b"], (2 / 3)**0.5)

    def test_constants_have_distinct_audit_counts(self):
        gt = [0, 0, 1, 2, 0, 0, 1, 2]
        pred = [1, 2, 3, 3, 1, 1, 2, 3]
        summary = summarize_grouped_pairs(gt, pred, ["gt", "gt", "pred", "pred", "both", "both", "ok", "ok"], [1] * 8, min_group_size=2)["summary"]
        self.assertEqual(summary["groups_total"], 4)
        self.assertEqual(summary["groups_constant_ground_truth"], 1)
        self.assertEqual(summary["groups_constant_prediction"], 1)
        self.assertEqual(summary["groups_constant_both"], 1)
        self.assertEqual(summary["usable_groups"], 1)
        self.assertEqual(summary["usable_rows"], 2)
        self.assertAlmostEqual(summary["spearman_rho"], 1)

    def test_cutoff_counts_finite_pairs_not_input_rows(self):
        summary = summarize_grouped_pairs([0, 1, 2], [0, 1, float("inf")], ["group"] * 3, [1] * 3, min_group_size=3)["summary"]
        self.assertEqual((summary["rows_total"], summary["finite_pairs"], summary["excluded_nonfinite"]), (3, 2, 1))
        self.assertEqual(summary["groups_below_min"], 1)
        self.assertEqual(summary["usable_rows"], 0)
        self.assertIsNone(summary["spearman_rho"])

    def test_same_family_at_different_clocks_is_two_groups(self):
        result = summarize_grouped_pairs([1, 2, 1, 2], [1, 2, 2, 1], ["same"] * 4, [1, 1, 2, 2], min_group_size=2)
        self.assertEqual(result["summary"]["groups_total"], 2)
        self.assertEqual(result["summary"]["clocks_total"], 2)
        self.assertAlmostEqual(result["summary"]["spearman_rho"], 0)

    def test_clock_rounding_nine_decimal_places(self):
        summary = summarize_grouped_pairs([1, 2], [1, 2], ["same"] * 2, [1.0000000001, 1.0000000002], min_group_size=2)["summary"]
        self.assertEqual((summary["groups_total"], summary["usable_groups"], summary["clocks_total"]), (1, 1, 1))
        self.assertAlmostEqual(summary["spearman_rho"], 1)

    def test_unusable_clock_is_excluded_from_macro_but_counted(self):
        result = summarize_grouped_pairs([1, 2, 3], [1, 2, 0], ["one"] * 2 + ["singleton"], [1, 1, 2], min_group_size=2)
        self.assertEqual((result["summary"]["clocks_total"], result["summary"]["usable_clocks"]), (2, 1))
        self.assertAlmostEqual(result["summary"]["spearman_rho"], 1)
        self.assertEqual(result["per_clock"][1]["status"], "no_usable_groups")

    def test_small_distinct_values_are_not_historically_approximate_constants(self):
        summary = summarize_grouped_pairs([0, 1e-14, 2e-14], [0, 1, 2], ["group"] * 3, [1] * 3, min_group_size=3)["summary"]
        self.assertEqual(summary["usable_groups"], 1)
        self.assertAlmostEqual(summary["spearman_rho"], 1)

    def test_empty_data_is_null_without_group_identifiers(self):
        result = summarize_grouped_pairs([], [], [], [])
        self.assertEqual(result["summary"]["status"], "no_groups")
        self.assertIsNone(result["summary"]["spearman_rho"])
        self.assertEqual(result["per_clock"], [])
        json.dumps(result, allow_nan=False)

    def test_no_group_ids_returned(self):
        result = summarize_grouped_pairs([1, 2], [1, 2], ["nonpublic-identity"] * 2, [1] * 2, min_group_size=2)
        self.assertNotIn("nonpublic-identity", json.dumps(result))
        self.assertEqual(result["per_clock"][0]["clock"], 1)

    def test_missing_group_malformed_clock_lengths_and_cutoff_errors(self):
        for groups, clocks in (["", "known"], [1, 1]), ([None, "known"], [1, 1]), (["known"] * 2, [1, "bad"]), (["known"] * 2, [1, float("nan")]), (["known"] * 2, [1, float("inf")]):
            with self.subTest(groups=groups, clocks=clocks):
                with self.assertRaises(ValueError):
                    summarize_grouped_pairs([1, 2], [1, 2], groups, clocks, min_group_size=2)
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            summarize_grouped_pairs([1], [1], [], [1])
        for cutoff in (0, 1, True, 2.5):
            with self.subTest(cutoff=cutoff):
                with self.assertRaisesRegex(ValueError, "at least two"):
                    summarize_grouped_pairs([], [], [], [], min_group_size=cutoff)


class GroupedCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="grouped-rank-")
        self.path = Path(self.temp.name)
        self.source = self.path / "predictions.csv"
        self.output = self.path / "summary"
        self.root = Path(__file__).resolve().parents[2]

    def tearDown(self):
        self.temp.cleanup()

    def write(self, *, column="family_id", group="explicit-group", clock=1):
        with self.source.open("w", newline="", encoding="utf-8") as target:
            fields = [column, "clock", "design_name", "original_path", *[f"{prefix}_{metric}" for metric in ("area", "delay", "total_power") for prefix in ("gt", "pred")]]
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            for truth, prediction in ((1, 1), (1, 2), (3, 3)):
                record = {column: group, "clock": clock, "design_name": "do-not-publish-design", "original_path": str(self.source)}
                for metric in ("area", "delay", "total_power"):
                    record[f"gt_{metric}"] = truth
                    record[f"pred_{metric}"] = prediction
                writer.writerow(record)

    def test_default_cutoff_ten_is_null_for_three_item_group(self):
        self.write()
        result = write_grouped_summary([("synthetic", self.source)], self.output)
        self.assertEqual(len(result["summary"]), 3)
        for row in result["summary"]:
            self.assertEqual(row["min_group_size"], 10)
            self.assertEqual(row["groups_below_min"], 1)
            self.assertIsNone(row["spearman_rho"])
        self.assertFalse((self.output / "grouped-rank-per-clock.csv").exists())

    def test_multiple_cutoffs_and_per_clock_output_contain_only_aggregates(self):
        self.write(column="group_id", group="nonpublic-group-marker")
        result = write_grouped_summary([("synthetic", self.source)], self.output, min_group_sizes=(2, 3, 10), group_column="group_id", include_per_clock=True)
        self.assertEqual(len(result["summary"]), 9)
        self.assertEqual(len(result["per_clock"]), 9)
        for file in self.output.iterdir():
            text = file.read_text()
            for forbidden in ("nonpublic-group-marker", "do-not-publish-design", str(self.source)):
                self.assertNotIn(forbidden, text)
        metadata = json.loads((self.output / "grouped-rank-summary.json").read_text())
        self.assertEqual(metadata["records"], result["summary"])
        self.assertEqual(metadata["per_clock"], result["per_clock"])
        before = (self.output / "grouped-rank-summary.csv").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            write_grouped_summary([("synthetic", self.source)], self.output)
        self.assertEqual((self.output / "grouped-rank-summary.csv").read_bytes(), before)

    def test_blank_group_never_infers_a_design_prefix(self):
        self.write(group="")
        with self.assertRaisesRegex(ValueError, "Missing explicit group ID"):
            write_grouped_summary([("synthetic", self.source)], self.output, min_group_sizes=(2,))
        self.assertFalse(self.output.exists())

    def test_missing_default_group_column_requires_explicit_override(self):
        self.write(column="group_id")
        with self.assertRaisesRegex(ValueError, "explicit group/clock"):
            summarize_grouped_csv("synthetic", self.source)
        self.assertEqual(len(summarize_grouped_csv("synthetic", self.source, group_column="group_id")["summary"]), 3)

    def test_malformed_or_nonfinite_clock_fails_before_output(self):
        for value in ("bad", "", "nan", "inf"):
            with self.subTest(clock=value):
                self.write(clock=value)
                with self.assertRaises(ValueError):
                    write_grouped_summary([("synthetic", self.source)], self.output, min_group_sizes=(2,))
                self.assertFalse(self.output.exists())

    def test_duplicate_split_and_cutoff_and_source_output_guards(self):
        self.write()
        for arguments in ([('synthetic', self.source)] * 2, []):
            with self.assertRaisesRegex(ValueError, "unique"):
                write_grouped_summary(arguments, self.output)
        with self.assertRaisesRegex(ValueError, "unique"):
            write_grouped_summary([("synthetic", self.source)], self.output, min_group_sizes=(2, 2))
        with self.assertRaisesRegex(ValueError, "outside"):
            write_grouped_summary([("synthetic", self.source)], self.root / "uncreated-grouped-payload")
        self.assertFalse((self.root / "uncreated-grouped-payload").exists())

    def test_script_help_and_execution_from_another_directory(self):
        self.write(column="group_id")
        script = self.root / "scripts/summarize_grouped_rank_correlations.py"
        help_result = subprocess.run([sys.executable, str(script), "--help"], cwd=self.path, capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--group-column", help_result.stdout)
        result = subprocess.run([sys.executable, str(script), "--input", f"synthetic={self.source}", "--output", str(self.output), "--group-column", "group_id", "--min-group-size", "2", "--min-group-size", "3", "--min-group-size", "10", "--per-clock"], cwd=self.path, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["summary_rows"], 9)
        self.assertEqual(json.loads(result.stdout)["per_clock_rows"], 9)


if __name__ == "__main__":
    unittest.main()
