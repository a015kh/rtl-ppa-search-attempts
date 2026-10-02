"""Hand-checked rank agreement and unchanged sampled evaluation semantics."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from revolution.rank_metrics import metric_rank_fields, paired_rank_metrics


class RankMetricTests(unittest.TestCase):
    def test_strict_monotonic_agreement(self):
        result = paired_rank_metrics([0, 2, 5, 9], [1, 4, 25, 81])
        self.assertAlmostEqual(result["spearman_rho"], 1.0)
        self.assertAlmostEqual(result["kendall_tau_b"], 1.0)
        self.assertEqual((result["n_pairs"], result["excluded_nonfinite"], result["status"]), (4, 0, "ok"))

    def test_reverse_agreement(self):
        result = paired_rank_metrics([1, 2, 3, 4], [9, 8, 7, 6])
        self.assertAlmostEqual(result["spearman_rho"], -1.0)
        self.assertAlmostEqual(result["kendall_tau_b"], -1.0)

    def test_average_spearman_ties_and_tau_b_hand_calculation(self):
        # Truth ranks [1.5,1.5,3] versus prediction ranks [1,2,3].
        # Two concordant pairs and one truth-only tie produce tau-b=2/sqrt(6).
        result = paired_rank_metrics([1, 1, 3], [1, 2, 3])
        self.assertAlmostEqual(result["spearman_rho"], math.sqrt(3) / 2)
        self.assertAlmostEqual(result["kendall_tau_b"], math.sqrt(2 / 3))

    def test_identical_ties_in_both_arrays_have_perfect_agreement(self):
        result = paired_rank_metrics([0, 0, 2, 2], [1, 1, 5, 5])
        self.assertAlmostEqual(result["spearman_rho"], 1.0)
        self.assertAlmostEqual(result["kendall_tau_b"], 1.0)
        self.assertEqual(result["n_pairs"], 4)

    def test_finite_pair_exclusion_preserves_pair_alignment(self):
        result = paired_rank_metrics([0, float("nan"), 2, 3, float("inf"), 5], [0, 1, 4, float("-inf"), 4, 10])
        self.assertEqual((result["n_pairs"], result["excluded_nonfinite"]), (3, 3))
        self.assertAlmostEqual(result["spearman_rho"], 1.0)
        self.assertAlmostEqual(result["kendall_tau_b"], 1.0)

    def test_two_nonconstant_pairs_are_defined(self):
        result = paired_rank_metrics([0, 1], [4, 3])
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["spearman_rho"], -1)
        self.assertAlmostEqual(result["kendall_tau_b"], -1)

    def test_fewer_than_two_pairs_are_null_and_serializable(self):
        for truth, predicted, count in ([], [], 0), ([0], [0], 1), ([float("nan")], [1], 0):
            with self.subTest(count=count):
                result = paired_rank_metrics(truth, predicted)
                self.assertEqual(result["status"], "insufficient_pairs")
                self.assertEqual(result["n_pairs"], count)
                self.assertIsNone(result["spearman_rho"])
                self.assertIsNone(result["kendall_tau_b"])
                self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_constant_status_distinguishes_both_sides(self):
        for truth, predicted, status in ([1, 1], [2, 3], "constant_ground_truth"), ([1, 2], [3, 3], "constant_prediction"), ([0, 0], [1, 1], "constant_both"):
            with self.subTest(status=status):
                result = paired_rank_metrics(truth, predicted)
                self.assertEqual(result["status"], status)
                self.assertIsNone(result["spearman_rho"])
                self.assertIsNone(result["kendall_tau_b"])

    def test_small_but_distinct_values_are_not_approximate_ties(self):
        result = paired_rank_metrics([0, 1e-14, 2e-14], [0, 1, 2])
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["spearman_rho"], 1.0)

    def test_pair_length_and_dimension_must_match(self):
        for truth, predicted in ([1], [1, 2]), ([[1, 2]], [[1, 2]]):
            with self.subTest(truth=truth):
                with self.assertRaisesRegex(ValueError, "one-dimensional"):
                    paired_rank_metrics(truth, predicted)

    def test_flat_field_names_are_metric_specific(self):
        result = metric_rank_fields(["area", "delay"], [[0, 1, 2], [2, 2, 2]], [[0, 2, 4], [1, 2, 3]])
        self.assertEqual(result["area_rank_n_pairs"], 3)
        self.assertEqual(result["area_rank_excluded_nonfinite"], 0)
        self.assertEqual(result["area_rank_status"], "ok")
        self.assertAlmostEqual(result["area_spearman_rho"], 1)
        self.assertIsNone(result["delay_kendall_tau_b"])
        self.assertEqual(result["delay_rank_status"], "constant_ground_truth")

    def test_empty_columns_still_expose_configured_metric_status(self):
        result = metric_rank_fields(["area", "delay", "total_power"], [], [])
        for name in ("area", "delay", "total_power"):
            self.assertEqual(result[f"{name}_rank_n_pairs"], 0)
            self.assertEqual(result[f"{name}_rank_status"], "insufficient_pairs")
            self.assertIsNone(result[f"{name}_spearman_rho"])
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_unlisted_metric_columns_use_existing_evaluator_fallback_names(self):
        result = metric_rank_fields([], [[1, 2]], [[2, 3]])
        self.assertIn("metric0_spearman_rho", result)


class EvaluatorRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("rank_evaluation_fixture", root / "src/text2text_regression/ppa_evaluation.py")
        cls.evaluation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.evaluation)

    def loader(self, labels):
        torch = self.torch
        class Loader(list):
            pass
        loader = Loader([{"input_ids": torch.ones(len(labels), 1, dtype=torch.long), "labels": torch.tensor([[label] for label in labels])}])
        loader.dataset = SimpleNamespace(samples=[{"design_name": str(i), "family_id": "", "clock": 1} for i in range(len(labels))])
        return loader

    def tokenizer(self, mapping):
        class Tokenizer:
            pad_token_id = 0
            def batch_decode(self, rows, skip_special_tokens=True):
                return [mapping[row[0]] for row in rows]
        return Tokenizer()

    def model(self, draws):
        torch = self.torch
        class Model:
            def eval(self):
                return self
            def decode(self, batch, num_samples):
                if num_samples != 2:
                    raise AssertionError("Expected unchanged sampled decoding")
                return torch.tensor(draws)
        return Model()

    def evaluate(self, loader, model, mapping, output):
        return self.evaluation.evaluate_model(model, loader, "fixture", tokenizer=self.tokenizer(mapping), device=self.torch.device("cpu"), metrics=["area", "delay", "total_power"], num_samples=2, output_dir=output)

    def test_additive_ranks_preserve_zero_target_errors_and_decode_failures(self):
        # A malformed draw is ignored; a sample with no valid draw is excluded.
        mapping = {1: "10\n2\n4", 2: "invalid", 3: "0\n1\n1", 4: "0\n1\n1", 10: "10\n2\n4", 11: "0\n0\n0", 12: "1\n1\n1"}
        with tempfile.TemporaryDirectory(prefix="rank-evaluation-") as output:
            result = self.evaluate(self.loader([10, 11, 12]), self.model([[[1], [2]], [[3], [4]], [[2], [2]]]), mapping, output)
            self.assertEqual(result["total_samples"], 2)
            self.assertEqual(result["decode_fail"], 1)
            self.assertAlmostEqual(result["avg_relative_mae"], 2e10 / 6)
            self.assertEqual(result["avg_acc_at_20_percent"], 66.67)
            self.assertEqual(result["all_metrics_acc_at_20_percent"], 50.0)
            self.assertEqual(result["area_acc_at_10_percent"], 100.0)
            self.assertEqual(result["delay_acc_at_10_percent"], 50.0)
            for metric in ("area", "delay", "total_power"):
                self.assertEqual(result[f"{metric}_rank_n_pairs"], 2)
                self.assertEqual(result[f"{metric}_rank_status"], "ok")
                self.assertAlmostEqual(result[f"{metric}_spearman_rho"], 1.0)
            import pandas as pd
            saved = pd.read_csv(Path(output) / "predictions_ground_truth_fixture.csv")
            self.assertEqual(saved["gt_area"].tolist(), [10, 0])
            self.assertEqual(saved["pred_delay"].tolist(), [2, 1])

    def test_nonfinite_decoded_value_changes_only_new_rank_pair_count(self):
        mapping = {1: "nan\n1\n1", 2: "2\n2\n2", 3: "3\n3\n3", 10: "1\n1\n1", 11: "2\n2\n2", 12: "3\n3\n3"}
        with tempfile.TemporaryDirectory(prefix="rank-evaluation-") as output:
            result = self.evaluate(self.loader([10, 11, 12]), self.model([[[1], [1]], [[2], [2]], [[3], [3]]]), mapping, output)
            self.assertEqual((result["total_samples"], result["decode_fail"]), (3, 0))
            self.assertTrue(math.isnan(result["avg_relative_mae"]))
            self.assertEqual(result["area_rank_n_pairs"], 2)
            self.assertEqual(result["area_rank_excluded_nonfinite"], 1)
            self.assertEqual(result["delay_rank_n_pairs"], 3)
            self.assertAlmostEqual(result["area_kendall_tau_b"], 1)

    def test_all_failed_decodes_keep_old_empty_result_and_null_ranks(self):
        mapping = {1: "invalid", 10: "1\n1\n1", 11: "2\n2\n2"}
        with tempfile.TemporaryDirectory(prefix="rank-evaluation-") as output:
            result = self.evaluate(self.loader([10, 11]), self.model([[[1], [1]], [[1], [1]]]), mapping, output)
            self.assertEqual(result["avg_relative_mae"], 1.0)
            self.assertEqual(result["avg_acc_at_10_percent"], 0)
            self.assertEqual(result["decode_fail"], 2)
            self.assertEqual(result["area_rank_n_pairs"], 0)
            self.assertIsNone(result["area_spearman_rho"])
            self.assertEqual(result["area_rank_status"], "insufficient_pairs")
            self.assertFalse((Path(output) / "predictions_ground_truth_fixture.csv").exists())
            json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
