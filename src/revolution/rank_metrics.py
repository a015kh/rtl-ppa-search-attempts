"""Pooled scalar rank agreement for finite prediction/ground-truth pairs.

Rows carry equal weight. Zero values remain observations, Spearman uses average
tie ranks, and Kendall uses tau-b. These are standalone prediction statistics,
not within-family optimization/search scores.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr


def paired_rank_metrics(
    ground_truth: Iterable[float], predictions: Iterable[float]
) -> dict[str, Any]:
    """Compute coefficients after removing pairs containing any nonfinite value.

No imputation, zero-target removal, approximate tie threshold, p-value, or
cross-group averaging is performed. Undefined coefficients serialize as null.
"""
    truth = np.asarray(list(ground_truth), dtype=np.float64)
    predicted = np.asarray(list(predictions), dtype=np.float64)
    if truth.ndim != 1 or predicted.ndim != 1 or len(truth) != len(predicted):
        raise ValueError("Rank metrics require equally sized one-dimensional paired values")
    finite = np.isfinite(truth) & np.isfinite(predicted)
    truth = truth[finite]
    predicted = predicted[finite]
    result = {
        "n_pairs": int(finite.sum()),
        "excluded_nonfinite": int((~finite).sum()),
        "spearman_rho": None,
        "kendall_tau_b": None,
        "status": "ok",
    }
    if len(truth) < 2:
        result["status"] = "insufficient_pairs"
        return result
    truth_constant = bool(np.all(truth == truth[0]))
    predicted_constant = bool(np.all(predicted == predicted[0]))
    if truth_constant or predicted_constant:
        result["status"] = (
            "constant_both" if truth_constant and predicted_constant
            else "constant_ground_truth" if truth_constant else "constant_prediction"
        )
        return result
    rho = float(spearmanr(truth, predicted).statistic)
    tau_b = float(kendalltau(truth, predicted, variant="b", method="auto").statistic)
    result["spearman_rho"] = rho if np.isfinite(rho) else None
    result["kendall_tau_b"] = tau_b if np.isfinite(tau_b) else None
    if result["spearman_rho"] is None or result["kendall_tau_b"] is None:
        result["status"] = "undefined"
    return result


def metric_rank_fields(
    metrics: list[str],
    ground_truth_columns: list[list[float]],
    prediction_columns: list[list[float]],
) -> dict[str, Any]:
    """Add consistently named per-metric fields to an evaluator's result dict."""
    if len(ground_truth_columns) != len(prediction_columns):
        raise ValueError("Ground-truth and prediction metric columns must match")
    output: dict[str, Any] = {}
    for index in range(max(len(metrics), len(ground_truth_columns))):
        name = metrics[index] if index < len(metrics) else f"metric{index}"
        truth = ground_truth_columns[index] if index < len(ground_truth_columns) else []
        predicted = prediction_columns[index] if index < len(prediction_columns) else []
        ranks = paired_rank_metrics(truth, predicted)
        output[f"{name}_rank_n_pairs"] = ranks["n_pairs"]
        output[f"{name}_rank_excluded_nonfinite"] = ranks["excluded_nonfinite"]
        output[f"{name}_spearman_rho"] = ranks["spearman_rho"]
        output[f"{name}_kendall_tau_b"] = ranks["kendall_tau_b"]
        output[f"{name}_rank_status"] = ranks["status"]
    return output
