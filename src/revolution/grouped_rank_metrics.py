"""Aggregate rank agreement inside explicitly identified family/clock groups.

There is no design-name inference. Group coefficients carry equal weight inside
each rounded clock; defined clock macro coefficients then carry equal weight.
Group identifiers are used privately during calculation and are not returned.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import math
from typing import Any

from .rank_metrics import paired_rank_metrics


COUNT_FIELDS = (
    "rows_total", "finite_pairs", "excluded_nonfinite", "groups_total",
    "groups_below_min", "groups_constant_ground_truth", "groups_constant_prediction",
    "groups_constant_both", "groups_undefined", "usable_groups", "usable_rows",
)


def _counts() -> dict[str, int]:
    return {field: 0 for field in COUNT_FIELDS}


def summarize_grouped_pairs(
    ground_truth: Iterable[float],
    predictions: Iterable[float],
    group_ids: Iterable[str],
    clocks: Iterable[float],
    min_group_size: int = 10,
) -> dict[str, Any]:
    """Return split-level and numeric-clock aggregates without group identities.

The cutoff is applied to finite prediction/target pairs, separately per metric.
Zero targets remain included. Ties and constant detection use exact equality;
this intentionally differs from historical approximate-constant exclusion at
absolute tolerance 1e-12. Undefined/below-cutoff groups do not enter the means.
"""
    if isinstance(min_group_size, bool) or not isinstance(min_group_size, int) or min_group_size < 2:
        raise ValueError("min_group_size must be an integer of at least two")
    truth, predicted, groups, periods = map(list, (ground_truth, predictions, group_ids, clocks))
    if not (len(truth) == len(predicted) == len(groups) == len(periods)):
        raise ValueError("Ground truth, predictions, group IDs and clocks must have equal lengths")
    by_clock: dict[float, dict[str, tuple[list[float], list[float]]]] = defaultdict(dict)
    for index, (gt, pred, group, clock) in enumerate(zip(truth, predicted, groups, periods)):
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"Grouped input row {index} requires an explicit nonblank group ID")
        try:
            period = float(clock)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Grouped input row {index} requires a numeric clock") from exc
        if not math.isfinite(period):
            raise ValueError(f"Grouped input row {index} requires a finite clock")
        period = round(period, 9)
        # Do not normalize or truncate an explicit user-supplied identity.
        pair = by_clock[period].setdefault(group, ([], []))
        pair[0].append(gt)
        pair[1].append(pred)

    clock_results = []
    for clock, clock_groups in sorted(by_clock.items()):
        counts = _counts()
        rhos, taus = [], []
        for gt, pred in clock_groups.values():
            ranks = paired_rank_metrics(gt, pred)
            counts["rows_total"] += len(gt)
            counts["finite_pairs"] += ranks["n_pairs"]
            counts["excluded_nonfinite"] += ranks["excluded_nonfinite"]
            counts["groups_total"] += 1
            if ranks["n_pairs"] < min_group_size:
                counts["groups_below_min"] += 1
            elif ranks["status"] != "ok":
                field = {
                    "constant_ground_truth": "groups_constant_ground_truth",
                    "constant_prediction": "groups_constant_prediction",
                    "constant_both": "groups_constant_both",
                }.get(ranks["status"], "groups_undefined")
                counts[field] += 1
            else:
                counts["usable_groups"] += 1
                counts["usable_rows"] += ranks["n_pairs"]
                rhos.append(ranks["spearman_rho"])
                taus.append(ranks["kendall_tau_b"])
        clock_results.append({
            "clock": clock, "min_group_size": min_group_size, **counts,
            "clocks_total": 1, "usable_clocks": int(bool(rhos)),
            "spearman_rho": math.fsum(rhos) / len(rhos) if rhos else None,
            "kendall_tau_b": math.fsum(taus) / len(taus) if taus else None,
            "status": "ok" if rhos else "no_usable_groups",
        })
    totals = {field: sum(row[field] for row in clock_results) for field in COUNT_FIELDS}
    usable = [row for row in clock_results if row["status"] == "ok"]
    summary = {
        "min_group_size": min_group_size, **totals,
        "clocks_total": len(clock_results), "usable_clocks": len(usable),
        "spearman_rho": math.fsum(row["spearman_rho"] for row in usable) / len(usable) if usable else None,
        "kendall_tau_b": math.fsum(row["kendall_tau_b"] for row in usable) / len(usable) if usable else None,
        "status": "ok" if usable else "no_usable_groups" if clock_results else "no_groups",
    }
    return {"summary": summary, "per_clock": clock_results}
