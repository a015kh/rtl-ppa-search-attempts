# Sanitized historical search aggregates

See [search findings and definitions](../../docs/HISTORICAL_SEARCH_EXPERIMENTS.md).

- evolve-task-comparisons.csv/json: all 18 unique completed paired tasks, with the earlier v6.2.1 estimator.
- evolve-finalist-status-coverage.json: success and failure coverage without candidate records.
- evolve-timing-by-task.csv and evolve-timing-summary.csv/json: selected non-wrapper durations, including shared generation once per arm.
- evolve-luna-timing-example.json: separate retained chart reconstruction.
- revolution-proxy-coverage.json: 15 earlier v6.1.4 proxy task summaries; no matched DC/PT arm.

PPA ratios and reductions are task-level aggregate measurements. Seconds are saved event durations. Cold search excludes shared generation and final validation. No raw design records, predictions or library information are included. Private audits verify budgets, checkpoint loads, counts, golden scores, baseline reductions and timing. No experiments were rerun.
