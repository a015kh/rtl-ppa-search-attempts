# Workflow map and repository layout

Use the canonical entrypoints below from a source checkout. Every artifact-producing workflow receives explicit user-owned input locations and a new output location. The public repository contains source, synthetic tests and aggregate historical measurements; collected datasets, checkpoints, benchmark payloads and libraries are external.

## Select a workflow

| Task | Canonical entrypoint | Required inputs | Local outputs |
|---|---|---|---|
| Try the offline setup | `python scripts/synthetic_walkthrough.py --output /path/to/new-walkthrough` | Python 3.11+; newly fabricated fixtures are created locally | Fabricated input/configuration files, prepared test shards and audit, assembled dataset and `summary.json` assertions |
| Prepare measured evaluation shards | `python scripts/prepare_test_dataset.py preflight` then `build`, each with `--config` | Explicit completed result/report artifacts and conversion/filtering policy | Configured test JSONL shards and `audit.json` counts/checksums |
| Assemble a training/evaluation dataset | `python scripts/generate_dataset.py preflight`, `build`, `verify`, each with `--config` | Measured base/external/augmentation/test inputs, provenance inputs and local catalog | Training/validation/test JSONL shards, lineage/source metadata, manifest and verification summary |
| Train the maintained PPA model | `python scripts/ppa.py train` | `--config`, compatible JSONL `--dataset`, local base `--model`, new `--output` | Model checkpoints, training history and configured training/evaluation records |
| Evaluate a compatible PPA checkpoint | `python scripts/ppa.py evaluate` | `--config`, `--dataset`, exact `--model`, new `--output`, explicit `--split` | Prediction/ground-truth CSV, metric summary and evaluation configuration |
| Summarize saved prediction ranking | `python scripts/summarize_rank_correlations.py` | One or more explicit `--input split=/path/to/predictions.csv`; new `--output`; optional recorded `--decode-fail split=count` | Aggregate-only per-metric Spearman rho/Kendall tau-b CSV and JSON; no design records |
| Evaluate ranking within problems | `python scripts/summarize_grouped_rank_correlations.py` | Explicit grouped prediction CSVs, verified group-ID/clock columns, minimum finite-pair policy and new output | Equal-group/equal-clock macro correlations and exclusion counts; optional numeric-clock aggregates |
| Predict sampled PPA values | `python scripts/ppa.py predict` | Same explicit artifact/configuration locations and `--split`; optional limit/device | Prediction JSON with decoded values and failed-decode records |
| Run REvolution-derived population search | `python scripts/run_evolution.py` | External benchmark root, generation backend/model, chosen evaluator; synthesis flow/library or compatible estimator as required; explicit save path | Candidate RTL, correctness/PPA feedback, run records and summaries |
| Run local EvolVE search adapters | `python scripts/run_evolve.py icrtl`, `four-pipeline`, `verilogeval` or `cvdp` | Explicit JSON `--config`, external benchmark/tool/model inputs, adapter arguments after `--`, output directory | Candidate/state records, verification and evaluator measurements, run summaries |
| Inspect historical learning objectives | `python scripts/run_experiment.py <choice>` | Explicit `--dataset`, `--model`, `--output`; historical options after `--` | The selected experiment's checkpoints, logs and evaluation records |
| Inspect recorded v6.3 findings | [v6.3 results document](V63_RESULTS.md) and `results/v6.3/` | Bundled aggregate tables/curves | No execution required; historical standalone evidence only |

The installed `rtl-ppa-test-dataset` command mirrors measured test-data preparation and `rtl-ppa-dataset` mirrors dataset assembly. `rtl-ppa-search` mirrors the EvolVE dispatcher, with `revolution-evolve` retained as a compatibility alias. `rtl-ppa-walkthrough` mirrors the offline synthetic walkthrough, `rtl-ppa-rank-summary` mirrors the pooled saved-prediction rank summary, and `rtl-ppa-grouped-rank-summary` mirrors the within-problem summary. The PPA and historical-experiment commands above are source-checkout launchers.

## Inputs and preparation order

The shortest setup check is the [synthetic walkthrough](SYNTHETIC_WALKTHROUGH.md). It creates fabricated measured inputs and uses real public preparation/assembly code. It requires no model weights or EDA tools and makes no research-performance claim.

For your own data, first complete RTL collection, candidate generation and measurement in your chosen upstream workflow. The offline [test-data preparation](TEST_DATASET_PREPARATION.md) stage can then convert completed result JSON, parse/filter explicitly supplied DC/PT report jobs, or select successful measured resynthesis rows. Choose the `legacy_result_json`, `dcpt_inputs_reports` or `resynthesis_success` policy deliberately; these have different input contracts. Historical overlap-removal simulation is a separate historical stage not included in this release; its invocation and validation are unverified, so conversion alone is not a full historical benchmark reconstruction.

Next, [dataset assembly](DATASET_GENERATION.md) consolidates compatible measured inputs, applies the recovered deduplication/source/lineage policy and writes dataset splits. It does not run synthesis, generate RTL or manufacture fresh PPA labels. Offline assembly uses a local prepared source catalog; public catalog reconstruction requires explicit `--allow-network` and the optional data dependencies.

Finally, point the maintained PPA launcher at your compatible local dataset/model artifacts and an explicit experiment configuration. Training, standalone evaluation and sampled inference share the retained formatting/backend assumptions. [Configuration documentation](CONFIGURATION.md) describes split fields, checkpoint format, optional checksums and external dependencies. A new dataset or model is a new experiment until its provenance and content are checked; these commands do not by themselves reproduce the published v6.3 checkpoint or results.

Search is a separate workflow. Supply an external benchmark, generation model and correctness strategy, then choose a compatible trained estimator or reference synthesis evaluator. Its output is evidence for that run's code/model/tool configuration. Standalone PPA prediction accuracy is not a substitute for measured search performance.

The maintained evaluator also reports per-metric Spearman rho and Kendall tau-b for decoded finite pairs. The rank-summary utility computes the same coefficients from already saved prediction CSVs without inference. Its input contract, tie handling and limits are described in [v6.3 results](V63_RESULTS.md#post-hoc-rank-correlations). Supply failed-decode counts separately when known; CSVs contain decoded examples only. `scripts/plot_rank_correlations.py` plots the resulting aggregate table. Pooled split correlations do not establish within-family candidate ranking or search improvement.

Use the separate [in-group evaluation](IN_GROUP_EVALUATION.md) utility for the
central within-problem ranking question. It requires explicit verified group
membership, partitions by fixed clock and reports exclusion/coverage counts.
It does not infer functional families from filenames or lineage hashes.

## Repository layout

| Location | Purpose |
|---|---|
| `scripts/` | Public launchers listed above, compatibility wrappers and auxiliary report/plot utilities |
| `configs/` | Generic PPA, dataset, preparation and search examples; no private artifact locations |
| `src/text2text_regression/` | Maintained training, standalone evaluation, shared metrics/formatters and Google-derived numeric model code |
| `src/revolution/` | REvolution-derived population search, evaluator adapters, dataset/preparation support and generic synthesis resources |
| `src/revolution/evolve/` | Separate local EvolVE engine and benchmark/evaluator adapters |
| `experiments/training/` | Selected historical token, latent, ranking, pairwise and post-finetuning attempts |
| `tests/` | Synthetic unit/interface/migration checks; external payload checks are identified separately |
| `docs/` | Workflow/input descriptions, historical experiment index, results, attribution and validation limits |
| `results/v6.3/`, `figures/` | Sanitized recorded aggregates and plots; no raw designs, predictions or weights |
| `licenses/`, `THIRD_PARTY_NOTICES.md`, `LICENSE` | Applicable source origins, authorship notices and component licenses |

The internal `revolution` module name is retained to preserve imports and REvolution provenance. The public package is named `rtl-ppa-search-attempts`.

## Maintained workflows, historical attempts and utilities

The table's PPA, dataset/preparation and search launchers are the canonical public interfaces. Their explicit inputs replace private machine/artifact defaults. “Canonical” identifies the supported configuration path; a help/import/synthetic check does not imply successful full training or a verified end-to-end search advantage. External artifacts and tools remain necessary for those runs.

Historical attempts are selected through `scripts/run_experiment.py`. The [experiment index](EXPERIMENTS.md) and [detailed usage notes](../experiments/README.md) explain their separate split schemas, model heads, dependencies and evidence limits. These objectives are not validated interchangeable training defaults, and no verified Number Token Loss comparison or NT-loss attribution for v6.3 is claimed.

Other report, plotting, benchmark-building and augmentation helpers remain inspectable source utilities. They are not alternate canonical training/evaluation launchers. Older wrappers and duplicate-looking module files exist for compatibility and historical organization; use the public launcher table instead of choosing a script by filename alone. Utility-generated tables should retain the actual model/tool identity and should not relabel historical results as v6.3 evidence.

The published [validation record](VALIDATION.md) distinguishes observed engineering checks, external dependency skips and inherited failures from research experiments. The source-only release has no retained evidence that the v6.3 model ran inside optimization search.
