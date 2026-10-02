# RTL PPA and Search: Research Attempts

Research code for predicting RTL power, delay and area (PPA), and using predictions or synthesis measurements to guide RTL optimization. The release includes a maintained training/evaluation workflow, two search implementations, and selected historical loss and ranking experiments.

The final recorded DC/PT v6.3 model completed ten training epochs. Its standalone evaluation varied substantially across the saved sets:

| Saved evaluation set | Mean metric accuracy below 20% relative error | All three metrics below 20% relative error |
|---|---:|---:|
| `test` | 80.45% | 59.38% |
| `test_realbench` | 31.26% | 14.17% |

These percentages use successfully decoded examples; failed decodes are listed separately in the [complete nine-set results](docs/V63_RESULTS.md) and [aggregate table](results/v6.3/evaluation-summary.csv). Mean metric accuracy averages area, delay and total power; joint accuracy requires all three to pass. Zero targets dominate some saved mean-relative-error values. The [recorded loss curves](docs/V63_RESULTS.md#recorded-loss-curves) show training loss falling while loss on the capped validation subset rises. These measurements do not isolate the cause of the generalization gap. **No retained evidence establishes v6.3 evaluation inside optimization search**, so standalone accuracy is not a demonstrated search benefit.

The population search is derived from [REvolution](https://github.com/kmcho2019/REvolution) and its [paper](https://arxiv.org/abs/2510.21407). The numeric prediction backend and training workflow adapt [Google DeepMind's regress-lm](https://github.com/google-deepmind/regress-lm). The local EvolVE implementation is inspired by the [EvolVE paper](https://arxiv.org/abs/2601.18067). See [third-party notices](THIRD_PARTY_NOTICES.md) for component origins and licenses.

## What is included

- PPA model training, standalone evaluation and sampled inference.
- Offline test-data preparation from completed result/report inputs, with [explicit conversion and filtering policies](docs/TEST_DATASET_PREPARATION.md).
- Dataset assembly from previously measured PPA inputs, with lineage-aware splitting and [recorded source-distribution tables](docs/DATASET_GENERATION.md).
- REvolution-derived population search and a local EvolVE implementation with IC-RTL, VerilogEval and CVDP adapters.
- Historical Number Token Loss, latent-loss, ranking and pairwise-classification attempts, with [an experiment index](docs/EXPERIMENTS.md).
- [Aggregate DC/PT v6.3 training and evaluation results](docs/V63_RESULTS.md), including loss curves and all nine saved split summaries.

Datasets, trained checkpoints, benchmark payloads, cell libraries and per-design results are not included. Supply compatible artifacts and licensed tools locally. The aggregate v6.3 results measure standalone prediction; no retained evidence establishes v6.3 evaluation inside search.

## Start with the offline synthetic walkthrough

From a checkout, run the [synthetic walkthrough](scripts/synthetic_walkthrough.py) using Python 3.11 or newer:

```bash
python scripts/synthetic_walkthrough.py --output /path/to/new-synthetic-walkthrough
```

The [walkthrough guide](docs/SYNTHETIC_WALKTHROUGH.md) explains the fabricated inputs and expected local outputs. It needs no dataset, checkpoint, GPU, API access or synthesis tools. It exercises the real preparation/assembly interfaces with fabricated measurements; it is an engineering demonstration, not model training, PPA inference, optimization search or additional v6.3 evidence.

Use the [workflow map](docs/WORKFLOWS.md) to select a canonical entrypoint and see its required inputs and outputs. The map also distinguishes maintained workflows from historical experiments and auxiliary scripts.

## Installation

Use Python 3.11 or newer in a separate environment. Linux is recommended for GPU training and required by the retained EDA workflows.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[ppa,dev]"
```

Choose a PyTorch build compatible with your GPU. Some model families have access or license requirements. No model is downloaded by the example commands. External tools, API access and PDKs are required only for workflows that use them. The Dockerfile is an optional tool-build recipe; it does not provide commercial tools, model artifacts or cell libraries.

## Prepare evaluation shards

Supply completed result JSON, raw DC/PT reports, or measured resynthesis artifacts to the offline test-data converter. Edit its configuration to choose the explicit conversion/filtering policy and a new output directory:

```bash
python scripts/prepare_test_dataset.py preflight --config /path/to/test-config.json
python scripts/prepare_test_dataset.py build --config /path/to/test-config.json
```

The [test-data preparation guide](docs/TEST_DATASET_PREPARATION.md) defines required inputs and success-filter behavior. This stage writes configured JSONL shards and an audit summary; it does not execute tools. The separate historical overlap-removal simulation is outside this release. Feed compatible prepared shards into dataset assembly below.

## Assemble a dataset

The recovered v6.3 builder combines compatible, already-measured DC/PT inputs; it does not run synthesis or generate RTL. Supply the input files and source catalog described in [dataset generation](docs/DATASET_GENERATION.md), then edit `configs/dataset.example.toml` for your own paths:

```bash
python scripts/generate_dataset.py preflight --config /path/to/dataset.toml
python scripts/generate_dataset.py build --config /path/to/dataset.toml
python scripts/generate_dataset.py verify --config /path/to/dataset.toml
```

The output must be new and separate from inputs and the repository. Catalog creation from upstream datasets requires explicit `--allow-network` and the optional `.[data]` dependencies. Offline assembly uses only Python's standard library. The installed `rtl-ppa-dataset` command provides the same interface. Historical input datasets and generated records are not included in this release.

## Train, evaluate and predict

Copy `configs/ppa.example.json` to a configuration for your own experiment. Select compatible fields, split names, training parameters and inference settings deliberately. Dataset and model locations are explicit:

```bash
python scripts/ppa.py train --config configs/ppa.example.json --dataset /path/to/dataset --model /path/to/base-model --output /path/to/new-training
python scripts/ppa.py evaluate --config configs/ppa.example.json --dataset /path/to/dataset --model /path/to/checkpoint --output /path/to/new-evaluation --split test --device cuda:0
python scripts/ppa.py predict --config configs/ppa.example.json --dataset /path/to/dataset --model /path/to/checkpoint --output /path/to/new-predictions --split test --limit 3 --device cpu
```

Add `--dry-run` to inspect dispatch and input validation. Outputs must be new and separate from input artifacts. Evaluation loads only the requested split and shares the retained evaluator with training. Prediction produces sampled floating-point outputs; invalid decodes are recorded. [Input and checkpoint requirements](docs/CONFIGURATION.md) explain compatibility and optional checksums.

## Optimization search

The release fixes an unreachable auxiliary-file copying loop: problem-specific flat non-RTL inputs are now staged before simulation. [Scope and historical limits](docs/SEARCH_AUXILIARY_FILES.md) explain the fix; hierarchical/include staging is not covered, and saved historical runs are not repaired.

Supply external benchmarks and configure the selected synthesis flow and library. Start with launcher help:

```bash
python scripts/run_evolution.py --help
python scripts/run_evolve.py --help
```

A population-search DC/PT invocation uses explicit benchmark, output, tool-flow and library locations:

```bash
python scripts/run_evolution.py --benchmark_root /path/to/benchmarks --benchmarks ICRTL --problems example_task --save_path /path/to/new-search --synthesis_evaluator dcpt --dcpt_flow_root /path/to/synthesis-flow --dcpt_config /path/to/tool-config.json --dcpt_cell_library /path/to/cell-library/library.v --api_backend openai --model_name your-generation-model
```

Configure API credentials in environment variables required by the chosen backend; never commit them. Select a benchmark/task that matches your external benchmark's manifest and verification files. Commercial-flow integration expects the API described in [configuration documentation](docs/CONFIGURATION.md).

For EvolVE, copy `configs/evolve.example.json`, fill in external benchmark/tool/model locations, and set inference length and sample count explicitly:

```bash
python scripts/run_evolve.py icrtl --config /path/to/evolve.json -- --task example_task --run-dir /path/to/new-evolve-search
python scripts/run_evolve.py four-pipeline --config /path/to/evolve.json -- --pipelines trained_model,dc_pt --run-dir /path/to/new-four-pipeline-search
python scripts/run_evolve.py verilogeval --config /path/to/evolve.json -- --output-root /path/to/new-verilogeval-search
```

The adapters use a separate process and require an explicit output directory. The installed `rtl-ppa-search` command provides the same EvolVE dispatcher; `revolution-evolve` remains a compatibility alias. Their scoring, resume behavior and correctness gates are retained. A simulation pass is not a formal equivalence proof. The local EvolVE implementation is paper-inspired and has not been certified as a faithful reproduction of the paper.

## Historical experiments and recorded results

[Experimental launch instructions](docs/EXPERIMENTS.md) cover six preserved attempts. Their presence establishes implemented ideas, not successful runs or measured improvements. Number Token Loss uses CE weight 1.0 and NT weight 0.3 by default; no verified numerical comparison was recovered. The v6.3 model is not attributed to this loss.

[The v6.3 results](docs/V63_RESULTS.md) document 10 epochs, 146,040 steps, 292 training-loss points, eight capped-validation-loss points and nine standalone prediction summaries. Failed decodes, training-subset limits and zero-target sensitivity are stated alongside the measurements. The public aggregate records do not replace the unavailable datasets or checkpoints.

## Validation and licensing

Run synthetic and migration checks with the development dependencies:

```bash
python -m pytest tests/public_ppa tests/public_config tests/dataset_generation tests/evolve tests/experiments
```

See [validation notes](docs/VALIDATION.md) for the observed release checks and limits. Live training, optimization, commercial synthesis and artifact-specific evaluation require the external dependencies above.

The root [MIT license](LICENSE) preserves the REvolution notice. Google-derived files retain their Apache-2.0 headers and license; the adapted TorchRL excerpt retains its MIT notice. Component-specific terms remain applicable; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
