# Dataset construction and retained v6.3 source distribution

The released dataset builder consolidates **already measured Design Compiler / PrimeTime rows** into a unified dataset with provenance, deduplication and lineage-aware training/validation splits. It does not collect arbitrary RTL, generate augmentation candidates, run synthesis or create fresh PPA labels. Those upstream stages require separately prepared inputs and tool flows. No collected dataset, source catalog, lineage payload or benchmark records are included in this code release.

The [project findings](PROJECT_FINDINGS.md) describe this one retained version as
the **final DC/PT training corpus**, historically identified as `v6.3.0`. Its PPA
data remains private. The public Verilog-only companion subset is a separate,
rights-reviewed artifact, not a public copy of the measured training corpus.
Training augmentation and the **Same-Task RTL Ranking Suite** evaluation
candidates serve different roles; the [suite naming guide](IN_GROUP_EVALUATION.md)
maps reader-facing cohort names to unchanged saved split identifiers.

The separate [test-result preparation workflow](TEST_DATASET_PREPARATION.md)
converts supplied historical result JSON or completed DC/PT reports into test
JSONL. It exposes named filtering policies and records accepted/rejected counts;
it does not execute synthesis. Supply those prepared files through `[tests.sources]`
in the assembler configuration. The [synthetic walkthrough](SYNTHETIC_WALKTHROUGH.md)
runs test conversion and dataset assembly using invented inputs without external
artifacts. [The workflow map](WORKFLOWS.md) distinguishes these supported stages
from historical helpers.

For a source-only companion dataset, use the separate
[Verilog-only exporter](VERILOG_ONLY_RELEASE.md). It removes measurement metadata
by selecting only explicit source fields, deduplicates exact source strings and
reports both measured-row and unique-code mixture counts. It does not distribute
the PPA dataset or assume that a public source is licensed for redistribution.

## Run the builder with explicit inputs

Copy `configs/dataset.example.toml` to an experiment configuration and supply all required historical-format artifacts. Inspect the interface first:

```bash
python scripts/generate_dataset.py --help
python scripts/generate_dataset.py preflight --config configs/dataset.example.toml
python scripts/generate_dataset.py build --config configs/dataset.example.toml
python scripts/generate_dataset.py verify --config configs/dataset.example.toml
```

Execution is offline by default. A local prepared source catalog avoids network access. Optional catalog reconstruction from public Hugging Face datasets requires explicit `--allow-network`; this permits retrieval of the configured public source revisions, not training or synthesis. Input paths, selected source revisions and output location come from configuration. Use a new output directory separate from all inputs. Raw source collection and DC/PT labeling must already be complete. No trained estimator or older model weights are needed to consolidate the dataset.

The recovered workflow requires measured base/test shards, prepared and measured external-candidate batches, their selection metadata, measured augmentation shards, the legacy source-matching input and a compatible source catalog (or explicitly authorized reconstruction). The configuration example names their roles. These inputs are excluded from the repository; downloading an upstream RTL dataset alone does not recreate measured historical inputs.

The builder preserves the historical construction logic:

- Validate required report/identity fields and finite nonnegative PPA labels. Zero labels remain allowed, matching the retained data; they were not removed for this release.
- Resolve source attribution using the retained catalog and match policy; duplicate source matches use the historical priority. This category is metadata attribution, not a claim of a unique original owner for every RTL string.
- Keep the higher-priority measured records when the same design name, exact RTL and normalized clock identify a duplicate. The recorded processing order gives the resynthesis pool precedence over successive external and augmentation pools.
- Associate generated candidates with their seed lineages. Preserve orphan augmentation with an explicit flag, following the original policy.
- Assign whole lineages to training or validation with seed 42 and target validation fraction 0.1, balancing the base and augmentation row pools within the recorded 0.005 tolerance.
- Write four training/validation shards, seven evaluation shards, source/lineage metadata and a generated manifest with counts and checksums. These are local output artifacts, not newly bundled public data.

Generated metadata can contain input locations and design identifiers. Treat output datasets/manifests as your own experiment artifacts and review their permissions separately before distributing them. The public examples contain generic locations only.

## Recovered source distribution

The table below was recomputed by streaming all retained DC/PT v6.3 JSONL files and verified against their recorded split counts, file checksums and aggregate source counts. It covers **519,243 rows** in `train`, `train_aug`, `valid` and `valid_aug`. A row is an RTL/clock measurement record; counts are not unique modules and are not the sizes of the public upstream datasets. Augmented rows inherit their seed-source category, while their RTL was generated locally.

| Retained source category | Base rows | Augmented rows | Training rows | Validation rows | Total | Share of train/validation |
|---|---:|---:|---:|---:|---:|---:|
| MG-Verilog | 19,728 | 0 | 17,970 | 1,758 | 19,728 | 3.80% |
| MetRex | 133,231 | 226,521 | 323,620 | 36,132 | 359,752 | 69.28% |
| OpenRTLSet | 52,993 | 0 | 47,492 | 5,501 | 52,993 | 10.21% |
| PyraNet-Verilog | 29,743 | 21,223 | 45,855 | 5,111 | 50,966 | 9.82% |
| RTLCoder | 2,760 | 0 | 2,532 | 228 | 2,760 | 0.53% |
| VeriLogos Augmented | 33,044 | 0 | 29,852 | 3,192 | 33,044 | 6.36% |
| **Total** | **271,499** | **247,744** | **467,321** | **51,922** | **519,243** | **100%** |

Training includes `train` plus `train_aug`; validation includes `valid` plus `valid_aug`. The source CSV also records each source's percentage within the training and validation pools. Four sources have zero augmentation rows in the retained files; these are observed counts, not estimates. Rounded percentages can differ slightly from 100% when summed.

Source attribution references: [MetRex](https://huggingface.co/datasets/scale-lab/MetRex), [PyraNet-Verilog](https://huggingface.co/datasets/bnadimi/PyraNet-Verilog), [VeriLogos](https://github.com/97kjmin/VeriLogos), [MG-Verilog](https://github.com/GATECH-EIC/mg-verilog), [OpenRTLSet](https://github.com/UIUC-ChenLab/OpenRTLSet), and [RTLCoder](https://github.com/hkust-zhiyao/RTL-Coder). These links identify the upstream projects; public upstream licenses and record-level rights remain separate from this code license. The aggregates do not distribute their RTL or grant permission to redistribute collected data.

The retained base pool contains **196,018 original/resynthesis rows** and **75,481 external-candidate rows**. The augmentation pool contains **247,744 rows**: 64,805 attributed to `gpt-5.4-nano` generation and 182,939 to `Qwen3-Coder-Next`. Recorded optimization-direction counts are 86,074 area, 85,968 delay and 75,702 power rows. These are generation metadata categories; they do not demonstrate functional equivalence or an optimization gain.

Source recovery was not uniformly an exact-RTL match. The recorded construction counters include **19,957 declared seed-source fallback mappings** for the PyraNet augmentation pool and **10,062 augmentation rows whose seed was absent from the retained base pool**. The latter were kept and flagged under the historical orphan policy. These counters limit how strongly the table can be interpreted as exact source/seed verification. The recovered builder retains that behavior; release preparation did not silently repair those historical lineage or attribution choices.

## Complete retained split sizes

| Retained split | Rows | Role / source attribution |
|---|---:|---|
| `train` | 244,349 | training base |
| `train_aug` | 222,972 | training augmentation |
| `valid` | 27,150 | validation base |
| `valid_aug` | 24,772 | validation augmentation |
| `test` | 810 | Evaluation; `local/test` metadata |
| `test_deepseek_ref` | 13,173 | REvolution/DeepSeek search cohort A; `local/test_deepseek_ref` metadata |
| `test_ds_verilog_eval` | 45,923 | REvolution/DeepSeek search cohort B; `local/test_ds_verilog_eval` metadata |
| `test_pluto` | 72 | Evaluation; `local/test_pluto` metadata |
| `test_ppa_feedback` | 14,206 | PPA-guided search candidates; `local/test_ppa_feedback` metadata |
| `test_realbench` | 516 | Evaluation; `local/test_realbench` metadata |
| `test_rtl_opt` | 288 | Evaluation; `local/test_rtl_opt` metadata |

The seven evaluation splits total **74,988 rows**; all eleven retained splits total **594,231 rows**. Evaluation records carry `local/test...` source labels. Their original upstream provenance is not established by those labels, so the source mixture above excludes evaluation splits. Split names alone do not establish independence from training or from each other.

The researcher's corrected account identifies `test_deepseek_ref` and
`test_ds_verilog_eval` as two cohorts from the same REvolution/DeepSeek search
source. A retained-data comparison confirms shared task families and exact RTL
strings; they are not independent generation sources. `test_ppa_feedback`
contains the researcher's historical PPA-guided search candidates collected
using an earlier trained estimator. Recovered run logs identify a `v6.1.4`
T5Gemma small proxy named for five epochs; they do not cryptographically bind
every retained candidate to its weights. The [cohort origin record](../results/v6.3/evaluation-cohort-origins.json)
adds this clarified attribution without rewriting the historical `local/test...`
fields or changing the training-source mixture or split counts.

Separately inspected generator references and measured selections identify
`test_rtl_opt` and `test_pluto` with [RTL-OPT and Pluto](BENCHMARK_REFERENCES.md).
They are selected local DC/PT remeasurements: 48 RTL variants at six targets for
RTL-OPT, and four designs with three variants at six targets for Pluto. This
additional benchmark attribution does not change the saved `local/test...`
metadata, recover missing upstream revision pins or establish complete benchmark
coverage, record-level licensing or independence.

### Recovered parent-provenance limit

Retained training and validation lineage metadata contains **42,602 training lineages and 4,732 validation lineages**, with **zero overlap**, totaling 47,334. This confirms only the saved lineage assignment. A later audit recovered the actual Pyra augmentation parents and found that **all 21,223 retained Pyra augmentation rows across 139 families had incorrect saved parent hashes**. The historical builder interpreted their sample identifiers against an unrelated legacy seed input.

Using the recovered true parents, **21 seed families cross training/validation
splits, involving 3,345 child measurement rows**. Zero overlap in incorrectly
assigned lineage hashes therefore does not establish a correct true-parent
partition or verified family independence. The effect on the recorded model
metrics is **unquantified**; this audit does not by itself identify the cause of
weak ranking or invalidate every evaluation result. The historical source
distribution and builder semantics above remain unchanged, and no silent dataset
repair or metric recomputation is included in this release.

Complete split sizes differ from the historical prediction evaluation denominators. The saved training evaluation used only 1,024 examples; validation prediction evaluation covered `valid`, not `valid_aug`. Failed decodes are also excluded from the historical accuracy denominator. See [v6.3 evaluation results](V63_RESULTS.md) for the corresponding metrics and zero-target caveats.

## Reproduction and release limits

The retained manifest records the construction settings and selected public source revisions, but the exact original dataset-generation code revision was not recorded. The released builder preserves the recovered algorithm and can process compatible user-supplied inputs; it cannot recover excluded historical payloads or guarantee identical historical data from current upstream downloads. A newly generated dataset is a new artifact until its content and provenance have been verified.

The data labels in the retained selected variant were produced by Design Compiler / PrimeTime. Public upstream source labels were not used as interchangeable DC/PT labels. Synthesis libraries, tools, constraints and their units must be supplied by the upstream measurement workflow; this consolidator does not choose a cell library or convert arbitrary label unit conventions.

Machine-readable public aggregates:

- [Data source distribution CSV](../results/v6.3/data-source-distribution.csv)
- [Source distribution JSON](../results/v6.3/source-distribution.json)
- [All retained split counts and lineage totals](../results/v6.3/split-counts.json)

Only aggregate counts, proportions and source/split labels are included. The full private manifest, source catalog, lineage JSONL, raw designs, labels and weights remain excluded.
