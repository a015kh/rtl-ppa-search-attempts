# Release validation

Preparation verified source privacy, package build/install, launcher help/import/dispatch and synthetic regression checks. These are engineering checks of released source, not new training or optimization experiments.

## Observed source checks

The release was checked in an isolated Linux environment using Python 3.13, CPU PyTorch 2.9.0, the declared PPA dependencies and `ntloss==0.3.1`. Dependency consistency checks passed. Package imports, eight supported launcher help checks and the rank-plot utility help check passed from another directory. Earlier checks also passed all six experimental entrypoint help checks, five installed EvolVE adapter help checks and synthetic explicit-input dispatch; those workflows are unchanged by the rank-statistics extension.

The current full suite recorded **364 passed, two skipped, zero failed and 31
passed subtests**. It includes actual Number Token Loss composition, masked-label
and gradient checks on fabricated inputs, unchanged evaluator behavior, optional
checksums, explicit locations and missing external dependencies. The skipped
checks require external IC-RTL benchmark payloads or a commercial synthesis flow
and benchmark.

The earlier release check exposed the functional failure
`tests/revolution/test_algorithm.py::test_copy_misc_files_filters`. The auxiliary
copy loop was unreachable. This release restores flat, problem-specific non-RTL
file staging before candidate simulation, with binary-data, reserved-file,
directory-boundary and symlink regression checks for both engine save paths.
The formerly failing test now passes. Scoring, correctness gates and model losses
remain unchanged. See [the auxiliary-file behavior and limits](SEARCH_AUXILIARY_FILES.md).
No claim is made about which historical searches were affected.

The dataset-generation/preparation suite contains **61 tests** using fabricated
measured rows and a local source catalog. They cover offline assembly, schema and
label preservation, deduplication priority, whole-lineage splits, checksums,
network opt-in, relative configuration paths, protected inputs/outputs, the three
named test-result policies and a runnable walkthrough. A comparison with the
preserved builder produced six byte-identical output files for 18 synthetic
records. All 17 recovered report/conversion functions match their retained source
ASTs. Synthetic report and resynthesis comparisons each reproduced two accepted
and four rejected points, with matching emitted rows and rejection counts.

The [walkthrough](SYNTHETIC_WALKTHROUGH.md) converts two fabricated test results,
assembles an 18-row dataset and verifies its labels, checksums, deduplication and
zero training/validation lineage overlap. It succeeds with isolated Python
(`-I -S`) from another directory using only the standard library. Eight canonical
launcher help checks and the rank-plot utility help check passed from another directory. No retained dataset was
regenerated and no learned inference or synthesis is part of this walkthrough.

The package preserves generic synthesis scripts as installed resources, and applicable license texts are included in the package distribution. Source line endings are normalized for Linux shell scripts. A fresh Git checkout is built and installed separately before preparation is delivered; detailed machine-specific verification records remain outside this public source tree.

## Aggregate-result verification

All nine public evaluation rows were checked against the saved aggregate CSV and its independently reconstructed summaries. Recorded relative errors agree within floating-point reconstruction tolerance; threshold percentages and decoded/failed counts agree. No per-design records were exported.

The dataset source-distribution tables were recomputed from all eleven retained files and checked against recorded checksums and counts. The combined training/validation pool contains 519,243 rows across six source categories, and the seven evaluation files contain 74,988 rows. Public exports include aggregate counts and proportions; source catalogs, lineage payloads and raw records remain excluded. Detailed provenance limits accompany the tables.

The training history contains exactly 292 training-loss entries and eight evaluation-loss entries. Only numeric step, epoch, training loss, evaluation loss and learning rate fields were exported. The published loss plots use these values without smoothing; plotted evaluation accuracy uses the public aggregate table. All three PNG figures, including the new rank-correlation chart, were inspected visually for labels, legibility and clipping. SVG versions contain the same chart data.

The post-hoc rank extension adds **19 synthetic regression tests** covering exact
ties, zero targets, paired nonfinite exclusions, constant/empty inputs, failed
decodes, sampled medians, JSON nulls, unchanged historical error/accuracy values
and aggregate-only launcher behavior. All 54 coefficients across nine splits
and three metrics agree with an independent calculation and the release's SciPy
implementation within floating-point precision. Source and installed-alias checks
use exact saved numeric strings privately; no model is loaded. Public CSV/JSON
counts and coefficients agree, all 54 four-decimal documentation values match,
and all 54 SVG annotations and bar values match the aggregate table at the
documented rounding/serialization precision. Existing public accuracy/history
files remain byte-identical. Only rank aggregates and their plots were added.

## External and historical limitations

No training job, optimization search, commercial synthesis run or Docker build is claimed as part of source preparation. Public tests use synthetic inputs and mocks. Artifact-specific execution requires compatible datasets, trained checkpoints and external tools/libraries.

Historical integration checks successfully loaded the retained v6.3 model for small standalone prediction/evaluation samples. Those historical private-artifact checks do not make the source-only release independently reproduce the recorded results.

The external-payload skips are not evidence of new training or search outcomes.
The release verification records current source behavior without treating omitted
payloads as shipped dependencies. Test-result conversion/filtering is available;
historical simulation-based training-pool overlap removal remains excluded, and
exact reproduction of the seven retained v6.3 test sets is not established.
