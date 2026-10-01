# Release validation

Preparation verified source privacy, package build/install, launcher help/import/dispatch and synthetic regression checks. These are engineering checks of released source, not new training or optimization experiments.

## Observed source checks

The release was checked in an isolated Linux environment using Python 3.13, CPU PyTorch 2.9.0, the declared PPA dependencies and `ntloss==0.3.1`. Dependency consistency checks passed. Package imports, seven supported entrypoint help checks, all six experimental entrypoint help checks, five installed EvolVE adapter help checks, and synthetic explicit-input dispatch passed from another directory.

The release-focused suite (`tests/public_ppa`, `tests/public_config`, `tests/evolve`, `tests/experiments`) recorded **112 passed, one skipped and 11 passed subtests**. This includes actual Number Token Loss composition, masked-label and gradient checks on small fabricated inputs, evaluator regression checks, optional checksums, explicit locations and missing external dependencies. The skipped check requires an external IC-RTL benchmark payload.

The full inherited suite recorded **245 passed, two skipped, one failed and 11 passed subtests**. The sole failure is `tests/revolution/test_algorithm.py::test_copy_misc_files_filters`: the preserved historical `_copy_misc_files` method is unimplemented. Its algorithm behavior was not changed during source organization. The second skipped check requires an external commercial synthesis flow and benchmark. The formerly missing synthetic fixtures and duplicate test-module collection problem were repaired.

The package preserves generic synthesis scripts as installed resources, and applicable license texts are included in the package distribution. Source line endings are normalized for Linux shell scripts. A fresh Git checkout is built and installed separately before preparation is delivered; detailed machine-specific verification records remain outside this public source tree.

## Aggregate-result verification

All nine public evaluation rows were checked against the saved aggregate CSV and its independently reconstructed summaries. Recorded relative errors agree within floating-point reconstruction tolerance; threshold percentages and decoded/failed counts agree. No per-design records were exported.

The training history contains exactly 292 training-loss entries and eight evaluation-loss entries. Only numeric step, epoch, training loss, evaluation loss and learning rate fields were exported. The published loss plots use these values without smoothing; plotted evaluation accuracy uses the public aggregate table. Both PNG plots were inspected visually for labels, legibility and clipping. SVG versions contain the same chart data.

## External and historical limitations

No training job, optimization search, commercial synthesis run or Docker build is claimed as part of source preparation. Public tests use synthetic inputs and mocks. Artifact-specific execution requires compatible datasets, trained checkpoints and external tools/libraries.

Historical integration checks successfully loaded the retained v6.3 model for small standalone prediction/evaluation samples. Those historical private-artifact checks do not make the source-only release independently reproduce the recorded results.

The remaining inherited test failure and skipped external-payload checks are not evidence of new training or search outcomes. The release verification records current source behavior without treating omitted payloads as shipped dependencies.
