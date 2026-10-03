# Release validation

Preparation verified source privacy, package build/install, launcher help/import/dispatch and synthetic regression checks. These are engineering checks of released source, not new training or optimization experiments.

## Observed source checks

The release was checked in an isolated Linux environment using Python 3.13, CPU PyTorch 2.9.0, the declared PPA dependencies and `ntloss==0.3.1`. Dependency consistency checks passed. Package imports, nine earlier supported launcher help checks, the Verilog-only launcher help check and both rank-plot utility help checks passed from another directory. Earlier checks also passed all six experimental entrypoint help checks, five installed EvolVE adapter help checks and synthetic explicit-input dispatch; those workflows are unchanged by the result and source-only preparation extensions.

The current full suite recorded **413 passed, two skipped, zero failed and 75
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
(`-I -S`) from another directory using only the standard library. Nine canonical
launcher help checks and both rank-plot utility help checks passed from another directory. No retained dataset was
regenerated and no learned inference or synthesis is part of this walkthrough.

The package preserves generic synthesis scripts as installed resources, and applicable license texts are included in the package distribution. Source line endings are normalized for Linux shell scripts. A fresh Git checkout is built and installed separately before preparation is delivered; detailed machine-specific verification records remain outside this public source tree.

## Aggregate-result verification

All nine public evaluation rows were checked against the saved aggregate CSV and its independently reconstructed summaries. Recorded relative errors agree within floating-point reconstruction tolerance; threshold percentages and decoded/failed counts agree. No per-design records were exported.

The dataset source-distribution tables were recomputed from all eleven retained files and checked against recorded checksums and counts. The combined training/validation pool contains 519,243 rows across six source categories, and the seven evaluation files contain 74,988 rows. Public exports include aggregate counts and proportions; source catalogs, lineage payloads and raw records remain excluded. Detailed provenance limits accompany the tables.

The training history contains exactly 292 training-loss entries and eight evaluation-loss entries. Only numeric step, epoch, training loss, evaluation loss and learning rate fields were exported. The published loss plots use these values without smoothing; plotted evaluation accuracy uses the public aggregate table. All five PNG figures, including the pooled chart and two in-group charts, were inspected visually for labels, legibility and clipping. SVG versions contain the same chart data.

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
files remain byte-identical.

The [in-group extension](IN_GROUP_EVALUATION.md) adds **20 synthetic regression
tests**. They check exact ties, constant and undersized groups, zeros, nonfinite
pairs, fixed-clock partitioning, equal-group/equal-clock weighting, required
explicit membership, aggregate-only outputs and execution from another
directory. A synthetic case has a pooled Spearman coefficient above 0.99 while
every within-group ordering is reversed, guarding against using pooled
correlations as the optimization-ranking result.

All **63 reconstructed split/metric/cutoff summaries** and **405 numeric-clock
rows** agree with an independent calculation. The public summary also includes
18 explicitly unverified train/validation records, giving 81 rows. Public
CSV/JSON records agree; nine strict-constraint sensitivity records reproduce the
documented whole-group exclusions. Numeric strings are preserved from the saved
records in private verification inputs. Source and installed grouped-summary
checks use those inputs without loading a model. The 96 displayed coefficient
cells (including N/A values), usable-group counts and all 48 numeric annotations
and bar values in the two new SVG figures match the aggregate tables at the
documented rounding/serialization precision. Existing aggregate files remain
byte-identical. Group reconstruction coverage, repeated records, mixed nominal
clock constraints and limits on replay are reported alongside the results;
these engineering checks do not prove functional equivalence or establish a
v6.3 search experiment.

## Verilog-only preparation checks

The [companion exporter](VERILOG_ONLY_RELEASE.md) adds **29 synthetic tests**
covering exact source/comment/line-ending preservation, global deduplication,
clock-repeat counts, explicit source and kind policies, optional checksums,
protected outputs, withholding of unsafe text, privacy of manifests, output
schema/tamper checks and source-rights status. Unknown sources default to
withholding; unverified rights produce staging-only packages. URLs with embedded
credentials, direct IP addresses or malformed ports are rejected. Corrupted
category distributions and malformed status values are also rejected.

Source and installed preparation commands are checked from another directory
with three fabricated measured rows containing private tool-report/PPA fields.
They produce exactly two unchanged Verilog strings and only the five whitelisted
dataset columns. Preflight, build and independent package verification succeed.
Source dispatch uses isolated Python (`-I -S`) and needs no third-party package.
Literal scanning preserves exact case-folded substring semantics while a bounded
hash cache avoids repeated content scans of identical RTL; per-row policies and
withholding counts remain unchanged. No collected RTL is bundled in Git or the
package distribution, and the exporter does not certify record-level rights.

## External and historical limitations

No training job, optimization search, commercial synthesis run or Docker build is claimed as part of source preparation. Public tests use synthetic inputs and mocks. Artifact-specific execution requires compatible datasets, trained checkpoints and external tools/libraries.

Historical integration checks successfully loaded the retained v6.3 model for small standalone prediction/evaluation samples. Those historical private-artifact checks do not make the source-only release independently reproduce the recorded results.

The external-payload skips are not evidence of new training or search outcomes.
The release verification records current source behavior without treating omitted
payloads as shipped dependencies. Test-result conversion/filtering is available;
historical simulation-based training-pool overlap removal remains excluded, and
exact reproduction of the seven retained v6.3 test sets is not established.
