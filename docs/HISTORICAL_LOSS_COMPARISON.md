# Historical CE and Number Token Loss comparison

This recovered comparison helps explain the researcher's decision to retain
cross-entropy (CE). At the **16-valid-draw setting**, CE+Number Token Loss (NTL)
has lower same-task Spearman rho and Kendall tau-b on two selected candidate
cohorts, despite some favorable overall correlations. These are descriptive
historical records, not a controlled causal ablation or a universally negative
finding about NTL. See the first-person [project findings](PROJECT_FINDINGS.md).

The selected comparison uses PPA-guided search candidates and REvolution/DeepSeek
search cohort B. The guiding PPA estimator that originally collected the former
is a separate role from the two CE/CE+NTL estimators compared below. Run logs
name a `v6.1.4` T5Gemma small proxy from a five-epoch run, without cryptographic
per-candidate checkpoint binding. The retrospective loss comparison uses later
`v6.2.1` models. Cohorts A and B share a REvolution/DeepSeek
source and some code, so split labels do not imply independent test sources.
See the [origin and overlap clarification](IN_GROUP_EVALUATION.md).

## Models and prediction procedure

The evaluator identifies two **earlier v6.2.1** T5Gemma models: CE and CE+NTL,
each named as ten epochs and checkpoint step **79,640**. These are not the final
CE model or the final training corpus. The historical loss labels are supplied
by the evaluator configuration and cache identities; there is no cryptographic
association between the cached predictions and checkpoint contents. The full
original training/generation invocation and runtime revision are unavailable.

The recovered generation implementation uses a 2,048-token input cap, 128-token
target cap, BF16 and scientific-notation precision four. It requests 64
stochastic generations, in chunks of eight, and caches **only valid decoded
draws**. Each curve point takes a component-wise median of the first available
`k` valid draws. Thus `k=16` means **up to 16 valid draws from 64 attempts**, not
16 total attempts. Four CE and three CE+NTL PPA-feedback examples have fewer
than 16 valid draws; none has zero. All selected cohort B examples have at
least 16. The curve points share prefixes and are not independent repeated runs.
Sampling temperature and top-p defaults are not explicitly recorded.

Both models have matching ordered candidate/clock identities and formatted
ground truths in these two cached cohorts. A compatible preserved matcher
reproduces the saved summaries. Its compatibility is not proof of the original
runtime Git revision. No inference, training or search was rerun.

## Recovered coefficients

Historical **overall** values first pool matched records within each clock and
then average defined clock coefficients equally. Historical **same-task** values
average defined task coefficients within each clock, then average clock means.
The minimum is ten usable records per task/clock. Constant inputs yield undefined
coefficients and are omitted. These conditional means have different coverage
and weighting. The historical overall procedure also differs from the final
model's pooled-across-all-rows table; the two tables are not a version ablation.
Historical correlations use the original synthesis-reference numeric labels;
formatted cache targets are used to check shared cohort identities. This is
another distinction from the final model's formatter-rounded saved targets.

| Historical cohort / metric | CE overall rho / tau-b | CE+NTL overall rho / tau-b | CE same-task rho / tau-b | CE+NTL same-task rho / tau-b |
|---|---:|---:|---:|---:|
| PPA-guided search candidates — area | 0.8935 / 0.7588 | 0.8671 / 0.7442 | 0.1340 / 0.1077 | 0.1210 / 0.0946 |
| PPA-guided search candidates — delay | 0.6719 / 0.5311 | 0.7393 / 0.5894 | 0.1728 / 0.1449 | 0.1015 / 0.0848 |
| PPA-guided search candidates — power | 0.8813 / 0.7381 | 0.8375 / 0.7001 | 0.0950 / 0.0774 | 0.0832 / 0.0639 |
| REvolution/DeepSeek search cohort B — area | 0.9717 / 0.9186 | 0.9795 / 0.9291 | 0.0125 / -0.0076 | -0.0966 / -0.0901 |
| REvolution/DeepSeek search cohort B — delay | 0.9513 / 0.8821 | 0.9426 / 0.8749 | 0.0526 / 0.0558 | -0.1692 / -0.1670 |
| REvolution/DeepSeek search cohort B — power | 0.9747 / 0.9155 | 0.9747 / 0.9180 | -0.0531 / -0.0535 | -0.1515 / -0.1550 |

Unrounded numeric values are in [summary.csv](../results/historical-loss-comparison/summary.csv)
and [summary.json](../results/historical-loss-comparison/summary.json).
[Per-clock coverage and coefficients](../results/historical-loss-comparison/per-clock.csv)
include the usable-group counts for each model and metric. Do not read rounded
cells as evidence of equal predictions or identical eligible groups.

## Selection, repetition and usable groups

Each proper cohort has **1,350 cached candidate/clock rows**, comprising 15
selected tasks with up to 15 candidates at each of six clocks. The selection
takes the first 15 qualifying tasks in sorted order, requiring at least ten
records at every clock. Candidate selection shuffles within groups with seed 42
and takes up to 15. This is a selected cohort, not the full test set.

| Cohort | Cached rows per model | Matched reference rows per model | Selected tasks | Grouped interpretation |
|---|---:|---:|---:|---|
| PPA-guided search candidates (`test_ppa_feedback`) | 1,350 | 1,428 | 15 | Repeated reference records reuse predictions; generally 10–14 usable task groups per metric/clock |
| REvolution/DeepSeek search cohort B (`test_ds_verilog_eval`) | 1,350 | 1,350 | 15 | Constant arrays leave only 2–5 usable task groups per metric/clock |
| Historical augmentation pool (`test_aug_100`) | 90 | 90 | invalid grouping | Grouped numerical scores withheld |

Usable group counts below sum task/clock groups across six clocks. Parentheses
give the minimum–maximum usable task groups within a clock. They measure
conditional coverage, rather than additional independent tasks.

| Cohort / metric | CE usable groups (per-clock range) | CE+NTL usable groups (per-clock range) |
|---|---:|---:|
| PPA-guided search candidates — area | 74 (12–14) | 74 (12–14) |
| PPA-guided search candidates — delay | 70 (10–13) | 71 (10–13) |
| PPA-guided search candidates — power | 78 (12–14) | 78 (12–14) |
| REvolution/DeepSeek search cohort B — area | 22 (3–4) | 22 (3–4) |
| REvolution/DeepSeek search cohort B — delay | 23 (3–5) | 23 (2–5) |
| REvolution/DeepSeek search cohort B — power | 24 (3–5) | 24 (3–5) |

The PPA-feedback match expands one task to 25–30 records per clock by reusing
predictions against repeated reference entries. These records do not represent
additional independently predicted candidates. Within-task coefficients can be
affected by repetition even though groups receive equal macro weight.

The original `n_samples` field counts the **entire reference pool**, including
unmatched records: 15,985 for PPA-feedback and 45,935 for cohort B. It is not
the evaluated denominator. The public export distinguishes reference rows,
cache rows, matched rows, usable rows and usable groups. Omitted constant groups
can differ between CE and CE+NTL. The [coverage/status record](../results/historical-loss-comparison/coverage-status.json)
keeps the cohort and failure limits explicit.

## Invalid historical augmentation grouping

Every one of the 2,519 historical augmentation reference modules has the problem
label `sample`. The 90 cached rows cover 65 distinct initial seed identifiers,
yet that label combines them into one nominal group per clock. The nearly
perfect nominal grouped coefficients equal the corresponding overall scores
because unrelated seeds were pooled together. They are not valid same-task
rank statistics and are excluded from the comparison table and public numerical
export. The original output and audit witnesses are preserved privately.

Hardcoded conclusions in an old report generator are also excluded. The public
comparison is checked against recorded numeric outputs and compatible replay,
with its limitations stated here. Intended task grouping is not a recovered
formal-equivalence proof for every candidate, and these coefficients establish
no final-model search result.
