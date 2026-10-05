# PPA Prediction for RTL Optimization: Findings and Open Problems

I started this project to make RTL optimization less dependent on expensive synthesis feedback. My goal was to predict power, delay and area (PPA) from Verilog and a clock constraint, then use those predictions to choose promising implementations during search. The practical question was whether a model could distinguish better and worse implementations of the same hardware task quickly enough to help optimization.

The main finding is a gap between two evaluation questions. The final model often has strong rank correlation across a mixed collection of designs, but its ordering of alternatives within the same task is weak. For example, the retained REvolution/DeepSeek search cohort B has the following standalone results:

| Metric | Pooled Spearman rho | Pooled Kendall tau-b | Same-task Spearman rho | Same-task Kendall tau-b | Usable task/clock groups |
|---|---:|---:|---:|---:|---:|
| Area | 0.9740 | 0.8995 | 0.1863 | 0.1523 | 633 |
| Delay | 0.9007 | 0.7893 | 0.1951 | 0.1703 | 599 |
| Total power | 0.9789 | 0.9019 | 0.1418 | 0.1128 | 648 |

The pooled columns include different tasks and clock points. The same-task columns average defined correlations within task/clock groups, with a minimum of ten usable records, then average groups and clocks equally. These are different summaries with different weighting and coverage. Their contrast shows why the pooled statistic alone cannot answer my optimization question. All values were reconstructed from retained predictions; no new inference or search was performed. The [complete ranking method](IN_GROUP_EVALUATION.md) records exclusions and sensitivity checks.

I call the selected retained DC/PT training corpus and its ten-epoch CE model the **final corpus** and **final model** below. Their historical identifier is **v6.3.0**. Only this final training-corpus version is retained in the closeout. Earlier loss and search experiments remain historical evidence, rather than additional released training datasets or controlled evidence of improvement between versions.

## 1. The comparison that optimization needs

A model can learn that a large datapath generally costs more than a small controller and still fail to choose between two implementations of one datapath. That broad distinction is useful for estimation, but an optimization loop repeatedly faces the narrower decision. It needs to know which candidate is better while the requested behavior, interface and operating conditions remain fixed.

This motivated my **Same-Task RTL Ranking Suite**. A group contains candidate implementations of one intended hardware task at one recorded clock target. Different task specifications and widths remain separate. The reader-facing names map to the retained identifiers as follows:

| Reader-facing cohort | Retained split identifier |
|---|---|
| REvolution/DeepSeek search cohort A | `test_deepseek_ref` |
| REvolution/DeepSeek search cohort B | `test_ds_verilog_eval` |
| PPA-guided search candidates | `test_ppa_feedback` |

I collected both REvolution/DeepSeek cohorts from DeepSeek search results in
REvolution. Their legacy names do not identify independent generation sources.
The retained cohorts share all 15 task families from cohort A and 764 exact RTL
source strings, although their code contents and records are not identical.
I keep their recorded evaluations separate while treating their overlap as a
limit on independent evidence.

The PPA-guided candidates come from one of my historical searches using an
earlier, stale trained PPA estimator. Recovered run logs name a proxy from a
five-epoch **v6.1.4** run based on `google/t5gemma-s-s-prefixlm`. They identify the
configured and loaded model, without a cryptographic binding to every retained
candidate. That estimator selected or scored search candidates; it is distinct
from the final CE model evaluated on the saved candidates in this report.
These are search-generated evaluation cohorts, shaped by their generation and
selection procedures, rather than interchangeable independent test samples.
The [cohort origin record](../results/v6.3/evaluation-cohort-origins.json) separates
my source recollection from the verified retained overlap.

These names describe the comparison, rather than the directory layout or generation machinery. Candidates are intended to implement the same behavior. The recovered group metadata does not establish formal equivalence for every candidate. Synthesis success, matching interfaces and an LLM's explanation of equivalence are insufficient substitutes for correctness verification.

I also need the same synthesis conditions: tool flow, cell library, timing assumptions and power assumptions. The historical grouping fixes the nominal clock, but some candidates use different full constraints, including real versus virtual clocks. A stricter sensitivity analysis removes those mixed-constraint groups and leaves the main conclusion of weak ranking essentially unchanged. That check addresses the identified differences; it does not certify all candidates or environments.

The evaluation goal informed augmentation. I wanted training examples with related RTL implementations and measured PPA variation, so that learning was exposed to the distinctions search would ask it to make. More unrelated designs alone might mainly improve recognition of broad cost scales. Training augmentation and evaluation candidates have separate roles, and their names or lineage hashes do not automatically establish independence. The retained data construction and later provenance audit place limits on that interpretation.

## 2. Why I synthesized six clock constraints

I measured designs at **1, 3, 5, 8, 10 and 15 ns**. My first reason was that synthesis behavior depends on the constraints. The same RTL can produce different implementation choices when timing is tightened or relaxed. I wanted the estimator to learn a conditional prediction problem rather than associate each source file with one supposedly universal PPA value.

My working understanding was that a moderately tight timing target could encourage useful PPA tradeoffs, while an impractical target could produce a different optimization outcome. This helped motivate the clock sweep. I do not have a controlled retained example demonstrating that a moderately tight clock always gives the best PPA, or that Design Compiler universally stops optimizing under infeasible constraints. I therefore present this as my synthesis intuition and design rationale, rather than an established rule of the tool.

My second reason was the intended search interface. I wanted to use the PPA model while optimizing under these clock targets. Measuring several targets increased the range of operating conditions represented in training and made it more likely that a design would have a feasible target among them. The six values cannot guarantee that every design meets timing, and I did not retain an ablation establishing that six is the optimal number or that this grid is optimal.

The measurements also change more than a single period field. The recovered constraint generator sets input/output delay using `min(1.0, round(0.6 * period, 2))`; designs without a detected clock receive a virtual clock. Power can vary with frequency and activity assumptions as well as implementation changes. I cannot attribute every observed power difference to an improved circuit structure.

A repeated measurement at another clock is another condition for the same RTL, not another independent implementation. This is why the grouped ranking fixes clock before comparing candidates. Allowing different clocks inside a group could reward recognition of a condition change while obscuring the ordering needed for one optimization decision.

## 3. Transformer regression inherited from regress-lm

The transformer regression method comes directly from [Google DeepMind's regress-lm](https://github.com/google-deepmind/regress-lm). I adapted its text-to-text numeric prediction approach to RTL/PPA inputs and evaluation. I do not claim a new transformer regression architecture. The imported upstream revision was not recorded, although the source origin, applicable notices and RTL/PPA modifications are documented.

The related literature already connects regression language models with ranking and optimization. [OmniPred: Language Models as Universal Regressors](https://arxiv.org/abs/2402.14547) reports Spearman and Kendall metrics. [Regression Language Models for Code](https://arxiv.org/abs/2509.26476v1) explicitly studies ranking candidate solutions to the same programming problem. My adaptation asks how this idea works for implementations of the same intended hardware behavior under fixed synthesis conditions. A strong coefficient in another domain does not establish that this RTL estimator makes useful optimization decisions.

The final workflow encodes RTL and clock context, then generates area, delay and total power as scientific-notation text. The selected backbone is T5Gemma, from the `google/t5gemma-l-l-prefixlm` family. Training uses sequence-to-sequence cross-entropy. Inference samples 16 output sequences, parses the valid numeric outputs and takes a component-wise median. The median area, delay and power can come from different sampled triples; the resulting vector need not be one jointly sampled output.

The final model completed **ten epochs and 146,040 optimizer steps**. Recovered settings include a learning rate of `1e-5`, per-device batch size four, gradient accumulation eight, BF16, seed 42 and a linear learning-rate schedule. Inputs are limited to **3,072 tokens** and targets to 128 tokens. The scientific-notation precision setting is four. The [training results](V63_RESULTS.md) preserve the settings and history, including 292 training-loss points and eight validation-loss points.

These choices impose practical limits. Truncation can remove relevant RTL or context. Numeric text generation requires valid syntax and adequate precision. Invalid outputs must be counted, and sampled decoding increases inference work. A lower synthesis cost is not sufficient if generation, inference or correctness checking consumes the saved time. The exact base-model revision and complete original invocation are missing, so the recovered settings do not guarantee exact training reproduction.

Training loss fell while the recorded validation loss rose. That validation-loss measurement uses a capped subset of up to 1,024 examples, separately from standalone prediction evaluation. The divergence warrants attention, but it does not identify overfitting, distribution shift or a data issue as the cause.

## 4. Why I retained CE after trying Number Token Loss

Cross-entropy treats target tokens as categorical labels. For a target token sequence, its basic objective is

\[
L_{\mathrm{CE}}=-\frac{1}{N}\sum_{t\in\mathcal{T}}\log p_\theta(y_t\mid x,y_{<t}),
\]

where \(\mathcal{T}\) contains supervised positions and \(N=|\mathcal{T}|\). Masked labels are excluded. This trains both number tokens and the syntax required to decode them. However, token identity does not itself express how numerically close two decoded answers are.

That motivated the Number Token Loss attempt. [Regress, Don't Guess](https://proceedings.mlr.press/v267/zausinger25a.html) introduces regression-like losses on numeric tokens. This is an independent paper, rather than a DeepMind follow-up. The recovered experimental code combines the objectives with defaults

\[
L=L_{\mathrm{CE}}+0.3L_{\mathrm{NT}}.
\]

The optional dependency is `ntloss==0.3.1`. The retained implementation enables digit-level processing and probability-mass reweighting, using the package's Wasserstein variant. Reweighting is a recovered package option, rather than a setting used in the original NTL paper experiments. CE still supervises syntax and other nonnumeric tokens. Both components honor masked labels. These are recovered implementation defaults; the complete original run arguments are unavailable.

Numeric-token distance is still different from comparing complete PPA values or ordering two implementations. In scientific notation, an exponent error and a mantissa-digit error can have very different consequences. A per-digit objective also does not explicitly ask whether one same-task candidate should rank above another. This is a plausible explanation for an objective mismatch, rather than a demonstrated cause of the observed results.

I retained CE because the NTL experiments did not provide the grouped Spearman and Kendall performance I needed. A recovered historical comparison supports this choice on two selected candidate cohorts. It concerns **earlier v6.2.1 models**, each identified as a ten-epoch checkpoint at step 79,640; it is not a comparison involving the final model:

| Earlier cohort / metric | CE same-task rho / tau-b | CE+NTL same-task rho / tau-b |
|---|---:|---:|
| PPA-guided search candidates — area | 0.1340 / 0.1077 | 0.1210 / 0.0946 |
| PPA-guided search candidates — delay | 0.1728 / 0.1449 | 0.1015 / 0.0848 |
| PPA-guided search candidates — power | 0.0950 / 0.0774 | 0.0832 / 0.0639 |
| REvolution/DeepSeek search cohort B — area | 0.0125 / -0.0076 | -0.0966 / -0.0901 |
| REvolution/DeepSeek search cohort B — delay | 0.0526 / 0.0558 | -0.1692 / -0.1670 |
| REvolution/DeepSeek search cohort B — power | -0.0531 / -0.0535 | -0.1515 / -0.1550 |

Both models were evaluated on matching selected candidate/target identities. The reported setting uses up to the first 16 **valid cached draws from 64 generation attempts**, rather than 16 total attempts. It selects 15 tasks and up to 15 candidates per task/clock. Repeated reference records and omitted constant groups affect the averages; cohort B has only two to five usable groups per metric/clock. The [historical comparison](HISTORICAL_LOSS_COMPARISON.md) supplies coverage and aggregation details.

Some historical overall correlations favor NTL, while these within-task correlations favor CE. The result is descriptive evidence for my decision, without establishing a universally inferior loss or a controlled causal ablation. The final model is CE-trained and must not be described as NTL-trained.

## 5. How I understand Spearman and Kendall tau-b

Spearman rho measures association after replacing values with ranks:

\[
\rho_s=\operatorname{corr}(\operatorname{rank}(y),\operatorname{rank}(\hat y)).
\]

I understand it as asking whether a model assigns higher values to implementations with higher measured values. It is insensitive to strictly increasing changes of scale. This is useful when ordering matters more than calibration, but a good rho does not mean the predicted costs or constraints are accurate.

Kendall's coefficient considers pairs. A pair is concordant when prediction and measurement agree on its ordering, and discordant when they disagree. With \(C\) concordant pairs, \(D\) discordant pairs, and \(T_y,T_{\hat y}\) counting pairs tied only in the respective variable, the tie-adjusted form is

\[
\tau_b=\frac{C-D}{\sqrt{(C+D+T_y)(C+D+T_{\hat y})}}.
\]

This connects naturally to choosing between two candidates. However, tau-b is not generally the probability of selecting the better candidate, especially with ties. Neither coefficient measures the magnitude of a mistaken choice, the regret of selecting a candidate, or whether the best few candidates were found. The definitions follow [SciPy's Spearman documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html) and [Kendall documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html).

A synthetic example makes the distinction concrete. Suppose three comparable candidates have measured costs `[100, 101, 102]`, while predictions are `[102, 101, 100]`. Every relative error is below 20%, yet both rank coefficients are -1: the ordering is reversed. These are invented numbers, not project PPA measurements. Conversely, perfectly ordered predictions can still have large absolute errors. Optimization needs ordering, feasibility and cost accuracy to be considered together.

Across unrelated designs, many correctly ordered broad cost differences can dominate a pooled coefficient. Within one task, differences may be much smaller and depend on implementation details. My grouped evaluation asks this narrower question directly. It computes each PPA metric separately, averages defined groups equally within each clock, then averages defined clocks equally. A large family does not outweigh a small eligible family simply by supplying more rows.

Ties, constant arrays and missing decodes matter. Formatter precision can create ties. A constant prediction cannot usefully order a group, although its correlation is undefined rather than zero. Undefined and undersized groups are omitted and counted, so the mean must be read with its coverage. Failed decodes are absent from saved prediction CSVs and reported separately. Repeated measurements are retained records, not independent new implementations. No independence-based significance claims are made.

On the three candidate-rich final cohorts, headline same-task rho values range approximately from **0.12 to 0.27**, and tau-b from **0.10 to 0.21**. Selected local [RTL-OPT](https://arxiv.org/abs/2601.01765) and [Pluto](https://arxiv.org/abs/2510.14756) remeasurements add smaller groups. Pluto supplies three alternatives per task/clock; recovered RTL-OPT pairs supply two. Their smaller cutoffs are reported separately. A two-element nonconstant group produces only +1 or -1, expressing ordering direction rather than a rich ranking.

The provenance review also found an invalid historical grouping. In the earlier `test_aug_100` analysis, unrelated seeds were combined under the literal problem name `sample`. Its nearly perfect nominal grouped scores cannot support a same-task finding and are excluded from valid comparison tables. An attractive number is uninformative if the group definition fails to represent the decision of interest.

## 6. What remains unresolved, and what I am releasing

The final standalone evaluation has useful signal and substantial limits. On the main test, mean metric accuracy below 20% relative error is **80.45%**, while **59.38%** of decoded examples satisfy that threshold for all three metrics. The local RTL-OPT subset has **61.44% / 41.11%**, Pluto **50.93% / 26.39%**, and RealBench **31.26% / 14.17%**, respectively. These are this project's selected measurements, not the benchmark papers' results. The [nine-set results](V63_RESULTS.md) retain all per-metric summaries and failures.

The saved training prediction evaluation covers only 1,024 examples. Historical accuracy denominators exclude failed decodes. Zero targets can dominate mean relative error because its denominator is `abs(target) + 1e-10`; their origin remains unresolved. These limitations prevent me from reducing model quality to one accuracy number.

Provenance also limits the training/validation interpretation. The recovered Pyra augmentation audit found incorrect historical parent hashes for **21,223 measurement rows across 139 families**. Actual parent/child relationships cross training and validation in **21 families involving 3,345 child rows**. Disjoint saved lineage hashes therefore do not establish a correct true-parent partition. The effect on model metrics has not been quantified, and the retained corpus was not repaired. This issue does not establish the cause of weak ranking on separately reconstructed benchmark groups.

I see several open problems. First, the training objective needs better alignment with candidate selection, while preserving numerical accuracy. Pairwise or listwise objectives may help, but the preserved ranking and latent-loss implementations do not establish that they succeed. Second, prediction must represent relevant constraints and handle feasibility; nominal clock text alone may be insufficient. Third, uncertainty should help allocate reference measurements where ordering is ambiguous. Sample variation needs calibration before it can serve that purpose.

Correctness is another essential condition. A cheaper candidate with changed behavior is not an optimization success. Group metadata, successful synthesis and recorded simulation checks have different evidential strength. Future evaluation would need explicit correctness coverage alongside rank quality, usable-group coverage, top-candidate selection and measured selection regret.

The remaining end-to-end question requires controlled search: comparable generation and verification settings, fixed budgets, repeated runs and reference-flow measurement of selected candidates, with time spent on inference and tools recorded. **No retained evidence establishes that the final model was evaluated inside optimization search.** Historical searches used earlier models and had mixed outcomes. I cannot treat standalone accuracy, high pooled correlation or weak grouped correlation as a measured final search result.

I am closing the project with an unmet demonstrated optimization objective and reusable code and evidence. The full final PPA corpus and model remain private; generated labels contain information tied to commercial synthesis/signoff flows and cell libraries. The public repository provides code, synthetic fixtures and aggregate findings. A separately reviewed Verilog-only companion contains 41 exact RTL-OPT source files with their applicable notice. It is not the training corpus and cannot reproduce the model by itself.

My main lesson is to evaluate the decision the system is intended to support. In this project, that means ordering valid alternatives for the same hardware task under the same conditions, then checking whether those decisions improve measured optimization outcomes. The retained results establish a gap in that chain. They do not establish that learned PPA estimation is inherently ineffective.
