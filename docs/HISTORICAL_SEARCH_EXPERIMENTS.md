# Historical search experiments and the time bottleneck

These are this project's historical attempts, not the upstream papers' reported results. They use earlier estimators, not the final v6.3.0 CE model. No training, search, synthesis or inference was rerun for this reconstruction.

## What the two algorithms establish

| Implementation | Retained experiment | Estimator | Evidence boundary |
|---|---|---|---|
| Population search adapted from [REvolution](https://github.com/kmcho2019/REvolution) ([paper](https://arxiv.org/abs/2510.21407)) | 15 proxy-guided task summaries; 20 generations and 210 candidate records per task | v6.1.4; checkpoint name indicates five epochs; configured/loaded identity recovered from runtime logs | No matched DC/PT-guided arm recovered; saved proxy scores cannot establish measured search superiority |
| Local [EvolVE](https://arxiv.org/abs/2601.18067)-inspired implementation | 18 tasks with both model and DC/PT arms | DC/PT v6.2.1, ten epochs, checkpoint step 79,640; loads verified per task | Descriptive comparison of reference-measured finalists; one recorded attempt per arm, without repeated-trial uncertainty |

REvolution generated and selected alternatives in a proxy campaign with 3,150 candidate records across 15 saved task summaries. The historical `synthesis_ppa` success field can refer to proxy evaluation; it does not establish successful DC/PT measurement. Earlier failed launch summaries also exist. The retained task summaries are a coverage record, not a ground-truth efficacy table. Later REvolution DC/PT runs have different generation settings and cannot serve as a matched control. My assessment is that the estimator did not provide effective enough search guidance; the recovered REvolution records do not independently quantify that assessment as a head-to-head result.

## EvolVE comparison setup

Both arms used the same converted task inputs, shared generated roots, 12 chains with five nodes each (60 nodes per arm), three finalists, and clock targets of 1, 3, 5, 8, 10 and 15 ns. Generation used the recorded `gpt-5.6-luna` identifier with `max` reasoning. The first five completed tasks come from an initial four-pipeline campaign; the remaining 13 come from its trained-model/DC/PT follow-up. These 18 tasks are unique. Other arms, incomplete initial tasks, smoke tests and retries are not additional paired observations. This is recovered completed coverage, not a random task sample.

The model arm used predicted PPA during search; the DC/PT arm used reference-flow measurements. Saved finalist records subsequently measure selected candidates with DC/PT. We compare the best recorded golden finalist in each arm, not its predicted score. For this VerilogEval adapter, golden PPA is measured area × delay × fixed-activity power; VCD power is not used in these comparison scores. The golden score is its negative product, so a smaller score magnitude represents lower measured PPA. All 18 pairs select the 15 ns point. Matching nominal clocks does not prove functional equivalence or identical physical delay. Recorded simulation checks are not universal formal-equivalence guarantees. Run-time source Git revisions were not recorded.

Node and non-null-score counts, checkpoint loads and search times were checked against read-only run databases. Golden best scores, periods and baseline ratios were independently checked against saved finalists. A valid node means a non-null search score under that arm's evaluator, not guaranteed reference feasibility. Complete finalist coverage retains 427 successful clock records, 214 timing failures and seven failures (648 in total). Failure records were not converted to successes.

## Measured search outcomes

The model arm has worse golden PPA on **14 of 18 tasks**, ties on **three**, and is slightly better on **one**. For `Prob082_lfsr32`, its PPA is about 0.203% lower than the DC/PT arm. These observations support a failure to obtain a consistent optimization advantage; they do not establish that a model can never outperform DC/PT guidance.

Baseline reduction is `100 × (1 − candidate PPA / baseline PPA)` using the saved comparable baseline. Negative values mean worsening. The model/DCPT ratio directly compares measured finalists: above one is worse for the model. Percentages from different tasks are not raw PPA magnitudes.

| Task | Model valid / 60 | DC/PT valid / 60 | Model baseline reduction (%) | DC/PT baseline reduction (%) | Model / DCPT golden PPA | Model outcome |
|---|---:|---:|---:|---:|---:|---|
| Prob018_mux256to1 | 59 | 60 | -17.571 | 0.000 | 1.175714 | worse |
| Prob021_mux256to1v | 60 | 59 | -32.413 | 39.708 | 2.196191 | worse |
| Prob023_vector100r | 60 | 57 | 0.000 | 0.000 | 1.000000 | tie |
| Prob030_popcount255 | 60 | 57 | 0.059 | 3.668 | 1.037461 | worse |
| Prob052_gates100 | 59 | 60 | 74.781 | 84.687 | 1.646938 | worse |
| Prob055_conditional | 60 | 60 | 28.339 | 41.557 | 1.226169 | worse |
| Prob068_countbcd | 59 | 59 | 0.124 | 55.170 | 2.227895 | worse |
| Prob080_timer | 60 | 60 | -8.386 | 25.163 | 1.448302 | worse |
| Prob082_lfsr32 | 60 | 60 | 3.051 | 2.854 | 0.997975 | better |
| Prob092_gatesv100 | 59 | 59 | 0.000 | 0.000 | 1.000000 | tie |
| Prob097_mux9to1v | 60 | 59 | 35.286 | 38.287 | 1.048644 | worse |
| Prob105_rotate100 | 59 | 59 | 17.330 | 18.566 | 1.015187 | worse |
| Prob108_rule90 | 48 | 60 | 0.000 | 0.000 | 1.000000 | tie |
| Prob115_shift18 | 59 | 59 | -19.293 | 0.000 | 1.192934 | worse |
| Prob118_history_shift | 59 | 60 | 0.000 | 33.425 | 1.502055 | worse |
| Prob123_bugs_addsubz | 60 | 58 | 48.763 | 52.753 | 1.084450 | worse |
| Prob124_rule110 | 47 | 60 | 32.362 | 33.154 | 1.011847 | worse |
| Prob153_gshare | 58 | 60 | 6.050 | 34.106 | 1.425784 | worse |

For popcount255, the model yields 0.059% baseline reduction versus 3.668% with DC/PT guidance. The timer worsens by 8.386% with the model versus a 25.163% improvement with DC/PT. Gates100 improves substantially in both arms, but DC/PT achieves 84.687% reduction versus 74.781% for the model. These are recorded outcomes, not causal estimates of changing only the scoring signal: generated refinements differ and repeated runs were not retained.

The model arm's saved cold-search time is shorter on 15 tasks and longer on three. This is pipeline-search duration plus model loading; it excludes shared idea/root generation, preflight and final golden validation. It is not complete end-to-end time. Faster search therefore does not establish better PPA or a proportional full-pipeline speedup.

## Where the recorded time goes

I found LLM response and candidate generation to be the principal time bottleneck in the current pipeline. Across the 18 pairs, generation accounts for about **99.1%** of selected event-duration totals in the model arm and **82.4%** in the DC/PT arm. This category comprises idea generation, root generation and iterative refinement, including the full recorded LLM invocation duration. It does not separate model compute, service queueing, network delay or client overhead.

| Arm | Generation (s) | Simulation (s) | Synthesis or proxy (s) | Other events (s) | Generation share |
|---|---:|---:|---:|---:|---:|
| trained_model | 228813.972 | 656.395 | 1388.840 | 69.520 | 99.08% |
| dc_pt | 242811.914 | 653.365 | 51161.181 | 10.453 | 82.41% |

Wrapper events are excluded to avoid counting an operation together with its enclosing timer. Shared idea/root generation is included once in each arm's accounting; the two rows must not be added as independent total work. Durations can overlap under concurrent calls. These shares describe selected recorded event time, not a wall-clock critical-path decomposition, and omit events outside the selected categories. Failed timed attempts remain included.

An additional EvolVE timing example using Luna max records 72 generation calls totaling 30,383.970 seconds (421.9996 seconds per call on average), versus 393.593 seconds for VCS simulation and 1,172.085 seconds for synthesis/proxy events. Generation is 95.10% of that chart's selected event time. Root-generation calls average 686.816 seconds; refinement calls average 366.891 seconds. This example uses a Yosys/OpenROAD arm and is timing evidence, not a model-versus-DC/PT quality comparison. The chart totals were verified against its timing CSV.

REvolution summaries record overall/generation time and call counts, rather than an independently verified LLM-response-only decomposition. EvolVE percentages are not transferred to REvolution. Long-tail synthesis failures can also be expensive. The practical lesson is to address generation latency and scheduling alongside scoring cost: an estimator cannot remove time spent waiting for RTL proposals, and it must preserve useful selection quality.

## Public records and replay limits

[Sanitized aggregate records](../results/historical-search/README.md) contain task-level outcomes, node/failure coverage, model identifiers and timing totals. They contain no candidate RTL, predictions, proprietary libraries, workstation paths, private links or model weights. Source hashes and detailed provenance remain private. Older weights are excluded from the final closeout package; changing API models and external EDA dependencies also limit replay. These experiments establish neither final v6.3 search performance nor a general impossibility result for learned PPA estimation.
