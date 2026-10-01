# Recorded DC/PT v6.3 training and evaluation

These are recorded standalone results for the Design Compiler / PrimeTime v6.3.0 PPA model. They are published as aggregate evidence. Datasets, design-level predictions and checkpoints are not included. No retained evidence establishes that this v6.3 model was evaluated inside optimization search. The measurements do not demonstrate a search improvement or a controlled improvement over earlier model versions.

## Training provenance

The selected T5Gemma large encoder/decoder model completed **10 epochs and 146,040 optimizer steps**. The base model family was `google/t5gemma-l-l-prefixlm`; its exact revision and the complete original launch command were not recorded. The imported regress-lm source revision was also not recorded. These gaps limit exact reproduction.

| Recovered setting | Value |
|---|---:|
| Initial learning rate | 0.00001 |
| Training batch size per device | 4 |
| Evaluation batch size per device | 4 |
| Gradient accumulation steps | 8 |
| Weight decay | 0.01 |
| Seed | 42 |
| Precision | BF16 |
| Optimizer | AdamW fused (PyTorch) |
| Learning-rate schedule | Linear |
| Evaluation interval | 20,000 steps |
| Save interval | 2,000 steps |
| Maximum input length | 3,072 tokens |
| Maximum target length | 128 tokens |
| Prediction sampling | 16 draws, component-wise median of valid draws |
| Scientific-notation precision setting | 4 |

The saved training arguments support the optimization settings above. The retained formatter and compatible launch profile supply the formatting/sampling settings; the full original invocation is unavailable. The retained training implementation uses a 0.1 warmup ratio and caps its loss-evaluation dataset at **1,024 validation examples**. This training-time loss measurement is different from the standalone prediction evaluation on full saved validation splits. The records do not prove that the published checkpoint used Number Token Loss; it must not be attributed to that experimental objective.

## Recorded loss curves

![Training and validation losses](../figures/v6.3-training-loss.png)

The public history contains **292 training-loss points and 8 evaluation-loss points**, including the final evaluation at step 146,040. It preserves only step, epoch, training loss, evaluation loss and learning rate from each recorded log entry. Missing fields remain blank in CSV and absent in JSON. Points have not been smoothed or resampled. Training loss and capped validation loss are shown on separate axes because they represent different recorded measurements. Their divergence motivates further examination; it does not isolate a cause such as overfitting or data shift.

## Standalone prediction evaluation

| Split | Decoded | Failed | Mean relative error | Mean metric accuracy <20% | All metrics <20% |
|---|---:|---:|---:|---:|---:|
| `train` | 1,024 | 0 | 0.00850522 | 99.41% | 98.73% |
| `valid` | 27,143 | 7 | 6.02803e+07 | 75.96% | 60.82% |
| `test` | 810 | 0 | 0.122938 | 80.45% | 59.38% |
| `test_deepseek_ref` | 13,172 | 1 | 0.160337 | 76.90% | 59.70% |
| `test_ds_verilog_eval` | 45,911 | 12 | 0.161441 | 75.22% | 54.60% |
| `test_pluto` | 72 | 0 | 0.269014 | 50.93% | 26.39% |
| `test_ppa_feedback` | 14,206 | 0 | 0.258567 | 71.49% | 50.70% |
| `test_realbench` | 515 | 1 | 4.80037e+09 | 31.26% | 14.17% |
| `test_rtl_opt` | 287 | 1 | 0.274675 | 61.44% | 41.11% |

| Split | Area <20% | Delay <20% | Total power <20% |
|---|---:|---:|---:|
| `train` | 99.71% | 99.51% | 99.02% |
| `valid` | 72.93% | 85.95% | 68.99% |
| `test` | 80.37% | 88.40% | 72.59% |
| `test_deepseek_ref` | 73.00% | 86.52% | 71.17% |
| `test_ds_verilog_eval` | 72.29% | 86.02% | 67.36% |
| `test_pluto` | 33.33% | 69.44% | 50.00% |
| `test_ppa_feedback` | 68.40% | 82.16% | 63.90% |
| `test_realbench` | 23.30% | 47.38% | 23.11% |
| `test_rtl_opt` | 48.43% | 81.53% | 54.36% |

![Evaluation threshold accuracies](../figures/v6.3-evaluation-accuracy.png)

The machine-readable summaries contain all saved per-metric, mean-metric and joint accuracies at **5%, 10% and 20%** relative-error thresholds, mean relative error, successfully decoded counts and failed-decode counts. The aggregate `total_samples` field is the successfully decoded denominator. `attempted_samples` is the sum of decoded and failed rows. The results were checked against reconstructed saved summaries. Split names are retained as labels; names alone do not establish family independence or absence of overlap.

The `train` prediction evaluation is limited to **1,024 examples**, so its accuracy is not an estimate for the full training dataset. Failed decodes are listed separately and excluded from the historical accuracy denominator. The mean metric accuracy averages area, delay and total-power accuracies; joint accuracy requires all three predictions in an example to pass the threshold.

The historical relative-error formula is `abs(prediction - target) / (abs(target) + 1e-10)`. Threshold tests use a strict `<`. Zero targets can dominate the mean: the saved validation results include 336 zero area labels, 585 zero delay labels and 396 zero total-power labels among decoded examples; RealBench has 9 zero labels for each metric. Their recorded mean relative errors (approximately 6.03e7 and 4.80e9) are not typical percentage errors. The available evidence does not establish whether the zero labels are valid cases or data-quality problems. They were not relabeled for this release.

## Public result files

- [Training settings](../results/v6.3/training-settings.json)
- [Training history CSV](../results/v6.3/training-history.csv) and [JSON](../results/v6.3/training-history.json)
- [Evaluation summary CSV](../results/v6.3/evaluation-summary.csv) and [JSON](../results/v6.3/evaluation-summary.json)
- [Zero-target aggregate diagnostics](../results/v6.3/zero-target-summary.json)

The full original trainer-state file and all per-design evaluation records are excluded. The code release supports experiments with user-supplied compatible artifacts; it does not provide an independent public-data reproduction of these historical measurements.
