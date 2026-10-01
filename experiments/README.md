# Historical training attempts

These entrypoints preserve research implementations alongside the supported
PPA workflow. They are experimental code, with no verified numerical comparison
establishing that their added objectives improve PPA prediction or optimization
search. Their historical split names and model settings are editable defaults;
the release contains none of the associated datasets or weights.

| Launcher choice | Motivation and preserved implementation | Historical context | Additional dependency |
| --- | --- | --- | --- |
| `number-token-loss` | Combine token cross-entropy with a loss that penalizes numerical distance between predicted digit tokens. | Preserved loss experiment; no recovered numerical ablation. This is not evidence that the retained v6.3 model used NT loss. | `ntloss==0.3.1` |
| `latent-losses` | Retain numeric generation while regularizing encoder representations with regression, RankSim, RNC, ConR, or pair ordering losses. | Scripts originally targeted v6.2.1 family/group splits. | SciPy and the PPA runtime |
| `mixed-rank` | Use PPA splits for CE and family/correlation splits for ordering regularizers. | v6.2.1 split formulation. | SciPy and the PPA runtime |
| `pair-classifier` | Classify metric-wise pairs as less, approximately equal, or greater using a generic encoder backbone and a learned comparison head. | v6.2.1 easy/hard training and random/family evaluation split formulation. | SciPy, PEFT, a compatible encoder |
| `t5gemma-pair-classifier` | Apply the pair-classification objective to a T5Gemma encoder, including optional LoRA. | v6.2.1 pair-classification variant. | SciPy, PEFT, the PPA runtime |
| `post-finetune` | Adapt an existing numeric-generation checkpoint using family-aware regression, ordering, and contrastive heads, with optional CE. | v6.2.1 family post-finetuning formulation. | SciPy and a compatible seq2seq checkpoint |

All six workflows require explicit dataset, model, and output arguments. Use a
new output directory outside the input artifacts. `--model` selects the exact
checkpoint directory or an explicitly named public Hub model; the launcher does
not search a training output tree for another checkpoint. The remaining options
after `--` belong to the selected historical script.

Install the research runtime and optional numeric loss:

```sh
python -m pip install '.[ppa,experiments]'
```

Inspect an entrypoint's options without loading a model:

```sh
python scripts/run_experiment.py number-token-loss \
  --dataset /path/to/dataset --model /path/to/checkpoint \
  --output /path/to/new-run -- --help
```

Use the following command form with any choice in the table. The example is a
dispatch preview and performs no training:

```sh
python scripts/run_experiment.py latent-losses \
  --dataset /path/to/dataset --model /path/to/checkpoint \
  --output /path/to/new-run --dry-run -- --do_train \
  --train_splits train --eval_splits valid
```

For a CE plus Number Token Loss training run, supply compatible external inputs:

```sh
python scripts/run_experiment.py number-token-loss \
  --dataset /path/to/dataset --model /path/to/checkpoint \
  --output /path/to/number-token-run -- --do_train \
  --losses ce ntloss --ce_weight 1.0 --ntloss_weight 0.3 \
  --train_splits train --eval_splits valid
```

The objective is `1.0 * CE + 0.3 * NTLoss` by default. Digit-level processing and
reweighting are enabled; the numeric component receives `ignore_index=-100`.
`--no_ntloss_digit_level` and `--no_ntloss_reweigh` reproduce the original
alternate switches. The tokenizer and model vocabularies must be compatible,
including digit tokens. The NTLoss algorithm comes from the external
[Number Token Loss package](https://github.com/ai4sd/number-token-loss).

`mixed-rank` selects datasets with `--ppa_train_splits`,
`--correlation_train_splits`, `--ppa_eval_splits`, and
`--correlation_eval_splits`. The two pair classifiers use
`--easy_train_splits`, `--hard_train_splits`, `--random_eval_splits`, and
`--family_eval_splits`. The other family variants use `--train_splits` and
`--eval_splits`. Their labels require the metric fields listed in the selected
script, and their grouping behavior uses the historical family/design/clock
fields. Inspect `--help` and the dataset classes before supplying a different
formulation; changing these fields changes the experiment.

For evaluation-only runs, pass `--do_eval` and select the exact trained model.
Pair classifiers require their saved `pairwise_classifier_head.pt`.
`post-finetune` regression-head evaluation requires `ppa_hybrid_heads.pt` in
the supplied checkpoint; `--eval_prediction_mode lm_head` instead evaluates its
numeric-generation head. For LoRA pair evaluation, use an explicit
`--eval_model_dir` for the adapter and a compatible base model; the T5Gemma
variant also requires its underlying base model via `--encoder_model_name`.

Resuming the token/latent/mixed trainers requires both
`--resume_from_checkpoint` and an exact `--checkpoint_path`. Pair classifiers
accept `--resume_from_checkpoint /path/to/checkpoint`. Outputs remain separate
from resumed checkpoint inputs. Tracking defaults to offline in the public
launcher; an explicit `WANDB_MODE` environment setting can override this.

The launchers alter artifact selection and configuration safety. Objective
formulas, data grouping, scoring, and evaluation calculations remain preserved.
The pair classifiers came from separate historical helper versions. The
T5Gemma attempt uses its matching weighted-pair dataset/trainer helpers in
`training/t5gemma_pair_helpers.py`, including its existing constant, linear,
and cosine schedules for easy/hard sample weights. The generic pair entrypoint
retains its earlier helper implementation. Neither variant substitutes helper
definitions from the other version.
Number Token Loss composition is extracted into a lightweight helper so it can
be tested on synthetic tensors without loading models or datasets. Only the
three required helper functions from the Rocket and earlier training scripts
are retained; those unrelated training workflows are not imported.
