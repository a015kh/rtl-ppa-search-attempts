# Historical learning-objective attempts

These implementations preserve research directions explored around the RTL/PPA workflow. They are experimental code rather than validated alternatives to the maintained training entrypoint. Original complete commands and a controlled numerical comparison are unavailable. Historical dataset-version names describe development context; they do not supply public data or prove that an experiment was run.

| Launcher choice | Motivation and implemented objective | Historical context | Evidence and dependencies |
|---|---|---|---|
| `number-token-loss` | Combine token cross-entropy with Number Token Loss to give numerical tokens an additional training signal. | Separate historical loss experiment; not established as the v6.3 checkpoint's objective. | Implementation and observed `ntloss==0.3.1` dependency recovered; no verified numerical comparison. |
| `latent-losses` | Add configurable RankSim, RNC or ConR regularization to latent representations alongside numeric generation. | v6.2.1 family/group split formulation. | Implemented objectives; aggregate efficacy results not recovered. |
| `mixed-rank` | Combine numeric token generation and family-aware ranking so related implementations can be compared. | v6.2.1 family/ranking formulation. | Requires suitable family identifiers and sufficient comparable family members; no controlled published comparison. |
| `pair-classifier` | Predict ordered PPA relationships between paired implementations using a classification formulation. | v6.2.1 easy/hard pair and random/family split formulation. | Requires valid pairs, labels and model/tokenizer compatibility; no verified v6.3 contribution. |
| `t5gemma-pair-classifier` | Pairwise PPA classification using a T5Gemma-specific formulation. | v6.2.1 pairwise training variant. | Separate implementation; no verified numerical comparison. |
| `post-finetune` | Apply family-aware refinement to an existing PPA model with the retained post-finetuning objectives. | v6.2.1 family post-finetuning formulation. | Requires compatible checkpoint and grouping inputs; no recovered controlled improvement claim. |

## Launching an attempt

Install the maintained PPA dependencies and optional experiment dependencies:

```bash
python -m pip install -e ".[ppa,experiments,dev]"
python scripts/run_experiment.py --help
```

Every experiment receives explicit dataset, model and output locations. Additional historical script arguments follow `--`. Inspect the specific experiment's help before running:

```bash
python scripts/run_experiment.py number-token-loss --dataset /path/to/dataset --model /path/to/base-model --output /path/to/new-nt-loss-run -- --help
python scripts/run_experiment.py number-token-loss --dataset /path/to/dataset --model /path/to/base-model --output /path/to/new-nt-loss-run -- --do_train --train_splits train --eval_splits valid --losses ce ntloss --ce_weight 1.0 --ntloss_weight 0.3
python scripts/run_experiment.py latent-losses --dataset /path/to/dataset --model /path/to/base-model --output /path/to/new-latent-run -- --help
python scripts/run_experiment.py mixed-rank --dataset /path/to/dataset --model /path/to/base-model --output /path/to/new-mixed-rank-run -- --help
python scripts/run_experiment.py pair-classifier --dataset /path/to/dataset --model /path/to/base-model --output /path/to/new-pair-run -- --help
python scripts/run_experiment.py t5gemma-pair-classifier --dataset /path/to/dataset --model /path/to/base-model --output /path/to/new-t5gemma-pair-run -- --help
python scripts/run_experiment.py post-finetune --dataset /path/to/dataset --model /path/to/checkpoint --output /path/to/new-post-run -- --help
```

Implementations are under `experiments/training/`. The launcher preserves their different parsers and maps the explicit common locations to each script. Grouping, pair construction, latent regularization and other experiment-specific options remain controlled by the script. The [experiment usage notes](../experiments/README.md) list objective-specific split arguments, saved classifier/head requirements, adapter evaluation and exact resume arguments. A launch help check verifies its import/argument surface; it does not validate a complete training run.

## Number Token Loss details

The default enabled objectives are cross-entropy (`ce`) and Number Token Loss (`ntloss`). The implemented weighted objective is `total_loss = 1.0 * CE + 0.3 * NT` with those default weights. Digit-level processing and numeric-token reweighting are enabled. `--no_ntloss_digit_level` and `--no_ntloss_reweigh` disable them; `--ntloss_squash_factor` optionally adjusts squashing. The same labels are passed to the numeric loss with explicit `ignore_index=-100`; the model supplies cross-entropy with its masked-label behavior. Synthetic loss checks cover composition and masking.

The optional dependency is pinned to `ntloss==0.3.1`, the version recovered from the observed research environment. The dependency supplies its own implementation and license; it is installed as a package, not vendored here.

No verified numerical Number Token Loss comparison was recovered. Old report-generator statements are not experimental evidence and are excluded. The standalone v6.3 results must not be described as NT-loss results. The release retains the attempt so its idea and implementation can be inspected or tested using compatible user-supplied data.
