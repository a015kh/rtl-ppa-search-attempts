import logging

import numpy as np
import torch
import torch.nn.functional as F
from omnipred import P10Tokenizer
from tqdm.auto import tqdm
from transformers import Trainer, TrainerCallback
from transformers.integrations import WandbCallback
from transformers import AutoTokenizer
from omnipred import T5GemmaModel
import ipdb

logger = logging.getLogger(__name__)


def tokenized_float(value: float, tokenizer: P10Tokenizer) -> float:
    ids = tokenizer(str(value))
    token_float = tokenizer.decode(ids.input_ids)
    return float(token_float)

class OmniPredNetlistTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        self._last_extra_losses = {
            "aux_loss": float(outputs.aux_loss.detach().cpu()),
            "main_loss": float(outputs.main_loss.detach().cpu()),
        }
        if hasattr(outputs, "ts_loss"):
            self._last_extra_losses["ts_loss"] = float(outputs.ts_loss.detach().cpu())

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        extras = getattr(self, "_last_extra_losses", None)
        if extras:
            logs.update(extras)  # merges into same packet
        return super().log(logs, *args, **kwargs)

    def _save(self, output_dir=None, state_dict=None):
        """Override to use save_pretrained on the inner HF model.

        The default Trainer._save calls safetensors.save_file on the raw
        state_dict, which raises a RuntimeError when weights are tied
        (e.g. embed_tokens and lm_head share the same tensor).
        Delegating to the wrapper's save_pretrained (which calls the inner
        HF model's save_pretrained) resolves this because HuggingFace's
        own serialization path is aware of tied weights and handles them
        correctly.
        """
        output_dir = output_dir or self.args.output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)

        # unwrap from DDP / FSDP / DeepSpeed if necessary
        model = self.accelerator.unwrap_model(self.model)

        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir)
        else:
            # Fallback to the parent implementation for models without save_pretrained
            super()._save(output_dir, state_dict=state_dict)

class ExtraLossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is None:
            return

        extras = getattr(trainer, "_last_extra_losses", None)
        if not extras:
            return

        # inject into the same log packet (respects logging_steps)
        logs.update(extras)

class GenerationWandbCallback(WandbCallback):
    """Custom callback to log generation outputs to Weights & Biases.
    This callback uses model.decode to generate outputs and logs them to W&B.
    """

    def __init__(
        self, trainer: Trainer, datasets: dict, tokenizer: P10Tokenizer, num_samples: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.datasets = datasets
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, **kwargs):

        # super().on_evaluate(args, state, control, **kwargs)
        if not self.trainer.is_world_process_zero():
            return

        # eval_dataloader = self.trainer.get_eval_dataloader()
        # self._evaluate(eval_dataloader)
        for split, dataset in self.datasets.items():
            dataloader = self.trainer.get_eval_dataloader(dataset)
            self._evaluate(dataloader, split)

    def _evaluate(self, eval_dataloader, split: str = "eval"):
        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        device = self.trainer.accelerator.device
        epoch_mae = []
        epoch_relative_mae = []
        total_samples = 0

        y_min, y_max = float("inf"), float("-inf")
        with torch.no_grad():
            model.eval()
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                if self.num_samples > 1:
                    output_ids = model.decode(
                        batch, num_samples=self.num_samples
                    )
                else:
                    output_ids = model.decode(
                        batch
                    ).squeeze()  # (B, num_samples, L_decode) -> (B, L_decode)
                if "labels" in batch:
                    labels = batch["labels"].cpu()
                    ground_truth = torch.zeros(
                        (labels.shape[0], labels.shape[1] + 1), dtype=torch.long
                    )
                    ground_truth[:, 1:] = labels
                    ground_truth[:, 0] = self.tokenizer.bos_token_id
                elif "targets" in batch:
                    ground_truth = batch["targets"].cpu()
                else:
                    ground_truth = batch["labels"].cpu()

                ground_truth_str = self.tokenizer.batch_decode(ground_truth.tolist(), skip_special_tokens=True)
                ground_truth_floats = torch.tensor(
                    [float(gt) for gt in ground_truth_str]
                )

                for sample_output_ids, sample_ground_truth in zip(
                    output_ids, ground_truth_floats
                ):
                    # Update min and max for normalization
                    y_min = min(y_min, sample_ground_truth)
                    y_max = max(y_max, sample_ground_truth)
                    # Decode the output IDs to strings
                    try:
                        if self.num_samples > 1:
                            output_str = self.tokenizer.batch_decode(sample_output_ids, skip_special_tokens=True)
                            # get the median of the output floats
                            output_float = float(np.median([float(os) for os in output_str]))
                        else:
                            output_str = self.tokenizer.decode(sample_output_ids, skip_special_tokens=True)
                            output_float = float(output_str)
                    except Exception as e:
                        # logger.info(f"Error decoding output IDs: {e}")
                        continue
                    
                    if output_float is float("nan"):
                        # logger.info("Output is NaN, skipping this sample.")
                        continue

                    # Calculate normalized mean absolute error
                    mae = abs(output_float - sample_ground_truth)
                    relative_mae = mae / (sample_ground_truth + 1e-10)
                    epoch_mae.append(mae)
                    epoch_relative_mae.append(relative_mae)
                    total_samples += 1

        if len(epoch_mae) == 0:
            # logger.info("No evaluation data found. Skipping evaluation.")
            # pass
            epoch_mae = y_max - y_min
            relative_mae = 1.0
            normalized_mae = 1.0
            acc_at_5_percent = 0.0
            acc_at_10_percent = 0.0
            acc_at_20_percent = 0.0
        else:
            epoch_mae = torch.tensor(epoch_mae).mean().item()
            relative_mae = torch.tensor(epoch_relative_mae).mean().item()
            normalized_mae = epoch_mae / (y_max - y_min) if y_max > y_min else epoch_mae
            acc_at_5_percent = round(
                float(np.mean(np.array(epoch_relative_mae) < 0.05)) * 100, 2
            )
            acc_at_10_percent = round(
                float(np.mean(np.array(epoch_relative_mae) < 0.1)) * 100, 2
            )
            acc_at_20_percent = round(
                float(np.mean(np.array(epoch_relative_mae) < 0.2)) * 100, 2
            )

        decode_ratio = (
            total_samples / len(eval_dataloader.dataset)
            if len(eval_dataloader.dataset) > 0
            else 0
        )
        result = {
            f"{split}/mae": epoch_mae,
            f"{split}/normalized_mae": normalized_mae,
            f"{split}/decode_ratio": decode_ratio,
            f"{split}/relative_mae": relative_mae,
            f"{split}/acc_at_5_percent": acc_at_5_percent,
            f"{split}/acc_at_10_percent": acc_at_10_percent,
            f"{split}/acc_at_20_percent": acc_at_20_percent,
        }
        self._wandb.log(result)
        logger.info(
            f"Split: {split} Epoch MAE: {epoch_mae}, Normalized MAE: {normalized_mae}, Decode Ratio: {decode_ratio}, Relative MAE: {relative_mae}"
        )
        logger.info(
            f"Split: {split} Acc@5%: {acc_at_5_percent}, Acc@10%: {acc_at_10_percent}, Acc@20%: {acc_at_20_percent}"
        )


class T5GemmaCallback(WandbCallback):
    def __init__(
        self,
        trainer: Trainer,
        datasets: dict,
        tokenizer: AutoTokenizer,
        num_samples: int,
        model: T5GemmaModel,
        greedy: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.datasets = datasets
        self.num_samples = num_samples
        self.model = model
        self.greedy = greedy

    def on_evaluate(self, args, state, control, **kwargs):

        # super().on_evaluate(args, state, control, **kwargs)
        if not self.trainer.is_world_process_zero():
            return
        
        for split, dataset in self.datasets.items():
            dataloader = self.trainer.get_eval_dataloader(dataset)
            self._evaluate(dataloader, split, num_samples=self.num_samples)
            if self.greedy:
                self._evaluate(dataloader, f"{split}_greedy", num_samples=1)

    def _evaluate(self, eval_dataloader, split: str = "eval", num_samples: int = 1):
        device = self.trainer.accelerator.device
        epoch_mae = []
        epoch_relative_mae = []
        total_samples = 0

        y_min, y_max = float("inf"), float("-inf")
        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(eval_dataloader, desc=f"Evaluating {split}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                output_ids = self.model.decode(
                    batch, num_samples=num_samples
                )
                if "labels" in batch:
                    labels = batch["labels"].cpu()
                    labels[labels == -100] = self.tokenizer.pad_token_id
                    ground_truth = labels
                elif "targets" in batch:
                    targets = batch["targets"].cpu()
                    targets[targets == -100] = self.tokenizer.pad_token_id
                    ground_truth = targets
                else:
                    ground_truth = batch["labels"].cpu()

                ground_truth_str = self.tokenizer.batch_decode(ground_truth.tolist(), skip_special_tokens=True)
                ground_truth_floats = torch.tensor(
                    [float(gt) for gt in ground_truth_str]
                )

                for sample_output_ids, sample_ground_truth in zip(
                    output_ids, ground_truth_floats
                ):
                    # Update min and max for normalization
                    y_min = min(y_min, sample_ground_truth)
                    y_max = max(y_max, sample_ground_truth)
                    # Decode the output IDs to strings
                    try:
                        output_str = self.tokenizer.batch_decode(sample_output_ids, skip_special_tokens=True)
                        # get the median of the output floats
                        output_float = float(np.median([float(os) for os in output_str]))
                    except Exception as e:
                        # logger.info(f"Error decoding output IDs: {e}")
                        # ipdb.set_trace()
                        continue
                    
                    if output_float is float("nan"):
                        # logger.info("Output is NaN, skipping this sample.")
                        continue

                    # Calculate normalized mean absolute error
                    mae = abs(output_float - sample_ground_truth)
                    relative_mae = mae / (sample_ground_truth + 1e-10)
                    epoch_mae.append(mae)
                    epoch_relative_mae.append(relative_mae)
                    total_samples += 1

        if len(epoch_mae) == 0:
            # logger.info("No evaluation data found. Skipping evaluation.")
            # pass
            epoch_mae = y_max - y_min
            relative_mae = 1.0
            normalized_mae = 1.0
            acc_at_5_percent = 0.0
            acc_at_10_percent = 0.0
            acc_at_20_percent = 0.0
        else:
            epoch_mae = torch.tensor(epoch_mae).mean().item()
            relative_mae = torch.tensor(epoch_relative_mae).mean().item()
            normalized_mae = epoch_mae / (y_max - y_min) if y_max > y_min else epoch_mae
            acc_at_5_percent = round(
                float(np.mean(np.array(epoch_relative_mae) < 0.05)) * 100, 2
            )
            acc_at_10_percent = round(
                float(np.mean(np.array(epoch_relative_mae) < 0.1)) * 100, 2
            )
            acc_at_20_percent = round(
                float(np.mean(np.array(epoch_relative_mae) < 0.2)) * 100, 2
            )

        decode_ratio = (
            total_samples / len(eval_dataloader.dataset)
            if len(eval_dataloader.dataset) > 0
            else 0
        )
        result = {
            f"{split}/mae": epoch_mae,
            f"{split}/normalized_mae": normalized_mae,
            f"{split}/decode_ratio": decode_ratio,
            f"{split}/relative_mae": relative_mae,
            f"{split}/acc_at_5_percent": acc_at_5_percent,
            f"{split}/acc_at_10_percent": acc_at_10_percent,
            f"{split}/acc_at_20_percent": acc_at_20_percent,
        }
        self._wandb.log(result)
        logger.info(
            f"Split: {split} Epoch MAE: {epoch_mae}, Normalized MAE: {normalized_mae}, Decode Ratio: {decode_ratio}, Relative MAE: {relative_mae}"
        )
        logger.info(
            f"Split: {split} Acc@5%: {acc_at_5_percent}, Acc@10%: {acc_at_10_percent}, Acc@20%: {acc_at_20_percent}"
        )

class GenerationWandbCallbackWithFloat(GenerationWandbCallback):
    """Custom callback to log generation outputs to Weights & Biases for float regression tasks.
    This callback uses model.decode to generate outputs and logs them to W&B.
    """

    def __init__(
        self, trainer: Trainer, datasets: dict, tokenizer: P10Tokenizer, *args, **kwargs
    ):
        super().__init__(trainer, datasets, tokenizer, *args, **kwargs)
        self.mean = self.trainer.train_dataset.mean
        self.std = self.trainer.train_dataset.std

    def _evaluate(self, eval_dataloader, split: str = "eval"):
        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        device = self.trainer.accelerator.device
        epoch_mae = []
        epoch_relative_mae = []
        y_values = []
        y_predictions = []
        total_samples = 0

        y_min, y_max = float("inf"), float("-inf")
        with torch.no_grad():
            model.eval()
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                batch_output = model(**batch)
                output_floats = (
                    batch_output.logits.squeeze(1).float() * self.std + self.mean
                )  # Denormalize the output
                labels = batch["labels"].float().squeeze(1) * self.std + self.mean

                # Compute metrics
                epoch_mae.extend(
                    F.l1_loss(output_floats, labels, reduction="none")
                    .abs()
                    .cpu()
                    .tolist()
                )
                epoch_relative_mae.extend(
                    (
                        F.l1_loss(output_floats, labels, reduction="none")
                        / (labels + 1e-6)
                    )
                    .abs()
                    .cpu()
                    .tolist()
                )
                y_values.extend(labels.cpu().numpy())
                y_predictions.extend(output_floats.cpu().numpy())
                total_samples += labels.size(0)

        if len(epoch_mae) == 0:
            # logger.info("No evaluation data found. Skipping evaluation.")
            # pass
            epoch_mae = y_max - y_min
            relative_mae = 1.0
            normalized_mae = 1.0
            acc_at_5_percent = 0.0
            acc_at_10_percent = 0.0
            acc_at_20_percent = 0.0
        else:
            y_max = max(y_values)
            y_min = min(y_values)
            epoch_mae = torch.tensor(epoch_mae).mean().item()
            relative_mae = torch.tensor(epoch_relative_mae).mean().item()
            normalized_mae = epoch_mae / (y_max - y_min) if y_max > y_min else epoch_mae
            acc_at_5_percent = round(
                float(np.mean(np.array(relative_mae) < 0.05)) * 100, 2
            )
            acc_at_10_percent = round(
                float(np.mean(np.array(relative_mae) < 0.1)) * 100, 2
            )
            acc_at_20_percent = round(
                float(np.mean(np.array(relative_mae) < 0.2)) * 100, 2
            )

        result = {
            f"{split}/mae": epoch_mae,
            f"{split}/normalized_mae": normalized_mae,
            f"{split}/relative_mae": relative_mae,
            f"{split}/acc_at_5_percent": acc_at_5_percent,
            f"{split}/acc_at_10_percent": acc_at_10_percent,
            f"{split}/acc_at_20_percent": acc_at_20_percent,
        }
        self._wandb.log(result)
        logger.info(
            f"Split: {split} Epoch MAE: {epoch_mae}, Normalized MAE: {normalized_mae}, Relative MAE: {relative_mae}"
        )
        logger.info(
            f"Split: {split} Acc@5%: {acc_at_5_percent}, Acc@10%: {acc_at_10_percent}, Acc@20%: {acc_at_20_percent}"
        )


class T5GemmaTrendCallback(T5GemmaCallback):
    """Evaluation callback for multi-clock (trend) Seq2Seq regression.

    Labels are newline-separated floats, one per synthesis clock period.
    The evaluation metric is the **average per-target accuracy** over all
    clock positions found in the decoded output.
    """

    def _evaluate(self, eval_dataloader, split: str = "eval", num_samples: int = 1):
        device = self.trainer.accelerator.device

        # Per-target accumulators
        all_relative_maes: list[float] = []   # flat list across all targets
        per_target_relative_maes: list[list[float]] = []  # [target_idx][sample]
        total_samples = 0
        total_decode_fail = 0

        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(eval_dataloader, desc=f"Evaluating {split}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                output_ids = self.model.decode(batch, num_samples=num_samples)

                # Retrieve ground-truth label tokens
                if "labels" in batch:
                    labels = batch["labels"].cpu()
                    labels[labels == -100] = self.tokenizer.pad_token_id
                    ground_truth = labels
                else:
                    ground_truth = batch["targets"].cpu()

                # Decode ground-truth strings (each has N floats sep by \n)
                ground_truth_strs = self.tokenizer.batch_decode(
                    ground_truth.tolist(), skip_special_tokens=True
                )

                for sample_output_ids, gt_str in zip(output_ids, ground_truth_strs):
                    # Parse ground-truth targets
                    try:
                        gt_floats = [float(v.strip()) for v in gt_str.strip().split("\n") if v.strip()]
                    except ValueError:
                        total_decode_fail += 1
                        continue

                    if len(gt_floats) == 0:
                        total_decode_fail += 1
                        continue

                    # Decode model output; output_ids is list of draws per sample
                    try:
                        out_str = self.tokenizer.batch_decode(
                            sample_output_ids, skip_special_tokens=True
                        )
                        # Each draw may contain newline-separated floats.
                        # Collect draws that match the expected number of targets.
                        decoded_per_draw: list[list[float]] = []
                        for draw in out_str:
                            try:
                                floats = [float(v.strip()) for v in draw.strip().split("\n") if v.strip()]
                                if len(floats) == len(gt_floats):
                                    decoded_per_draw.append(floats)
                            except ValueError:
                                continue

                        if len(decoded_per_draw) == 0:
                            total_decode_fail += 1
                            continue

                        # Median across draws per clock position
                        pred_floats = [
                            float(np.median([draw[i] for draw in decoded_per_draw]))
                            for i in range(len(gt_floats))
                        ]
                    except Exception:
                        total_decode_fail += 1
                        continue

                    # Accumulate per-target relative MAE
                    for i, (pred, gt) in enumerate(zip(pred_floats, gt_floats)):
                        rel_mae = abs(pred - gt) / (abs(gt) + 1e-10)
                        all_relative_maes.append(rel_mae)
                        if i >= len(per_target_relative_maes):
                            per_target_relative_maes.append([])
                        per_target_relative_maes[i].append(rel_mae)

                    total_samples += 1

        dataset_size = len(eval_dataloader.dataset)
        decode_ratio = total_samples / dataset_size if dataset_size > 0 else 0.0

        if total_samples == 0:
            result = {
                f"{split}/avg_acc_at_5_percent": 0.0,
                f"{split}/avg_acc_at_10_percent": 0.0,
                f"{split}/avg_acc_at_20_percent": 0.0,
                f"{split}/avg_relative_mae": 1.0,
                f"{split}/decode_ratio": decode_ratio,
                f"{split}/decode_fail": total_decode_fail,
            }
        else:
            arr_all = np.array(all_relative_maes)
            avg_rel_mae = float(arr_all.mean())
            avg_acc_5  = round(float(np.mean(arr_all < 0.05)) * 100, 2)
            avg_acc_10 = round(float(np.mean(arr_all < 0.1))  * 100, 2)
            avg_acc_20 = round(float(np.mean(arr_all < 0.2))  * 100, 2)

            result = {
                f"{split}/avg_relative_mae":      avg_rel_mae,
                f"{split}/avg_acc_at_5_percent":  avg_acc_5,
                f"{split}/avg_acc_at_10_percent": avg_acc_10,
                f"{split}/avg_acc_at_20_percent": avg_acc_20,
                f"{split}/decode_ratio":           decode_ratio,
                f"{split}/decode_fail":            total_decode_fail,
            }

            # Per-target breakdown
            for i, target_maes in enumerate(per_target_relative_maes):
                arr = np.array(target_maes)
                result[f"{split}/clock{i}_acc_at_10_percent"] = round(
                    float(np.mean(arr < 0.1)) * 100, 2
                )
                result[f"{split}/clock{i}_acc_at_20_percent"] = round(
                    float(np.mean(arr < 0.2)) * 100, 2
                )
                result[f"{split}/clock{i}_acc_at_5_percent"] = round(
                    float(np.mean(arr < 0.05)) * 100, 2
                )
                # result[f"{split}/clock{i}_relative_mae"] = float(arr.mean())

        if hasattr(self, "_wandb"):
            self._wandb.log(result)
        logger.info(f"[{split}] Trend eval: {result}")


class T5GemmaPPACallback(T5GemmaTrendCallback):
    """Like T5GemmaTrendCallback but logs per-metric names instead of clock{i}.

    Pass ``metrics`` (ordered list matching the dataset's target order) so that
    W&B keys read ``{split}/area_acc_at_10_percent`` etc. rather than
    ``{split}/clock0_acc_at_10_percent``.
    """

    def __init__(self, *args, metrics: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ppa_metrics = metrics  # e.g. ["area", "delay", "static_power", "switching_power"]

    def _evaluate(self, eval_dataloader, split: str = "eval", num_samples: int = 1):
        # Call parent which accumulates per_target_relative_maes and logs clock{i} keys.
        # We override fully to emit named keys instead.
        device = self.trainer.accelerator.device

        all_relative_maes: list[float] = []
        per_target_relative_maes: list[list[float]] = []
        max_relative_maes: list[float] = []
        total_samples = 0
        total_decode_fail = 0

        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(eval_dataloader, desc=f"Evaluating {split}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                output_ids = self.model.decode(batch, num_samples=num_samples)

                if "labels" in batch:
                    labels = batch["labels"].cpu()
                    labels[labels == -100] = self.tokenizer.pad_token_id
                    ground_truth = labels
                else:
                    ground_truth = batch["targets"].cpu()

                ground_truth_strs = self.tokenizer.batch_decode(
                    ground_truth.tolist(), skip_special_tokens=True
                )

                for sample_output_ids, gt_str in zip(output_ids, ground_truth_strs):
                    try:
                        gt_floats = [float(v.strip()) for v in gt_str.strip().split("\n") if v.strip()]
                    except ValueError:
                        total_decode_fail += 1
                        continue
                    if len(gt_floats) == 0:
                        total_decode_fail += 1
                        continue

                    try:
                        out_str = self.tokenizer.batch_decode(sample_output_ids, skip_special_tokens=True)
                        decoded_per_draw: list[list[float]] = []
                        for draw in out_str:
                            try:
                                floats = [float(v.strip()) for v in draw.strip().split("\n") if v.strip()]
                                if len(floats) == len(gt_floats):
                                    decoded_per_draw.append(floats)
                            except ValueError:
                                continue
                        if len(decoded_per_draw) == 0:
                            total_decode_fail += 1
                            continue
                        pred_floats = [
                            float(np.median([draw[i] for draw in decoded_per_draw]))
                            for i in range(len(gt_floats))
                        ]
                    except Exception:
                        total_decode_fail += 1
                        continue

                    sample_rel_maes = []
                    for i, (pred, gt) in enumerate(zip(pred_floats, gt_floats)):
                        rel_mae = abs(pred - gt) / (abs(gt) + 1e-10)
                        all_relative_maes.append(rel_mae)
                        sample_rel_maes.append(rel_mae)
                        if i >= len(per_target_relative_maes):
                            per_target_relative_maes.append([])
                        per_target_relative_maes[i].append(rel_mae)
                    if sample_rel_maes:
                        max_relative_maes.append(max(sample_rel_maes))
                    total_samples += 1

        dataset_size = len(eval_dataloader.dataset)
        decode_ratio = total_samples / dataset_size if dataset_size > 0 else 0.0

        if total_samples == 0:
            result = {
                f"{split}/avg_acc_at_5_percent":  0.0,
                f"{split}/avg_acc_at_10_percent": 0.0,
                f"{split}/avg_acc_at_20_percent": 0.0,
                f"{split}/avg_relative_mae":      1.0,
                f"{split}/decode_ratio":           decode_ratio,
                f"{split}/decode_fail":            total_decode_fail,
            }
        else:
            arr_all = np.array(all_relative_maes)
            arr_max = np.array(max_relative_maes)
            result = {
                f"{split}/avg_relative_mae":      float(arr_all.mean()),
                f"{split}/avg_acc_at_5_percent":  round(float(np.mean(arr_all < 0.05)) * 100, 2),
                f"{split}/avg_acc_at_10_percent": round(float(np.mean(arr_all < 0.1))  * 100, 2),
                f"{split}/avg_acc_at_20_percent": round(float(np.mean(arr_all < 0.2))  * 100, 2),
                f"{split}/all_metrics_acc_at_5_percent":  round(float(np.mean(arr_max < 0.05)) * 100, 2),
                f"{split}/all_metrics_acc_at_10_percent": round(float(np.mean(arr_max < 0.1))  * 100, 2),
                f"{split}/all_metrics_acc_at_20_percent": round(float(np.mean(arr_max < 0.2))  * 100, 2),
                f"{split}/decode_ratio":           decode_ratio,
                f"{split}/decode_fail":            total_decode_fail,
            }
            for i, target_maes in enumerate(per_target_relative_maes):
                arr = np.array(target_maes)
                # Use named metric if available, else fall back to metric{i}
                name = (self.ppa_metrics[i] if self.ppa_metrics and i < len(self.ppa_metrics)
                        else f"metric{i}")
                result[f"{split}/{name}_acc_at_5_percent"]  = round(float(np.mean(arr < 0.05)) * 100, 2)
                result[f"{split}/{name}_acc_at_10_percent"] = round(float(np.mean(arr < 0.1))  * 100, 2)
                result[f"{split}/{name}_acc_at_20_percent"] = round(float(np.mean(arr < 0.2))  * 100, 2)

        if hasattr(self, "_wandb"):
            self._wandb.log(result)
        logger.info(f"[{split}] PPA eval: {result}")
