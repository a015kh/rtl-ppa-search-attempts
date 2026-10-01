"""Fine-tune token PPA generation with latent plug-and-play losses.

This script keeps the task from ``finetune_ppa.py``: generate PPA metric values
as tokens.  The extra RankSim/RNC/ConR losses regularize pooled encoder
representations inside same-family batches, so the model can learn ordering
relations from generated/feedback groups without replacing token generation
with a pure regression-head predictor.  The optional ``pair`` loss adds a
smooth less/equal/greater margin objective on the latent regression head.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

from finetune_ppa import apply_lora, lora_target_modules
from omnipred import T5GemmaModel, T5GemmaModelConfig
from post_finetune_ppa import (
    DEFAULT_TRAIN_SPLITS,
    FamilyBatchSampler,
    PPACollator,
    PPAPostTuneDataset,
    compute_prediction_correlations,
    compute_prediction_error_summary,
    load_splits,
    normalize_split_name,
    parse_metric_values,
    write_csv,
    write_eval_summaries,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

DEFAULT_EVAL_SPLITS = [
    "test_aug_100",
    "test_aug_200",
    "test_pyra_200",
    "test_ppa_feedback",
    "test_ds_verilog_eval",
]
DEFAULT_METRICS = ["area", "delay", "total_power"]


def resolve_model_name_or_path(args) -> str:
    if args.base_model_dir:
        model_name_or_path = args.base_model_dir
        logger.info("Loading base model from %s", model_name_or_path)
        return model_name_or_path
    logger.info("Loading model from HF hub: %s", args.encoder_model_name)
    return args.encoder_model_name


def split_path(dataset_dir: str, split: str) -> str:
    return os.path.join(dataset_dir, f"{normalize_split_name(split)}.jsonl")


def existing_splits(dataset_dir: str, split_names: Iterable[str], required: bool) -> list[str]:
    found = []
    missing = []
    for split in split_names:
        split_name = normalize_split_name(split)
        if os.path.exists(split_path(dataset_dir, split_name)):
            found.append(split_name)
        else:
            missing.append(split_name)
    if missing:
        message = f"Missing split(s) in {dataset_dir}: {', '.join(missing)}"
        if required:
            raise FileNotFoundError(message)
        logger.warning("%s. They will be skipped.", message)
    return found


def compute_log_stats(samples: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    values = np.stack([sample["ppa_values"] for sample in samples]).astype(np.float32)
    log_values = np.log(np.maximum(values, 1e-12))
    mean = log_values.mean(axis=0)
    std = log_values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


def infer_hidden_size(config) -> int:
    for attr in ("d_model", "hidden_size"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    encoder_config = getattr(config, "encoder", None)
    if encoder_config is not None:
        for attr in ("d_model", "hidden_size"):
            value = getattr(encoder_config, attr, None)
            if value is not None:
                return int(value)
    raise ValueError("Could not infer hidden size from model config.")


class LatentRegularizedSeq2SeqModel(nn.Module):
    """Seq2seq model plus training-only heads for latent regularization."""

    def __init__(
        self,
        base_model: nn.Module,
        num_metrics: int,
        projector_dim: int,
        dropout: float,
        log_mean: torch.Tensor,
        log_std: torch.Tensor,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)
        self._keys_to_ignore_on_save = getattr(base_model, "_keys_to_ignore_on_save", None)
        self._keys_to_ignore_on_load_missing = getattr(base_model, "_keys_to_ignore_on_load_missing", None)
        self._keys_to_ignore_on_load_unexpected = getattr(base_model, "_keys_to_ignore_on_load_unexpected", None)
        hidden_size = infer_hidden_size(base_model.config)

        if projector_dim > 0:
            self.projector = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, projector_dim),
                nn.GELU(),
                nn.Linear(projector_dim, projector_dim),
            )
            rep_dim = projector_dim
        else:
            self.projector = nn.Identity()
            rep_dim = hidden_size

        self.regression_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_metrics),
        )
        self.register_buffer("log_mean", log_mean.clone())
        self.register_buffer("log_std", log_std.clone())
        self.rep_dim = rep_dim

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def save_pretrained(self, output_dir: str, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        self.base_model.save_pretrained(output_dir, **kwargs)
        torch.save(
            {
                "projector": self.projector.state_dict(),
                "regression_head": self.regression_head.state_dict(),
                "log_mean": self.log_mean.detach().cpu(),
                "log_std": self.log_std.detach().cpu(),
                "rep_dim": self.rep_dim,
            },
            os.path.join(output_dir, "ppa_latent_heads.pt"),
        )

    def normalized_targets(self, ppa_values: torch.Tensor) -> torch.Tensor:
        log_values = torch.log(ppa_values.clamp_min(1e-12))
        return (log_values - self.log_mean) / self.log_std

    def pooled_encoder_state(self, outputs, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = outputs.encoder_last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def pair_mask_from_families(families: list[str], device: torch.device, same_family_only: bool) -> torch.Tensor:
    n = len(families)
    mask = torch.ones(n, n, device=device, dtype=torch.bool)
    mask.fill_diagonal_(False)
    if same_family_only:
        family_mask = torch.tensor(
            [[families[i] == families[j] for j in range(n)] for i in range(n)],
            device=device,
            dtype=torch.bool,
        )
        mask = mask & family_mask
    return mask


def pairwise_label_distance(targets: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mean_l1":
        return torch.cdist(targets, targets, p=1) / max(targets.shape[1], 1)
    if mode == "l2":
        return torch.cdist(targets, targets, p=2)
    raise ValueError(f"Unknown label distance: {mode}")


def ranksim_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    pair_mask: torch.Tensor,
    label_distance: str,
    feature_temperature: float,
    label_temperature: float,
) -> torch.Tensor:
    """Listwise RankSim-style surrogate over batch neighbors."""
    if features.shape[0] < 3 or not pair_mask.any():
        return features.new_tensor(0.0)
    z = F.normalize(features, dim=-1)
    feature_dist = (1.0 - z @ z.T).clamp_min(0.0)
    label_dist = pairwise_label_distance(targets, label_distance)

    losses = []
    for i in range(features.shape[0]):
        valid = pair_mask[i]
        if valid.sum() < 2:
            continue
        target_probs = F.softmax(-label_dist[i, valid] / label_temperature, dim=0)
        feature_log_probs = F.log_softmax(-feature_dist[i, valid] / feature_temperature, dim=0)
        losses.append(-(target_probs.detach() * feature_log_probs).sum())
    if not losses:
        return features.new_tensor(0.0)
    return torch.stack(losses).mean()


def rnc_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    pair_mask: torch.Tensor,
    label_distance: str,
    temperature: float,
) -> torch.Tensor:
    """Rank-N-Contrast loss over target-distance rankings."""
    if features.shape[0] < 3 or not pair_mask.any():
        return features.new_tensor(0.0)
    z = F.normalize(features, dim=-1)
    logits = z @ z.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    label_dist = pairwise_label_distance(targets, label_distance)

    losses = []
    n = features.shape[0]
    for i in range(n):
        valid_refs = torch.nonzero(pair_mask[i], as_tuple=False).flatten()
        for j in valid_refs:
            denominator_mask = pair_mask[i] & (label_dist[i] >= label_dist[i, j])
            if denominator_mask.sum() == 0:
                continue
            losses.append(-(logits[i, j] - torch.logsumexp(logits[i, denominator_mask], dim=0)))
    if not losses:
        return features.new_tensor(0.0)
    return torch.stack(losses).mean()


def conr_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    preds: torch.Tensor,
    pair_mask: torch.Tensor,
    label_distance: str,
    label_width: float,
    pred_width: float,
    temperature: float,
    push_scale: float,
) -> torch.Tensor:
    """ConR-style regularizer using normalized PPA labels and latent-head predictions."""
    if features.shape[0] < 3 or not pair_mask.any():
        return features.new_tensor(0.0)

    z = F.normalize(features, dim=-1)
    logits = z @ z.T / temperature
    label_dist = pairwise_label_distance(targets, label_distance)
    pred_dist = pairwise_label_distance(preds, label_distance)

    positives = (label_dist <= label_width) & pair_mask
    negatives = (label_dist > label_width) & (pred_dist <= pred_width) & pair_mask
    exp_logits = torch.exp(logits)
    push_weight = torch.exp(label_dist.detach() * push_scale)
    neg_mass = (exp_logits * negatives * push_weight).sum(dim=1, keepdim=True)
    pos_mass = (exp_logits * positives).sum(dim=1, keepdim=True)
    denom = (pos_mass + neg_mass).clamp_min(1e-12)

    per_pair = -torch.log((exp_logits / denom).clamp_min(1e-12)) * positives
    pos_count = positives.sum(dim=1)
    has_valid_rows = (pos_count > 0) & (negatives.sum(dim=1) > 0)
    if not has_valid_rows.any():
        return features.new_tensor(0.0)
    return (per_pair.sum(dim=1)[has_valid_rows] / pos_count[has_valid_rows]).mean()


def pairwise_margin_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    pair_mask: torch.Tensor,
    margin: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Smooth less/equal/greater ordering loss over normalized PPA pairs."""
    if preds.shape[0] < 2 or not pair_mask.any():
        return preds.new_tensor(0.0)

    temperature = max(float(temperature), 1e-6)
    margin = margin.to(device=preds.device, dtype=preds.dtype).view(1, 1, -1)
    pred_delta = preds[:, None, :] - preds[None, :, :]
    target_delta = targets[:, None, :] - targets[None, :, :]

    greater = target_delta > margin
    less = target_delta < -margin
    greater_loss = F.softplus((margin - pred_delta) / temperature)
    less_loss = F.softplus((margin + pred_delta) / temperature)
    equal_loss = F.softplus((pred_delta.abs() - margin) / temperature)
    per_metric = torch.where(greater, greater_loss, torch.where(less, less_loss, equal_loss))

    valid = pair_mask.unsqueeze(-1).expand_as(per_metric)
    if not valid.any():
        return preds.new_tensor(0.0)
    return per_metric[valid].mean()


class PPALatentLossTrainer(Trainer):
    def __init__(
        self,
        *args,
        tokenizer,
        train_batch_sampler: FamilyBatchSampler | None,
        enabled_losses: set[str],
        ce_weight: float,
        regression_weight: float,
        ranksim_weight: float,
        rnc_weight: float,
        conr_weight: float,
        pair_weight: float,
        same_family_only: bool,
        label_distance: str,
        ranksim_feature_temperature: float,
        ranksim_label_temperature: float,
        rnc_temperature: float,
        conr_label_width: float,
        conr_pred_width: float,
        conr_temperature: float,
        conr_push_scale: float,
        pair_margin: list[float],
        pair_temperature: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.train_batch_sampler = train_batch_sampler
        self.enabled_losses = enabled_losses
        self.ce_weight = ce_weight
        self.regression_weight = regression_weight
        self.ranksim_weight = ranksim_weight
        self.rnc_weight = rnc_weight
        self.conr_weight = conr_weight
        self.pair_weight = pair_weight
        self.same_family_only = same_family_only
        self.label_distance = label_distance
        self.ranksim_feature_temperature = ranksim_feature_temperature
        self.ranksim_label_temperature = ranksim_label_temperature
        self.rnc_temperature = rnc_temperature
        self.conr_label_width = conr_label_width
        self.conr_pred_width = conr_pred_width
        self.conr_temperature = conr_temperature
        self.conr_push_scale = conr_push_scale
        self.pair_margin = torch.tensor(pair_margin, dtype=torch.float32)
        self.pair_temperature = pair_temperature
        self._loss_sums = defaultdict(float)
        self._loss_count = 0

    def get_train_dataloader(self):
        if self.train_batch_sampler is None:
            return super().get_train_dataloader()
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _remember(self, **values: torch.Tensor):
        self._loss_count += 1
        for key, value in values.items():
            self._loss_sums[key] += float(value.detach().float().cpu())

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ppa_values = inputs["ppa_values"]
        families = inputs["family"]
        model_inputs = {
            k: v
            for k, v in inputs.items()
            if k not in {"ppa_values", "family", "design_name", "clock"}
        }

        outputs = model(**model_inputs, return_dict=True)
        pooled = model.pooled_encoder_state(outputs, model_inputs["attention_mask"])
        latent = model.projector(pooled)
        target_norm = model.normalized_targets(ppa_values)
        pred_norm = model.regression_head(pooled)
        pair_mask = pair_mask_from_families(families, pooled.device, self.same_family_only)

        ce_loss = outputs.loss if outputs.loss is not None else pooled.new_tensor(0.0)
        regression_loss = F.smooth_l1_loss(pred_norm, target_norm)
        ranksim = ranksim_loss(
            latent,
            target_norm,
            pair_mask,
            self.label_distance,
            self.ranksim_feature_temperature,
            self.ranksim_label_temperature,
        )
        rnc = rnc_loss(latent, target_norm, pair_mask, self.label_distance, self.rnc_temperature)
        conr = conr_loss(
            latent,
            target_norm,
            pred_norm,
            pair_mask,
            self.label_distance,
            self.conr_label_width,
            self.conr_pred_width,
            self.conr_temperature,
            self.conr_push_scale,
        )
        pair = pairwise_margin_loss(
            pred_norm,
            target_norm,
            pair_mask,
            self.pair_margin,
            self.pair_temperature,
        )

        total = pooled.new_tensor(0.0)
        if "ce" in self.enabled_losses:
            total = total + self.ce_weight * ce_loss
        if "regression" in self.enabled_losses:
            total = total + self.regression_weight * regression_loss
        if "ranksim" in self.enabled_losses:
            total = total + self.ranksim_weight * ranksim
        if "rnc" in self.enabled_losses:
            total = total + self.rnc_weight * rnc
        if "conr" in self.enabled_losses:
            total = total + self.conr_weight * conr
        if "pair" in self.enabled_losses:
            total = total + self.pair_weight * pair

        self._remember(
            loss=total,
            ce=ce_loss,
            regression=regression_loss,
            ranksim=ranksim,
            rnc=rnc,
            conr=conr,
            pair=pair,
        )
        return (total, outputs) if return_outputs else total

    def log(self, logs: dict[str, float], *args, **kwargs):
        if self._loss_count:
            logs = dict(logs)
            for key, value in self._loss_sums.items():
                logs[f"train/{key}"] = value / self._loss_count
            self._loss_sums.clear()
            self._loss_count = 0
        return super().log(logs, *args, **kwargs)

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir, safe_serialization=True)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)


@torch.no_grad()
def evaluate_generation(
    model: LatentRegularizedSeq2SeqModel,
    dataset: PPAPostTuneDataset,
    tokenizer,
    split_name: str,
    output_dir: str,
    metrics: list[str],
    batch_size: int,
    device: torch.device,
    min_problem_samples: int,
    max_new_tokens: int,
    num_samples: int,
    greedy: bool,
) -> tuple[list[dict], list[dict]]:
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=PPACollator(tokenizer))
    rows = []
    decode_failures = 0
    expected_count = len(metrics)

    for raw_batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
        batch = {
            k: v.to(device)
            for k, v in raw_batch.items()
            if torch.is_tensor(v) and k not in {"ppa_values"}
        }
        output_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_samples,
            do_sample=(not greedy and num_samples > 1),
            pad_token_id=tokenizer.pad_token_id,
        ).view(raw_batch["input_ids"].shape[0], num_samples, -1)
        gt_values = raw_batch["ppa_values"].numpy()

        for i, sample_output_ids in enumerate(output_ids.detach().cpu().tolist()):
            decoded_per_draw = []
            for text in tokenizer.batch_decode(sample_output_ids, skip_special_tokens=True):
                pred_values = parse_metric_values(text, expected_count)
                if pred_values is not None:
                    decoded_per_draw.append(pred_values)
            if not decoded_per_draw:
                decode_failures += 1
                continue
            pred_values = [
                float(np.median([draw[metric_idx] for draw in decoded_per_draw]))
                for metric_idx in range(expected_count)
            ]
            row = {
                "design_name": raw_batch["design_name"][i],
                "problem_name": raw_batch["family"][i],
                "clock": raw_batch["clock"][i],
            }
            for metric_idx, metric in enumerate(metrics):
                row[f"gt_{metric}"] = float(gt_values[i, metric_idx])
                row[f"pred_{metric}"] = pred_values[metric_idx]
            rows.append(row)

    prediction_path = os.path.join(output_dir, f"predictions_ground_truth_{split_name}.csv")
    write_csv(prediction_path, rows)
    logger.info("Saved token-generation predictions to %s (%s decode failures)", prediction_path, decode_failures)

    corr_rows = compute_prediction_correlations(rows, metrics, min_problem_samples)
    write_csv(os.path.join(output_dir, f"problem_ordering_{split_name}.csv"), corr_rows)
    error_rows = compute_prediction_error_summary(rows, metrics, split_name)
    write_csv(os.path.join(output_dir, f"prediction_error_{split_name}.csv"), error_rows)
    write_csv(
        os.path.join(output_dir, f"decode_failures_{split_name}.csv"),
        [{"split": split_name, "decode_failures": decode_failures, "valid_predictions": len(rows)}],
    )
    return corr_rows, error_rows


def build_training_args(args) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        do_eval=args.valid_ratio > 0,
        eval_strategy="steps" if args.valid_ratio > 0 else "no",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        torch_compile=False,
        run_name=os.path.split(args.output_dir.rstrip(os.sep))[-1],
        remove_unused_columns=False,
    )


def make_dataset(records: list[dict], tokenizer, args, max_samples: int | None = None) -> PPAPostTuneDataset:
    return PPAPostTuneDataset(
        records,
        tokenizer,
        args.metrics,
        args.max_input_len,
        args.max_target_len,
        args.rtl_key,
        args.clock_key,
        args.num_digits,
        args.family_key,
        max_samples=max_samples,
        seed=args.seed,
    )


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    unknown_losses = sorted(set(args.losses) - {"ce", "regression", "ranksim", "rnc", "conr", "pair"})
    if unknown_losses:
        raise ValueError(f"Unknown loss(es): {', '.join(unknown_losses)}")
    if "ce" not in args.losses:
        logger.warning("Training without CE means token generation is regularized only indirectly.")
    if len(args.pair_margin) == 1:
        args.pair_margin = [args.pair_margin[0]] * len(args.metrics)
    elif len(args.pair_margin) != len(args.metrics):
        raise ValueError(
            f"Length of --pair_margin ({len(args.pair_margin)}) must be 1 or equal to "
            f"number of metrics ({len(args.metrics)})."
        )

    train_splits = existing_splits(args.dataset_dir, args.train_splits, required=True)
    eval_splits = existing_splits(args.dataset_dir, args.eval_splits, required=False)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "finetune_ppa_latent_losses_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    model_name_or_path = resolve_model_name_or_path(args)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, model_max_length=args.max_input_len)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_records = load_splits(args.dataset_dir, train_splits)
    rng = random.Random(args.seed)
    rng.shuffle(train_records)
    valid_records = []
    if args.valid_ratio > 0:
        n_valid = max(1, int(len(train_records) * args.valid_ratio))
        valid_records = train_records[:n_valid]
        train_records = train_records[n_valid:]

    train_dataset = make_dataset(train_records, tokenizer, args, max_samples=args.max_train_samples)
    valid_dataset = make_dataset(valid_records, tokenizer, args, max_samples=args.max_eval_samples) if valid_records else None
    if len(train_dataset) == 0:
        raise ValueError("No valid training samples were loaded.")

    log_mean, log_std = compute_log_stats(train_dataset.samples)
    config = T5GemmaModelConfig(
        model_name=model_name_or_path,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
    )
    t5gemma = T5GemmaModel(config=config)
    if args.use_lora:
        logger.info(
            "Enabling LoRA fine-tuning: r=%s alpha=%s dropout=%s targets=%s",
            args.lora_rank,
            args.lora_alpha,
            args.lora_dropout,
            lora_target_modules(args),
        )
        t5gemma = apply_lora(t5gemma, args)

    model = LatentRegularizedSeq2SeqModel(
        base_model=t5gemma.model,
        num_metrics=len(args.metrics),
        projector_dim=args.projector_dim,
        dropout=args.head_dropout,
        log_mean=log_mean,
        log_std=log_std,
    )

    if args.freeze_decoder:
        for name, param in model.base_model.named_parameters():
            if "decoder" in name:
                param.requires_grad = False

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Number of trainable parameters: %s", num_params)
    logger.info("Enabled losses: %s", ", ".join(args.losses))

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

    train_sampler = FamilyBatchSampler(train_dataset, args.train_batch_size, shuffle=True, seed=args.seed) if args.group_batches else None
    trainer = PPALatentLossTrainer(
        model=model,
        args=build_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=PPACollator(tokenizer),
        tokenizer=tokenizer,
        train_batch_sampler=train_sampler,
        enabled_losses=set(args.losses),
        ce_weight=args.ce_weight,
        regression_weight=args.regression_weight,
        ranksim_weight=args.ranksim_weight,
        rnc_weight=args.rnc_weight,
        conr_weight=args.conr_weight,
        pair_weight=args.pair_weight,
        same_family_only=args.same_family_only,
        label_distance=args.label_distance,
        ranksim_feature_temperature=args.ranksim_feature_temperature,
        ranksim_label_temperature=args.ranksim_label_temperature,
        rnc_temperature=args.rnc_temperature,
        conr_label_width=args.conr_label_width,
        conr_pred_width=args.conr_pred_width,
        conr_temperature=args.conr_temperature,
        conr_push_scale=args.conr_push_scale,
        pair_margin=args.pair_margin,
        pair_temperature=args.pair_temperature,
    )

    if args.do_train:
        if args.resume_from_checkpoint:
            if not args.checkpoint_path:
                raise ValueError("Resuming requires an explicit --checkpoint_path.")
            trainer.train(resume_from_checkpoint=args.checkpoint_path)
        else:
            trainer.train()
        trainer.save_model(args.output_dir)

    if args.do_eval and eval_splits:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        model.to(device)
        all_corr_rows = []
        all_error_rows = []
        for split in eval_splits:
            records = load_splits(args.dataset_dir, [split])
            dataset = make_dataset(records, tokenizer, args, max_samples=args.max_eval_samples)
            corr_rows, error_rows = evaluate_generation(
                model,
                dataset,
                tokenizer,
                split,
                args.output_dir,
                args.metrics,
                args.eval_batch_size,
                device,
                args.min_problem_samples,
                args.max_target_len,
                args.num_samples,
                args.greedy,
            )
            for row in corr_rows:
                row = dict(row)
                row["split"] = split
                all_corr_rows.append(row)
            all_error_rows.extend(error_rows)
        write_eval_summaries(args.output_dir, all_corr_rows, all_error_rows)

        if all_error_rows:
            summary = pd.DataFrame(all_error_rows)
            logger.info("Wrote token-generation evaluation summaries for %s split(s).", len(eval_splits))
            print(summary)


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False, description="Train token PPA generation with latent RankSim/RNC/ConR/pair losses.")

    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--train_splits", nargs="+", default=DEFAULT_TRAIN_SPLITS)
    parser.add_argument("--eval_splits", nargs="+", default=DEFAULT_EVAL_SPLITS)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--rtl_key", type=str, default="rtl")
    parser.add_argument("--clock_key", type=str, default="clock")
    parser.add_argument("--family_key", type=str, default=None)
    parser.add_argument("--num_digits", type=int, default=4)
    parser.add_argument("--max_input_len", type=int, default=2048)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--valid_ratio", type=float, default=0.02)
    parser.add_argument("--min_problem_samples", type=int, default=10)

    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--encoder_model_name", type=str, default=None, help="Explicit underlying base model when loading an adapter.")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    parser.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])

    parser.add_argument("--losses", nargs="+", default=["ce", "regression", "ranksim", "rnc", "conr"])
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--regression_weight", type=float, default=0.2)
    parser.add_argument("--ranksim_weight", type=float, default=0.2)
    parser.add_argument("--rnc_weight", type=float, default=0.2)
    parser.add_argument("--conr_weight", type=float, default=0.1)
    parser.add_argument("--pair_weight", type=float, default=0.2)
    parser.add_argument("--label_distance", choices=["mean_l1", "l2"], default="mean_l1")
    parser.add_argument("--ranksim_feature_temperature", type=float, default=0.2)
    parser.add_argument("--ranksim_label_temperature", type=float, default=0.5)
    parser.add_argument("--rnc_temperature", type=float, default=2.0)
    parser.add_argument("--conr_label_width", type=float, default=0.5)
    parser.add_argument("--conr_pred_width", type=float, default=0.5)
    parser.add_argument("--conr_temperature", type=float, default=0.07)
    parser.add_argument("--conr_push_scale", type=float, default=0.01)
    parser.add_argument("--pair_margin", type=float, nargs="+", default=[0.2])
    parser.add_argument("--pair_temperature", type=float, default=0.1)
    parser.add_argument("--projector_dim", type=int, default=256)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--same_family_only", action="store_true", default=True)
    parser.add_argument("--all_batch_pairs", action="store_false", dest="same_family_only")
    parser.add_argument("--group_batches", action="store_true", default=True)
    parser.add_argument("--no_group_batches", action="store_false", dest="group_batches")
    parser.add_argument("--freeze_decoder", action="store_true")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--wandb_project", type=str, default="text2text_regression_latent_losses")
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", action="store_false", dest="bf16")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
