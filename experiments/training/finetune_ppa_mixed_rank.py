"""Fine-tune PPA generation with split-specific CE and rank losses.

This training entrypoint separates the two dataset roles in dataset_v6.2.1:

* PPA splits such as train/valid/test optimize the token-generation
  cross-entropy objective.
* Correlation splits such as test_aug_100/test_pyra_200/test_ppa_feedback
  optimize only family-aware ordering losses (RankSim, RNC, and/or ConR).

The model still generates PPA values as text, so downstream evaluation scripts
can consume the same prediction CSVs produced by the existing PPA finetuners.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from collections import defaultdict
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

from finetune_ppa import apply_lora, lora_target_modules
from finetune_ppa_latent_losses import (
    DEFAULT_METRICS,
    LatentRegularizedSeq2SeqModel,
    conr_loss,
    existing_splits,
    infer_hidden_size,
    pair_mask_from_families,
    ranksim_loss,
    rnc_loss,
)
from omnipred import T5GemmaModel, T5GemmaModelConfig
from post_finetune_ppa import (
    DEFAULT_EVAL_SPLITS,
    compute_prediction_correlations,
    compute_prediction_error_summary,
    infer_family,
    load_splits,
    metric_value,
    parse_metric_values,
    stable_design_name,
    write_csv,
    write_eval_summaries,
)
from utils.ppa_dataset import format_float_scientific


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

DEFAULT_PPA_TRAIN_SPLITS = ["train"]
DEFAULT_PPA_EVAL_SPLITS = ["valid", "test"]
DEFAULT_CORRELATION_TRAIN_SPLITS = [
    "test_aug_100",
    "test_aug_200",
    "test_pyra_200",
]
DEFAULT_CORRELATION_EVAL_SPLITS = DEFAULT_EVAL_SPLITS


def resolve_model_name_or_path(args) -> str:
    if args.base_model_dir:
        model_name_or_path = args.base_model_dir
        logger.info("Loading base model from %s", model_name_or_path)
        return model_name_or_path
    logger.info("Loading model from HF hub: %s", args.encoder_model_name)
    return args.encoder_model_name


def lora_adapter_dir(args) -> str | None:
    adapter_dir = args.eval_model_dir or args.base_model_dir
    if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        return adapter_dir
    return None


def compute_log_stats(samples: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    values = np.stack([sample["ppa_values"] for sample in samples]).astype(np.float32)
    log_values = np.log(np.maximum(values, 1e-12))
    mean = log_values.mean(axis=0)
    std = log_values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


class MixedPPARankDataset(Dataset):
    """Tokenized PPA samples with explicit PPA-vs-correlation metadata."""

    def __init__(
        self,
        records: list[dict],
        tokenizer,
        metrics: list[str],
        max_input_len: int,
        max_target_len: int,
        rtl_key: str,
        clock_key: str,
        num_digits: int,
        family_key: str | None,
        sample_type: str,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        if sample_type not in {"ppa", "correlation"}:
            raise ValueError(f"Unknown sample_type: {sample_type}")

        self.tokenizer = tokenizer
        self.metrics = metrics
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.sample_type = sample_type
        self.samples = []

        skipped = 0
        for row in records:
            rtl = row.get(rtl_key)
            if not rtl:
                skipped += 1
                continue
            try:
                clock = float(row[clock_key])
                ppa_values = [metric_value(row, metric) for metric in metrics]
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue

            target_text = "\n".join(format_float_scientific(v, num_digits) for v in ppa_values)
            family = infer_family(row, family_key)
            self.samples.append(
                {
                    "rtl": str(rtl),
                    "clock": clock,
                    "design_name": stable_design_name(row),
                    "family": family,
                    "group": f"{family}@@clock={clock:.9g}",
                    "ppa_values": np.asarray(ppa_values, dtype=np.float32),
                    "target_text": target_text,
                    "split": row.get("_split", ""),
                    "sample_type": sample_type,
                }
            )

        if max_samples is not None and len(self.samples) > max_samples:
            rng = random.Random(seed)
            self.samples = rng.sample(self.samples, max_samples)

        logger.info(
            "MixedPPARankDataset[%s]: %s valid samples (%s skipped), %s families",
            sample_type,
            len(self.samples),
            skipped,
            len({sample["family"] for sample in self.samples}),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        metrics_str = ", ".join(self.metrics)
        source_text = (
            f"Predict the {metrics_str} for clock period {sample['clock']:.1f}ns. "
            f"The RTL code is as follows:\n{sample['rtl']}\n"
        )
        model_inputs = self.tokenizer(
            source_text,
            truncation=True,
            max_length=self.max_input_len,
            padding=False,
        )
        labels = self.tokenizer(
            text_target=sample["target_text"],
            truncation=True,
            max_length=self.max_target_len - 1,
            padding=False,
        )["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            labels = labels + [self.tokenizer.eos_token_id]

        model_inputs["labels"] = labels
        model_inputs["ppa_values"] = sample["ppa_values"]
        model_inputs["family"] = sample["family"]
        model_inputs["design_name"] = sample["design_name"]
        model_inputs["clock"] = sample["clock"]
        model_inputs["sample_type"] = sample["sample_type"]
        return model_inputs


class ConcatMixedDataset(Dataset):
    def __init__(self, datasets: Iterable[MixedPPARankDataset]):
        self.datasets = [dataset for dataset in datasets if len(dataset) > 0]
        self.samples = []
        self.offsets = []
        offset = 0
        for dataset in self.datasets:
            self.offsets.append(offset)
            self.samples.extend(dataset.samples)
            offset += len(dataset)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        for offset, dataset in reversed(list(zip(self.offsets, self.datasets))):
            if idx >= offset:
                return dataset[idx - offset]
        raise IndexError(idx)


class MixedPPACollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        metadata_keys = {"ppa_values", "family", "design_name", "clock", "sample_type"}
        token_features = [{k: v for k, v in f.items() if k not in metadata_keys} for f in features]
        labels = [f.pop("labels") for f in token_features]
        batch = self.tokenizer.pad(token_features, padding=True, return_tensors="pt")

        max_label_len = max(len(label) for label in labels)
        label_tensor = torch.full((len(labels), max_label_len), -100, dtype=torch.long)
        for i, label in enumerate(labels):
            label_tensor[i, : len(label)] = torch.tensor(label, dtype=torch.long)
        batch["labels"] = label_tensor
        batch["ppa_values"] = torch.tensor(
            np.stack([f["ppa_values"] for f in features]), dtype=torch.float32
        )
        batch["family"] = [f["family"] for f in features]
        batch["design_name"] = [f["design_name"] for f in features]
        batch["clock"] = [f["clock"] for f in features]
        batch["sample_type"] = [f["sample_type"] for f in features]
        return batch


class MixedBatchSampler(Sampler[list[int]]):
    """Build pure-type batches: random PPA batches and family-grouped rank batches."""

    def __init__(self, dataset: ConcatMixedDataset, batch_size: int, shuffle: bool, seed: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.ppa_indices = []
        self.correlation_groups = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            if sample["sample_type"] == "ppa":
                self.ppa_indices.append(idx)
            elif sample["sample_type"] == "correlation":
                self.correlation_groups[sample["group"]].append(idx)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []

        ppa_indices = list(self.ppa_indices)
        if self.shuffle:
            rng.shuffle(ppa_indices)
        for start in range(0, len(ppa_indices), self.batch_size):
            batch = ppa_indices[start : start + self.batch_size]
            if batch:
                batches.append(batch)

        for indices in self.correlation_groups.values():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if batch:
                    batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        ppa_batches = math.ceil(len(self.ppa_indices) / self.batch_size)
        corr_batches = sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.correlation_groups.values()
        )
        return ppa_batches + corr_batches


class MixedPPARankTrainer(Trainer):
    def __init__(
        self,
        *args,
        tokenizer,
        train_batch_sampler: MixedBatchSampler | None,
        correlation_losses: set[str],
        ppa_ce_weight: float,
        ranksim_weight: float,
        rnc_weight: float,
        conr_weight: float,
        same_family_only: bool,
        label_distance: str,
        ranksim_feature_temperature: float,
        ranksim_label_temperature: float,
        rnc_temperature: float,
        conr_label_width: float,
        conr_pred_width: float,
        conr_temperature: float,
        conr_push_scale: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.train_batch_sampler = train_batch_sampler
        self.correlation_losses = correlation_losses
        self.ppa_ce_weight = ppa_ce_weight
        self.ranksim_weight = ranksim_weight
        self.rnc_weight = rnc_weight
        self.conr_weight = conr_weight
        self.same_family_only = same_family_only
        self.label_distance = label_distance
        self.ranksim_feature_temperature = ranksim_feature_temperature
        self.ranksim_label_temperature = ranksim_label_temperature
        self.rnc_temperature = rnc_temperature
        self.conr_label_width = conr_label_width
        self.conr_pred_width = conr_pred_width
        self.conr_temperature = conr_temperature
        self.conr_push_scale = conr_push_scale
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

    @staticmethod
    def _masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
        if not row_mask.any():
            return logits.new_tensor(0.0)
        masked_logits = logits[row_mask]
        masked_labels = labels[row_mask]
        return F.cross_entropy(
            masked_logits.reshape(-1, masked_logits.shape[-1]),
            masked_labels.reshape(-1),
            ignore_index=-100,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ppa_values = inputs["ppa_values"]
        families = inputs["family"]
        sample_types = inputs["sample_type"]
        model_inputs = {
            k: v
            for k, v in inputs.items()
            if k not in {"ppa_values", "family", "design_name", "clock", "sample_type"}
        }

        ppa_mask = torch.tensor([sample_type == "ppa" for sample_type in sample_types], device=ppa_values.device)
        if ppa_mask.any():
            outputs = model(**model_inputs, return_dict=True)
        else:
            if hasattr(model.base_model, "get_encoder"):
                encoder = model.base_model.get_encoder()
            elif hasattr(model.base_model, "get_base_model"):
                encoder = model.base_model.get_base_model().get_encoder()
            else:
                raise AttributeError("Base seq2seq model does not expose get_encoder().")
            encoder_outputs = encoder(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                return_dict=True,
            )
            outputs = SimpleNamespace(
                logits=None,
                loss=None,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            )

        pooled = model.pooled_encoder_state(outputs, model_inputs["attention_mask"])
        device = pooled.device
        ppa_mask = ppa_mask.to(device)
        corr_mask = torch.tensor([sample_type == "correlation" for sample_type in sample_types], device=device)

        ppa_ce = (
            self._masked_ce_loss(outputs.logits, model_inputs["labels"], ppa_mask)
            if ppa_mask.any()
            else pooled.new_tensor(0.0)
        )
        ranksim = pooled.new_tensor(0.0)
        rnc = pooled.new_tensor(0.0)
        conr = pooled.new_tensor(0.0)

        if corr_mask.any():
            corr_pooled = pooled[corr_mask]
            corr_targets = model.normalized_targets(ppa_values[corr_mask])
            corr_latent = model.projector(corr_pooled)
            corr_preds = model.regression_head(corr_pooled)
            corr_families = [
                family
                for family, is_correlation in zip(families, corr_mask.detach().cpu().tolist())
                if is_correlation
            ]
            pair_mask = pair_mask_from_families(
                corr_families,
                corr_pooled.device,
                self.same_family_only,
            )
            if "ranksim" in self.correlation_losses:
                ranksim = ranksim_loss(
                    corr_latent,
                    corr_targets,
                    pair_mask,
                    self.label_distance,
                    self.ranksim_feature_temperature,
                    self.ranksim_label_temperature,
                )
            if "rnc" in self.correlation_losses:
                rnc = rnc_loss(
                    corr_latent,
                    corr_targets,
                    pair_mask,
                    self.label_distance,
                    self.rnc_temperature,
                )
            if "conr" in self.correlation_losses:
                conr = conr_loss(
                    corr_latent,
                    corr_targets,
                    corr_preds,
                    pair_mask,
                    self.label_distance,
                    self.conr_label_width,
                    self.conr_pred_width,
                    self.conr_temperature,
                    self.conr_push_scale,
                )

        total = pooled.new_tensor(0.0)
        total = total + self.ppa_ce_weight * ppa_ce
        total = total + self.ranksim_weight * ranksim
        total = total + self.rnc_weight * rnc
        total = total + self.conr_weight * conr
        # Some family-grouped correlation batches are too small to form a
        # ranking loss. Keep a graph-connected zero so Trainer.backward()
        # can safely pass through those no-op batches.
        total = total + pooled.sum() * 0.0

        self._remember(loss=total, ppa_ce=ppa_ce, ranksim=ranksim, rnc=rnc, conr=conr)
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
    dataset: MixedPPARankDataset,
    tokenizer,
    split_name: str,
    output_dir: str,
    metrics: list[str],
    batch_size: int,
    device: torch.device,
    min_problem_samples: int,
    max_new_tokens: int,
    num_samples: int,
    num_samples_per_pass: int,
    greedy: bool,
) -> tuple[list[dict], list[dict]]:
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=MixedPPACollator(tokenizer))
    rows = []
    decode_failures = 0
    expected_count = len(metrics)
    if num_samples < 1:
        raise ValueError("--num_samples must be >= 1")
    if num_samples_per_pass < 1:
        raise ValueError("--eval_num_samples_per_pass must be >= 1")
    if greedy and num_samples > 1:
        logger.warning("Greedy generation ignores --num_samples > 1; using one decode per sample.")
    effective_num_samples = 1 if greedy else num_samples
    num_samples_per_pass = min(num_samples_per_pass, effective_num_samples)

    for raw_batch in tqdm(dataloader, desc=f"Generating {split_name}"):
        batch = {
            k: v.to(device)
            for k, v in raw_batch.items()
            if torch.is_tensor(v) and k not in {"ppa_values"}
        }
        gt_values = raw_batch["ppa_values"].numpy()
        batch_size_actual = raw_batch["input_ids"].shape[0]
        decoded_values_per_sample: list[list[list[float]]] = [
            [] for _ in range(batch_size_actual)
        ]

        remaining = effective_num_samples
        while remaining > 0:
            current_num_samples = min(num_samples_per_pass, remaining)
            output_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
                num_return_sequences=current_num_samples,
                do_sample=(not greedy and effective_num_samples > 1),
                pad_token_id=tokenizer.pad_token_id,
            ).view(batch_size_actual, current_num_samples, -1)

            for i, sample_output_ids in enumerate(output_ids.detach().cpu().tolist()):
                for text in tokenizer.batch_decode(sample_output_ids, skip_special_tokens=True):
                    pred_values = parse_metric_values(text, expected_count)
                    if pred_values is not None:
                        decoded_values_per_sample[i].append(pred_values)

            remaining -= current_num_samples
            del output_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for i, decoded_values in enumerate(decoded_values_per_sample):
            if not decoded_values:
                decode_failures += 1
                continue
            pred_values = [
                float(np.median([draw[metric_idx] for draw in decoded_values]))
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
    logger.info("Saved predictions to %s (%s decode failures)", prediction_path, decode_failures)

    decode_path = os.path.join(output_dir, f"decode_failures_{split_name}.csv")
    write_csv(
        decode_path,
        [{"split": split_name, "decode_failures": decode_failures, "valid_predictions": len(rows)}],
    )
    if not rows:
        logger.warning("No valid decoded predictions for %s; skipping correlation/error summaries.", split_name)
        return [], []

    corr_rows = compute_prediction_correlations(rows, metrics, min_problem_samples)
    write_csv(os.path.join(output_dir, f"problem_ordering_{split_name}.csv"), corr_rows)
    error_rows = compute_prediction_error_summary(rows, metrics, split_name)
    write_csv(os.path.join(output_dir, f"prediction_error_{split_name}.csv"), error_rows)
    return corr_rows, error_rows


def build_training_args(args) -> TrainingArguments:
    trainer_do_eval = args.do_train and args.valid_ratio > 0
    return TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        do_eval=trainer_do_eval,
        eval_strategy="steps" if trainer_do_eval else "no",
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


def make_dataset(
    records: list[dict],
    tokenizer,
    args,
    sample_type: str,
    max_samples: int | None = None,
) -> MixedPPARankDataset:
    return MixedPPARankDataset(
        records=records,
        tokenizer=tokenizer,
        metrics=args.metrics,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
        rtl_key=args.rtl_key,
        clock_key=args.clock_key,
        num_digits=args.num_digits,
        family_key=args.family_key,
        sample_type=sample_type,
        max_samples=max_samples,
        seed=args.seed,
    )


def split_records_for_validation(records: list[dict], valid_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if valid_ratio <= 0 or len(records) == 0:
        return records, []
    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)
    n_valid = max(1, int(len(records) * valid_ratio))
    return records[n_valid:], records[:n_valid]


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    unknown_losses = sorted(set(args.correlation_losses) - {"ranksim", "rnc", "conr"})
    if unknown_losses:
        raise ValueError(f"Unknown correlation loss(es): {', '.join(unknown_losses)}")
    if not args.correlation_losses:
        raise ValueError("At least one correlation loss must be enabled.")

    ppa_train_splits = existing_splits(args.dataset_dir, args.ppa_train_splits, required=True)
    corr_train_splits = existing_splits(args.dataset_dir, args.correlation_train_splits, required=True)
    ppa_eval_splits = existing_splits(args.dataset_dir, args.ppa_eval_splits, required=False)
    corr_eval_splits = existing_splits(args.dataset_dir, args.correlation_eval_splits, required=False)

    os.makedirs(args.output_dir, exist_ok=True)
    args_name = (
        "finetune_ppa_mixed_rank_args.json"
        if args.do_train
        else "finetune_ppa_mixed_rank_eval_args.json"
    )
    with open(os.path.join(args.output_dir, args_name), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    model_name_or_path = resolve_model_name_or_path(args)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, model_max_length=args.max_input_len)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = None
    eval_dataset = None
    if args.do_train:
        ppa_records = load_splits(args.dataset_dir, ppa_train_splits)
        corr_records = load_splits(args.dataset_dir, corr_train_splits)
        corr_records, corr_valid_records = split_records_for_validation(
            corr_records,
            args.valid_ratio,
            args.seed,
        )

        ppa_dataset = make_dataset(ppa_records, tokenizer, args, "ppa", max_samples=args.max_ppa_train_samples)
        corr_dataset = make_dataset(corr_records, tokenizer, args, "correlation", max_samples=args.max_correlation_train_samples)
        train_dataset = ConcatMixedDataset([ppa_dataset, corr_dataset])
        if len(train_dataset) == 0:
            raise ValueError("No valid training samples were loaded.")

        if corr_valid_records:
            eval_dataset = make_dataset(
                corr_valid_records,
                tokenizer,
                args,
                "correlation",
                max_samples=args.max_eval_samples,
            )
        log_mean, log_std = compute_log_stats(train_dataset.samples)
    else:
        log_mean = torch.zeros(len(args.metrics), dtype=torch.float32)
        log_std = torch.ones(len(args.metrics), dtype=torch.float32)
        logger.info(
            "Skipping training dataset construction because --do_train is not set."
        )
    config = T5GemmaModelConfig(
        model_name=model_name_or_path,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
    )
    t5gemma = T5GemmaModel(config=config)
    if args.use_lora:
        adapter_dir = lora_adapter_dir(args) if args.do_eval and not args.do_train else None
        if adapter_dir:
            logger.info("Loading LoRA adapter for evaluation from %s", adapter_dir)
            t5gemma.model = PeftModel.from_pretrained(t5gemma.model, adapter_dir, is_trainable=False)
        else:
            logger.info(
                "Enabling LoRA fine-tuning: r=%s alpha=%s dropout=%s targets=%s",
                args.lora_rank,
                args.lora_alpha,
                args.lora_dropout,
                lora_target_modules(args),
            )
            t5gemma = apply_lora(t5gemma, args)

    if args.freeze_decoder:
        for name, param in t5gemma.model.named_parameters():
            if "decoder" in name:
                param.requires_grad = False

    # Ensure the hidden size can be resolved before building the wrapper.
    infer_hidden_size(t5gemma.model.config)
    model = LatentRegularizedSeq2SeqModel(
        base_model=t5gemma.model,
        num_metrics=len(args.metrics),
        projector_dim=args.projector_dim,
        dropout=args.head_dropout,
        log_mean=log_mean,
        log_std=log_std,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Number of trainable parameters: %s", num_params)
    logger.info("PPA CE splits: %s", ", ".join(ppa_train_splits))
    logger.info("Correlation rank splits: %s", ", ".join(corr_train_splits))
    logger.info("Correlation losses: %s", ", ".join(args.correlation_losses))

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

    train_sampler = (
        MixedBatchSampler(train_dataset, args.train_batch_size, shuffle=True, seed=args.seed)
        if train_dataset is not None
        else None
    )
    trainer = MixedPPARankTrainer(
        model=model,
        args=build_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MixedPPACollator(tokenizer),
        tokenizer=tokenizer,
        train_batch_sampler=train_sampler,
        correlation_losses=set(args.correlation_losses),
        ppa_ce_weight=args.ppa_ce_weight,
        ranksim_weight=args.ranksim_weight,
        rnc_weight=args.rnc_weight,
        conr_weight=args.conr_weight,
        same_family_only=args.same_family_only,
        label_distance=args.label_distance,
        ranksim_feature_temperature=args.ranksim_feature_temperature,
        ranksim_label_temperature=args.ranksim_label_temperature,
        rnc_temperature=args.rnc_temperature,
        conr_label_width=args.conr_label_width,
        conr_pred_width=args.conr_pred_width,
        conr_temperature=args.conr_temperature,
        conr_push_scale=args.conr_push_scale,
    )

    if args.do_train:
        if args.resume_from_checkpoint:
            if not args.checkpoint_path:
                raise ValueError("Resuming requires an explicit --checkpoint_path.")
            trainer.train(resume_from_checkpoint=args.checkpoint_path)
        else:
            trainer.train()
        trainer.save_model(args.output_dir)

    if args.do_eval:
        trainer.optimizer = None
        trainer.lr_scheduler = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        model.to(device)
        if args.bf16 and device.type == "cuda":
            model.to(dtype=torch.bfloat16)
        all_corr_rows = []
        all_error_rows = []
        eval_plan = [(split, "ppa") for split in ppa_eval_splits] + [
            (split, "correlation") for split in corr_eval_splits
        ]
        for split, sample_type in eval_plan:
            prediction_path = os.path.join(args.output_dir, f"predictions_ground_truth_{split}.csv")
            decode_path = os.path.join(args.output_dir, f"decode_failures_{split}.csv")
            if args.skip_existing_eval and os.path.exists(prediction_path) and os.path.exists(decode_path):
                logger.info("Skipping %s because existing prediction/decode files were found.", split)
                continue
            records = load_splits(args.dataset_dir, [split])
            dataset = make_dataset(records, tokenizer, args, sample_type, max_samples=args.max_eval_samples)
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
                args.eval_max_new_tokens or args.max_target_len,
                args.num_samples,
                args.eval_num_samples_per_pass,
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
            logger.info("Wrote evaluation summaries for %s split(s).", len(eval_plan))
            print(summary)


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False, 
        description="Train PPA CE on PPA splits and rank losses on correlation splits."
    )

    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--ppa_train_splits", nargs="+", default=DEFAULT_PPA_TRAIN_SPLITS)
    parser.add_argument("--ppa_eval_splits", nargs="+", default=DEFAULT_PPA_EVAL_SPLITS)
    parser.add_argument("--correlation_train_splits", nargs="+", default=DEFAULT_CORRELATION_TRAIN_SPLITS)
    parser.add_argument("--correlation_eval_splits", nargs="+", default=DEFAULT_CORRELATION_EVAL_SPLITS)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--rtl_key", type=str, default="rtl")
    parser.add_argument("--clock_key", type=str, default="clock")
    parser.add_argument("--family_key", type=str, default=None)
    parser.add_argument("--num_digits", type=int, default=4)
    parser.add_argument("--max_input_len", type=int, default=2048)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--max_ppa_train_samples", type=int, default=None)
    parser.add_argument("--max_correlation_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--valid_ratio", type=float, default=0.0)
    parser.add_argument("--min_problem_samples", type=int, default=10)

    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--eval_model_dir", type=str, default=None)
    parser.add_argument("--encoder_model_name", type=str, default=None, help="Explicit underlying base model when loading an adapter.")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    parser.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])

    parser.add_argument("--correlation_losses", nargs="+", default=["ranksim", "rnc", "conr"])
    parser.add_argument("--ppa_ce_weight", type=float, default=1.0)
    parser.add_argument("--ranksim_weight", type=float, default=0.2)
    parser.add_argument("--rnc_weight", type=float, default=0.2)
    parser.add_argument("--conr_weight", type=float, default=0.1)
    parser.add_argument("--label_distance", choices=["mean_l1", "l2"], default="mean_l1")
    parser.add_argument("--ranksim_feature_temperature", type=float, default=0.2)
    parser.add_argument("--ranksim_label_temperature", type=float, default=0.5)
    parser.add_argument("--rnc_temperature", type=float, default=2.0)
    parser.add_argument("--conr_label_width", type=float, default=0.5)
    parser.add_argument("--conr_pred_width", type=float, default=0.5)
    parser.add_argument("--conr_temperature", type=float, default=0.07)
    parser.add_argument("--conr_push_scale", type=float, default=0.01)
    parser.add_argument("--projector_dim", type=int, default=256)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--same_family_only", action="store_true", default=True)
    parser.add_argument("--all_batch_pairs", action="store_false", dest="same_family_only")
    parser.add_argument("--freeze_decoder", action="store_true")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--wandb_project", type=str, default="text2text_regression_mixed_rank")
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--eval_num_samples_per_pass", type=int, default=1)
    parser.add_argument("--eval_max_new_tokens", type=int, default=32)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--skip_existing_eval", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", action="store_false", dest="bf16")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
