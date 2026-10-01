"""Post fine-tuning for family-aware PPA ordering.

This script starts from a model trained by ``finetune_ppa.py`` and adapts it
with the losses suggested in the accompanying note:

* pointwise regression on log PPA values,
* same-family pairwise ranking,
* continuous-label supervised contrastive learning,
* optional seq2seq CE on the original stringified labels for calibration.

The generated prediction CSVs keep the same schema as ``finetune_ppa.py`` so
``synthesis_correlation_ppa_feedback.py`` can evaluate them unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from utils.ppa_dataset import format_float_scientific


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

DEFAULT_TRAIN_SPLITS = ["test_aug_100", "test_aug_200", "test_pyra_200"]
DEFAULT_EVAL_SPLITS = [
    "test_aug_100",
    "test_deepseek_ref",
    "test_ds_verilog_eval",
    "test_ppa_feedback",
]
DEFAULT_METRICS = ["area", "delay", "total_power"]

METRIC_COL_CANDIDATES = {
    "area": ["area_value", "area"],
    "delay": ["delay_value", "delay"],
    "total_power": ["total_power_value", "total_power"],
    "static_power": ["static_power_value", "static_power"],
    "switching_power": ["switching_power_value", "switching_power"],
}


def normalize_split_name(split_name: str) -> str:
    base = os.path.basename(split_name)
    return os.path.splitext(base)[0]


def resolve_checkpoint(path: str) -> str:
    return path


def read_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_splits(dataset_dir: str, split_names: Iterable[str]) -> list[dict]:
    records = []
    for split in split_names:
        split_name = normalize_split_name(split)
        path = os.path.join(dataset_dir, f"{split_name}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Split not found: {path}")
        split_records = read_jsonl(path)
        for row in split_records:
            row = dict(row)
            row["_split"] = split_name
            records.append(row)
        logger.info("Loaded %s rows from %s", len(split_records), path)
    return records


def metric_value(row: dict, metric: str) -> float:
    for col in METRIC_COL_CANDIDATES.get(metric, [metric]):
        if col in row and row[col] is not None:
            return float(row[col])
    raise KeyError(f"Could not find metric '{metric}' in row")


def infer_family(row: dict, family_key: str | None = None) -> str:
    if family_key and family_key in row and row[family_key] is not None:
        return str(row[family_key])
    for key in ("problem", "problem_name"):
        if key in row and row[key] is not None:
            return str(row[key])
    problem_root = row.get("problem_root")
    if problem_root:
        parts = str(problem_root).split("/")
        if len(parts) >= 3:
            return parts[-3]
        return str(problem_root)
    design_name = str(row.get("design_name") or row.get("Name") or "")
    if "__" in design_name:
        return design_name.split("__", 1)[0]
    if "_Gen" in design_name:
        return design_name.split("_Gen", 1)[0]
    return design_name or "unknown_family"


def stable_design_name(row: dict) -> str:
    return str(row.get("design_name") or row.get("Name") or row.get("verilog_id") or "")


@dataclass
class PPAStats:
    mean: torch.Tensor
    std: torch.Tensor

    def to(self, device: torch.device) -> "PPAStats":
        return PPAStats(self.mean.to(device), self.std.to(device))


def compute_log_stats(values: np.ndarray) -> PPAStats:
    log_values = np.log(np.maximum(values, 1e-12))
    mean = log_values.mean(axis=0)
    std = log_values.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return PPAStats(
        mean=torch.tensor(mean, dtype=torch.float32),
        std=torch.tensor(std, dtype=torch.float32),
    )


class PPAPostTuneDataset(Dataset):
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
        max_samples: int | None = None,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.metrics = metrics
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
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
                }
            )

        if max_samples is not None and len(self.samples) > max_samples:
            rng = random.Random(seed)
            self.samples = rng.sample(self.samples, max_samples)

        logger.info(
            "PPAPostTuneDataset: %s valid samples (%s skipped), %s families",
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
        return model_inputs


class PPACollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        metadata_keys = {"ppa_values", "family", "design_name", "clock"}
        token_features = [{k: v for k, v in f.items() if k not in metadata_keys} for f in features]
        labels = [f.pop("labels") for f in token_features]
        batch = self.tokenizer.pad(token_features, padding=True, return_tensors="pt")

        max_label_len = max(len(label) for label in labels)
        label_tensor = torch.full(
            (len(labels), max_label_len), -100, dtype=torch.long
        )
        for i, label in enumerate(labels):
            label_tensor[i, : len(label)] = torch.tensor(label, dtype=torch.long)
        batch["labels"] = label_tensor
        batch["ppa_values"] = torch.tensor(
            np.stack([f["ppa_values"] for f in features]), dtype=torch.float32
        )
        batch["family"] = [f["family"] for f in features]
        batch["design_name"] = [f["design_name"] for f in features]
        batch["clock"] = [f["clock"] for f in features]
        return batch


class FamilyBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: PPAPostTuneDataset, batch_size: int, shuffle: bool, seed: int):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.groups = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            self.groups[sample["group"]].append(idx)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for indices in self.groups.values():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) > 0:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.groups.values())


class HybridPPAModel(nn.Module):
    def __init__(
        self,
        base_model,
        num_metrics: int,
        stats: PPAStats,
        projector_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.base_model = base_model
        hidden_size = getattr(base_model.config, "d_model", None)
        if hidden_size is None:
            hidden_size = getattr(base_model.config, "hidden_size", None)
        if hidden_size is None and getattr(base_model.config, "encoder", None) is not None:
            hidden_size = getattr(base_model.config.encoder, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer backbone hidden size from model config.")
        self.regression_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_metrics),
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projector_dim),
            nn.GELU(),
            nn.Linear(projector_dim, projector_dim),
        )
        self.register_buffer("log_mean", stats.mean.clone())
        self.register_buffer("log_std", stats.std.clone())

    def encode_pool(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels=None):
        if labels is None:
            encoder_outputs = self.base_model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            outputs = SimpleNamespace(loss=None, encoder_last_hidden_state=encoder_outputs.last_hidden_state)
        else:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
                return_dict=True,
            )
        hidden = outputs.encoder_last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return outputs, pooled

    def normalized_targets(self, ppa_values: torch.Tensor) -> torch.Tensor:
        log_values = torch.log(ppa_values.clamp_min(1e-12))
        return (log_values - self.log_mean) / self.log_std

    def predict_values(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _, pooled = self.encode_pool(input_ids, attention_mask, labels=None)
        pred_norm = self.regression_head(pooled)
        pred_log = pred_norm * self.log_std + self.log_mean
        return torch.exp(pred_log).clamp_min(0.0)

    def save_pretrained(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.base_model.save_pretrained(output_dir, safe_serialization=True)
        torch.save(
            {
                "regression_head": self.regression_head.state_dict(),
                "projector": self.projector.state_dict(),
                "log_mean": self.log_mean.detach().cpu(),
                "log_std": self.log_std.detach().cpu(),
            },
            os.path.join(output_dir, "ppa_hybrid_heads.pt"),
        )

    def load_heads(self, heads_path: str):
        state = torch.load(heads_path, map_location="cpu")
        self.regression_head.load_state_dict(state["regression_head"])
        self.projector.load_state_dict(state["projector"])
        self.log_mean.copy_(state["log_mean"])
        self.log_std.copy_(state["log_std"])


def pairwise_ranking_loss(pred_norm: torch.Tensor, target_norm: torch.Tensor, margin_temp: float) -> torch.Tensor:
    losses = []
    for metric_idx in range(target_norm.shape[1]):
        target = target_norm[:, metric_idx]
        pred = pred_norm[:, metric_idx]
        target_diff = target.unsqueeze(1) - target.unsqueeze(0)
        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
        valid = target_diff.abs() > 1e-6
        if not valid.any():
            continue
        direction = target_diff.sign()
        weights = target_diff.abs().detach()
        losses.append((F.softplus(-direction * pred_diff / margin_temp) * weights)[valid].mean())
    if not losses:
        return pred_norm.new_tensor(0.0)
    return torch.stack(losses).mean()


def continuous_contrastive_loss(
    embeddings: torch.Tensor,
    target_norm: torch.Tensor,
    temperature: float,
    label_sigma: float,
) -> torch.Tensor:
    if embeddings.shape[0] < 3:
        return embeddings.new_tensor(0.0)
    z = F.normalize(embeddings, dim=-1)
    logits = z @ z.T / temperature
    eye = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
    scalar_target = target_norm.mean(dim=1)
    distances = (scalar_target.unsqueeze(1) - scalar_target.unsqueeze(0)).abs()
    target_weights = torch.exp(-distances / label_sigma).masked_fill(eye, 0.0)
    row_sums = target_weights.sum(dim=1, keepdim=True)
    valid_rows = row_sums.squeeze(1) > 1e-8
    if not valid_rows.any():
        return embeddings.new_tensor(0.0)
    target_probs = target_weights / row_sums.clamp_min(1e-8)
    log_probs = F.log_softmax(logits.masked_fill(eye, -1e4), dim=1)
    return -(target_probs[valid_rows] * log_probs[valid_rows]).sum(dim=1).mean()


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def train_one_epoch(
    model: HybridPPAModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    args,
    epoch: int,
) -> dict:
    model.train()
    totals = defaultdict(float)
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for step, raw_batch in enumerate(progress):
        batch = move_batch_to_device(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
            outputs, pooled = model.encode_pool(
                batch["input_ids"],
                batch["attention_mask"],
                labels=batch["labels"] if args.ce_weight > 0 else None,
            )
            pred_norm = model.regression_head(pooled)
            target_norm = model.normalized_targets(batch["ppa_values"])
            projector = model.projector(pooled)

            ce_loss = outputs.loss if args.ce_weight > 0 else pred_norm.new_tensor(0.0)
            regression_loss = F.smooth_l1_loss(pred_norm, target_norm)
            rank_loss = pairwise_ranking_loss(pred_norm, target_norm, args.rank_temperature)
            contrastive_loss = continuous_contrastive_loss(
                projector, target_norm, args.contrastive_temperature, args.contrastive_label_sigma
            )
            loss = (
                args.ce_weight * ce_loss
                + args.regression_weight * regression_loss
                + args.rank_weight * rank_loss
                + args.contrastive_weight * contrastive_loss
            )
            scaled_loss = loss / args.gradient_accumulation_steps

        scaled_loss.backward()
        if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        totals["loss"] += float(loss.detach().cpu())
        totals["ce"] += float(ce_loss.detach().cpu())
        totals["regression"] += float(regression_loss.detach().cpu())
        totals["rank"] += float(rank_loss.detach().cpu())
        totals["contrastive"] += float(contrastive_loss.detach().cpu())
        progress.set_postfix({k: f"{v / (step + 1):.4f}" for k, v in totals.items()})

    return {k: v / max(len(dataloader), 1) for k, v in totals.items()}


def has_constant(values: np.ndarray) -> bool:
    return len(values) == 0 or bool(np.all(np.isclose(values, values[0], rtol=0.0, atol=1e-12)))


def compute_prediction_correlations(rows: list[dict], metrics: list[str], min_problem_samples: int) -> list[dict]:
    results = []
    by_clock = defaultdict(list)
    for row in rows:
        by_clock[round(float(row["clock"]), 9)].append(row)

    for clock, clock_rows in sorted(by_clock.items()):
        for metric in metrics:
            gt_key = f"gt_{metric}"
            pred_key = f"pred_{metric}"
            grouped = defaultdict(list)
            for row in clock_rows:
                grouped[row["problem_name"]].append(row)

            prob_spearman = []
            prob_kendall = []
            for family, family_rows in grouped.items():
                if len(family_rows) < min_problem_samples:
                    continue
                gt = np.array([float(r[gt_key]) for r in family_rows], dtype=np.float64)
                pred = np.array([float(r[pred_key]) for r in family_rows], dtype=np.float64)
                valid = ~(np.isnan(gt) | np.isnan(pred))
                gt = gt[valid]
                pred = pred[valid]
                if len(gt) < min_problem_samples or has_constant(gt) or has_constant(pred):
                    continue
                sp = spearmanr(pred, gt).statistic
                kt = kendalltau(pred, gt).statistic
                if not np.isnan(sp):
                    prob_spearman.append(sp)
                if not np.isnan(kt):
                    prob_kendall.append(kt)

            results.append(
                {
                    "clock": clock,
                    "metric": metric,
                    "n_samples": len(clock_rows),
                    "n_problem_spearman": len(prob_spearman),
                    "avg_prob_spearman_r": float(np.mean(prob_spearman)) if prob_spearman else float("nan"),
                    "n_problem_kendall": len(prob_kendall),
                    "avg_prob_kendall_tau": float(np.mean(prob_kendall)) if prob_kendall else float("nan"),
                }
            )
    return results


def summarize_errors(values: np.ndarray, gt_values: np.ndarray) -> dict:
    values = values.astype(np.float64)
    gt_values = gt_values.astype(np.float64)
    valid = ~(np.isnan(values) | np.isnan(gt_values))
    values = values[valid]
    gt_values = gt_values[valid]
    if len(values) == 0:
        return {
            "n": 0,
            "avg_relative_mae": float("nan"),
            "median_relative_error": float("nan"),
            "p90_relative_error": float("nan"),
            "rmse_log": float("nan"),
            "mean_pred_gt_ratio": float("nan"),
            "median_pred_gt_ratio": float("nan"),
            "acc_at_5_percent": float("nan"),
            "acc_at_10_percent": float("nan"),
            "acc_at_20_percent": float("nan"),
            "acc_at_50_percent": float("nan"),
            "mean_gt": float("nan"),
            "mean_pred": float("nan"),
        }

    rel_error = np.abs(values - gt_values) / (np.abs(gt_values) + 1e-10)
    safe_values = np.maximum(values, 1e-12)
    safe_gt = np.maximum(gt_values, 1e-12)
    log_error = np.log(safe_values) - np.log(safe_gt)
    pred_gt_ratio = values / (gt_values + 1e-10)
    return {
        "n": int(len(values)),
        "avg_relative_mae": float(np.mean(rel_error)),
        "median_relative_error": float(np.median(rel_error)),
        "p90_relative_error": float(np.quantile(rel_error, 0.90)),
        "rmse_log": float(np.sqrt(np.mean(log_error ** 2))),
        "mean_pred_gt_ratio": float(np.mean(pred_gt_ratio)),
        "median_pred_gt_ratio": float(np.median(pred_gt_ratio)),
        "acc_at_5_percent": round(float(np.mean(rel_error < 0.05)) * 100, 2),
        "acc_at_10_percent": round(float(np.mean(rel_error < 0.10)) * 100, 2),
        "acc_at_20_percent": round(float(np.mean(rel_error < 0.20)) * 100, 2),
        "acc_at_50_percent": round(float(np.mean(rel_error < 0.50)) * 100, 2),
        "mean_gt": float(np.mean(gt_values)),
        "mean_pred": float(np.mean(values)),
    }


def compute_prediction_error_summary(rows: list[dict], metrics: list[str], split_name: str) -> list[dict]:
    summary_rows = []
    all_pred_cols = []
    all_gt_cols = []
    per_sample_rel_errors = []

    for metric in metrics:
        gt = np.array([float(row[f"gt_{metric}"]) for row in rows], dtype=np.float64)
        pred = np.array([float(row[f"pred_{metric}"]) for row in rows], dtype=np.float64)
        summary = summarize_errors(pred, gt)
        summary_rows.append({"split": split_name, "metric": metric, **summary})
        all_pred_cols.append(pred)
        all_gt_cols.append(gt)
        per_sample_rel_errors.append(np.abs(pred - gt) / (np.abs(gt) + 1e-10))

    if all_pred_cols:
        all_pred = np.concatenate(all_pred_cols)
        all_gt = np.concatenate(all_gt_cols)
        summary_rows.append(
            {
                "split": split_name,
                "metric": "__all_values__",
                **summarize_errors(all_pred, all_gt),
            }
        )

    if per_sample_rel_errors:
        max_rel_error = np.max(np.stack(per_sample_rel_errors, axis=1), axis=1)
        summary_rows.append(
            {
                "split": split_name,
                "metric": "__all_metrics_per_sample__",
                "n": int(len(max_rel_error)),
                "avg_relative_mae": float(np.mean(max_rel_error)),
                "median_relative_error": float(np.median(max_rel_error)),
                "p90_relative_error": float(np.quantile(max_rel_error, 0.90)),
                "rmse_log": float("nan"),
                "mean_pred_gt_ratio": float("nan"),
                "median_pred_gt_ratio": float("nan"),
                "acc_at_5_percent": round(float(np.mean(max_rel_error < 0.05)) * 100, 2),
                "acc_at_10_percent": round(float(np.mean(max_rel_error < 0.10)) * 100, 2),
                "acc_at_20_percent": round(float(np.mean(max_rel_error < 0.20)) * 100, 2),
                "acc_at_50_percent": round(float(np.mean(max_rel_error < 0.50)) * 100, 2),
                "mean_gt": float("nan"),
                "mean_pred": float("nan"),
            }
        )

    return summary_rows


def write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_metric_values(text: str, expected_count: int) -> list[float] | None:
    values = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if len(values) < expected_count:
        return None
    try:
        return [float(value) for value in values[:expected_count]]
    except ValueError:
        return None


def write_eval_summaries(output_dir: str, corr_rows: list[dict], error_rows: list[dict]):
    summary_path = os.path.join(output_dir, "problem_ordering_summary.csv")
    write_csv(summary_path, corr_rows)
    error_summary_path = os.path.join(output_dir, "prediction_error_summary.csv")
    write_csv(error_summary_path, error_rows)
    if corr_rows:
        finite_sp = [r["avg_prob_spearman_r"] for r in corr_rows if not np.isnan(r["avg_prob_spearman_r"])]
        finite_kt = [r["avg_prob_kendall_tau"] for r in corr_rows if not np.isnan(r["avg_prob_kendall_tau"])]
        logger.info(
            "Problem-ordering summary: mean spearman=%.4f, mean kendall=%.4f",
            float(np.mean(finite_sp)) if finite_sp else float("nan"),
            float(np.mean(finite_kt)) if finite_kt else float("nan"),
        )


@torch.no_grad()
def evaluate_split(
    model: HybridPPAModel,
    dataset: PPAPostTuneDataset,
    tokenizer,
    split_name: str,
    output_dir: str,
    metrics: list[str],
    batch_size: int,
    device: torch.device,
    min_problem_samples: int,
) -> tuple[list[dict], list[dict]]:
    model.eval()
    collator = PPACollator(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    rows = []
    for raw_batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
        batch = move_batch_to_device(raw_batch, device)
        pred_values = model.predict_values(batch["input_ids"], batch["attention_mask"]).float().cpu().numpy()
        gt_values = raw_batch["ppa_values"].numpy()
        for i in range(pred_values.shape[0]):
            row = {
                "design_name": raw_batch["design_name"][i],
                "problem_name": raw_batch["family"][i],
                "clock": raw_batch["clock"][i],
            }
            for metric_idx, metric in enumerate(metrics):
                row[f"gt_{metric}"] = float(gt_values[i, metric_idx])
                row[f"pred_{metric}"] = float(pred_values[i, metric_idx])
            rows.append(row)

    prediction_path = os.path.join(output_dir, f"predictions_ground_truth_{split_name}.csv")
    write_csv(prediction_path, rows)
    logger.info("Saved predictions to %s", prediction_path)

    corr_rows = compute_prediction_correlations(rows, metrics, min_problem_samples)
    corr_path = os.path.join(output_dir, f"problem_ordering_{split_name}.csv")
    write_csv(corr_path, corr_rows)
    logger.info("Saved problem ordering metrics to %s", corr_path)

    error_rows = compute_prediction_error_summary(rows, metrics, split_name)
    error_path = os.path.join(output_dir, f"prediction_error_{split_name}.csv")
    write_csv(error_path, error_rows)
    logger.info("Saved prediction error summary to %s", error_path)

    all_values = next((row for row in error_rows if row["metric"] == "__all_values__"), None)
    all_metrics = next((row for row in error_rows if row["metric"] == "__all_metrics_per_sample__"), None)
    if all_values and all_metrics:
        logger.info(
            "%s deviation: value avg_rel_mae=%.4f, value acc@20=%.2f%%, "
            "all-metrics acc@20=%.2f%%, median pred/gt=%.4f",
            split_name,
            all_values["avg_relative_mae"],
            all_values["acc_at_20_percent"],
            all_metrics["acc_at_20_percent"],
            all_values["median_pred_gt_ratio"],
        )
    return corr_rows, error_rows


@torch.no_grad()
def evaluate_lm_split(
    model: nn.Module,
    dataset: PPAPostTuneDataset,
    tokenizer,
    split_name: str,
    output_dir: str,
    metrics: list[str],
    batch_size: int,
    device: torch.device,
    min_problem_samples: int,
    max_new_tokens: int,
) -> tuple[list[dict], list[dict]]:
    model.eval()
    collator = PPACollator(tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    rows = []
    decode_failures = 0
    expected_count = len(metrics)

    for raw_batch in tqdm(dataloader, desc=f"Evaluating {split_name} (lm_head)"):
        batch = move_batch_to_device(raw_batch, device)
        output_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        decoded = tokenizer.batch_decode(output_ids.detach().cpu().tolist(), skip_special_tokens=True)
        gt_values = raw_batch["ppa_values"].numpy()

        for i, text in enumerate(decoded):
            pred_values = parse_metric_values(text, expected_count)
            if pred_values is None:
                decode_failures += 1
                continue
            row = {
                "design_name": raw_batch["design_name"][i],
                "problem_name": raw_batch["family"][i],
                "clock": raw_batch["clock"][i],
                "decoded_text": text,
            }
            for metric_idx, metric in enumerate(metrics):
                row[f"gt_{metric}"] = float(gt_values[i, metric_idx])
                row[f"pred_{metric}"] = float(pred_values[metric_idx])
            rows.append(row)

    prediction_path = os.path.join(output_dir, f"predictions_ground_truth_{split_name}.csv")
    write_csv(prediction_path, rows)
    logger.info("Saved LM-head predictions to %s (%s decode failures)", prediction_path, decode_failures)

    corr_rows = compute_prediction_correlations(rows, metrics, min_problem_samples)
    corr_path = os.path.join(output_dir, f"problem_ordering_{split_name}.csv")
    write_csv(corr_path, corr_rows)
    logger.info("Saved problem ordering metrics to %s", corr_path)

    error_rows = compute_prediction_error_summary(rows, metrics, split_name)
    error_path = os.path.join(output_dir, f"prediction_error_{split_name}.csv")
    write_csv(error_path, error_rows)
    logger.info("Saved prediction error summary to %s", error_path)

    write_csv(
        os.path.join(output_dir, f"lm_decode_failures_{split_name}.csv"),
        [{"split": split_name, "decode_failures": decode_failures, "valid_predictions": len(rows)}],
    )
    return corr_rows, error_rows


def build_optimizer(model: HybridPPAModel, args):
    if args.freeze_lm:
        for param in model.base_model.parameters():
            param.requires_grad = False
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(grouped, lr=args.learning_rate)


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_splits = [normalize_split_name(s) for s in args.train_splits]
    eval_splits = [normalize_split_name(s) for s in args.eval_splits]

    checkpoint = resolve_checkpoint(args.base_model_dir)
    logger.info("Loading base model from %s", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=args.max_input_len)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "post_finetune_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if args.do_eval and not args.do_train and args.eval_prediction_mode == "lm_head":
        base_model.to(device)
        all_corr_rows = []
        all_error_rows = []
        for split in eval_splits:
            records = load_splits(args.dataset_dir, [split])
            dataset = PPAPostTuneDataset(
                records,
                tokenizer,
                args.metrics,
                args.max_input_len,
                args.max_target_len,
                args.rtl_key,
                args.clock_key,
                args.num_digits,
                args.family_key,
                max_samples=args.max_eval_samples,
                seed=args.seed,
            )
            corr_rows, error_rows = evaluate_lm_split(
                base_model,
                dataset,
                tokenizer,
                split,
                args.output_dir,
                args.metrics,
                args.eval_batch_size,
                device,
                args.min_problem_samples,
                args.max_target_len,
            )
            for row in corr_rows:
                row = dict(row)
                row["split"] = split
                all_corr_rows.append(row)
            all_error_rows.extend(error_rows)
        write_eval_summaries(args.output_dir, all_corr_rows, all_error_rows)
        return

    train_records = load_splits(args.dataset_dir, train_splits)
    rng = random.Random(args.seed)
    rng.shuffle(train_records)
    if args.valid_ratio > 0:
        n_valid = max(1, int(len(train_records) * args.valid_ratio))
        valid_records = train_records[:n_valid]
        train_records = train_records[n_valid:]
    else:
        valid_records = []

    train_dataset = PPAPostTuneDataset(
        train_records,
        tokenizer,
        args.metrics,
        args.max_input_len,
        args.max_target_len,
        args.rtl_key,
        args.clock_key,
        args.num_digits,
        args.family_key,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    valid_dataset = PPAPostTuneDataset(
        valid_records,
        tokenizer,
        args.metrics,
        args.max_input_len,
        args.max_target_len,
        args.rtl_key,
        args.clock_key,
        args.num_digits,
        args.family_key,
        max_samples=args.max_eval_samples,
        seed=args.seed,
    ) if valid_records else None

    if len(train_dataset) == 0:
        raise ValueError("No valid training samples were loaded.")

    train_values = np.stack([sample["ppa_values"] for sample in train_dataset.samples])
    stats = compute_log_stats(train_values)
    model = HybridPPAModel(
        base_model=base_model,
        num_metrics=len(args.metrics),
        stats=stats,
        projector_dim=args.projector_dim,
        dropout=args.head_dropout,
    )
    heads_path = os.path.join(args.base_model_dir, "ppa_hybrid_heads.pt")
    if args.do_eval and not args.do_train:
        if not os.path.exists(heads_path):
            raise FileNotFoundError("Regression-head evaluation requires ppa_hybrid_heads.pt in the exact --base_model_dir checkpoint.")
        logger.info("Loading PPA hybrid heads from %s", heads_path)
        model.load_heads(heads_path)

    model.to(device)
    optimizer = build_optimizer(model, args)

    sampler = FamilyBatchSampler(train_dataset, args.train_batch_size, shuffle=True, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=PPACollator(tokenizer))
    total_update_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.num_train_epochs
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    if args.do_train:
        for epoch in range(args.num_train_epochs):
            sampler.set_epoch(epoch)
            epoch_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device, args, epoch)
            logger.info("Epoch %s metrics: %s", epoch + 1, epoch_metrics)
            model.save_pretrained(os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, f"checkpoint-epoch-{epoch + 1}"))
            if valid_dataset is not None and args.eval_during_train:
                evaluate_split(
                    model,
                    valid_dataset,
                    tokenizer,
                    "valid_adaptation",
                    args.output_dir,
                    args.metrics,
                    args.eval_batch_size,
                    device,
                    args.min_problem_samples,
                )
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    if args.do_eval:
        all_corr_rows = []
        all_error_rows = []
        for split in eval_splits:
            records = load_splits(args.dataset_dir, [split])
            dataset = PPAPostTuneDataset(
                records,
                tokenizer,
                args.metrics,
                args.max_input_len,
                args.max_target_len,
                args.rtl_key,
                args.clock_key,
                args.num_digits,
                args.family_key,
                max_samples=args.max_eval_samples,
                seed=args.seed,
            )
            if args.eval_prediction_mode == "lm_head":
                corr_rows, error_rows = evaluate_lm_split(
                    model.base_model,
                    dataset,
                    tokenizer,
                    split,
                    args.output_dir,
                    args.metrics,
                    args.eval_batch_size,
                    device,
                    args.min_problem_samples,
                    args.max_target_len,
                )
            else:
                corr_rows, error_rows = evaluate_split(
                    model,
                    dataset,
                    tokenizer,
                    split,
                    args.output_dir,
                    args.metrics,
                    args.eval_batch_size,
                    device,
                    args.min_problem_samples,
                )
            for row in corr_rows:
                row = dict(row)
                row["split"] = split
                all_corr_rows.append(row)
            all_error_rows.extend(error_rows)

        write_eval_summaries(args.output_dir, all_corr_rows, all_error_rows)


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False, description="Family-aware post-finetuning for PPA ordering.")
    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
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

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--projector_dim", type=int, default=256)
    parser.add_argument(
        "--eval_prediction_mode",
        type=str,
        choices=["regression_head", "lm_head"],
        default="regression_head",
        help="Use the post-finetune regression head or the seq2seq LM head for evaluation.",
    )

    # losses and parameters
    parser.add_argument("--ce_weight", type=float, default=0.2)
    parser.add_argument("--regression_weight", type=float, default=1.0)
    parser.add_argument("--rank_weight", type=float, default=1.0)
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--rank_temperature", type=float, default=1.0)
    parser.add_argument("--contrastive_temperature", type=float, default=0.2)
    parser.add_argument("--contrastive_label_sigma", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", action="store_false", dest="bf16")
    parser.add_argument("--freeze_lm", action="store_true")
    parser.add_argument("--eval_during_train", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
