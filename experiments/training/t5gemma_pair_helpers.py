"""Matching historical pair-dataset/trainer helpers for the T5Gemma variant.

Definitions preserve the recovered weighted-pair implementation. The separate
generic pair entrypoint retains its earlier implementation.
"""

from __future__ import annotations

import argparse

import csv

import json

import logging

import math

import os

import random

from collections import Counter, defaultdict

from dataclasses import dataclass

from typing import Iterable

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from peft import LoraConfig, PeftModel, get_peft_model

from scipy.stats import kendalltau, spearmanr

from torch.utils.data import ConcatDataset, DataLoader, Dataset

from tqdm.auto import tqdm

from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments


from post_finetune_ppa import infer_family, metric_value, normalize_split_name, stable_design_name

from experimental_helpers import get_backbone_hidden_size, resolve_lora_target_modules

from experimental_helpers import clock_filter

logger = logging.getLogger(__name__)

DEFAULT_METRICS = ["area", "delay", "total_power"]

DEFAULT_EASY_TRAIN_SPLITS = ["train"]

DEFAULT_HARD_TRAIN_SPLITS = ["test_aug_100", "test_aug_200", "test_pyra_200"]

DEFAULT_RANDOM_EVAL_SPLITS = ["test", "test_pluto", "test_realbench", "test_rtl_opt"]

DEFAULT_FAMILY_EVAL_SPLITS = ["test_aug_100", "test_ds_verilog_eval", "test_ppa_feedback"]

CLASS_NAMES = ["less", "almost", "greater"]

SAMPLE_KIND_EASY = 0

SAMPLE_KIND_HARD = 1

@dataclass
class SampleWeightSchedule:
    schedule: str
    easy_start: float
    easy_end: float
    hard_start: float
    hard_end: float

    @property
    def enabled(self) -> bool:
        return self.schedule != "constant"

    def weights(self, step: int, max_steps: int) -> tuple[float, float]:
        if self.schedule == "constant":
            return self.easy_start, self.hard_start

        progress = min(max(step / max(max_steps, 1), 0.0), 1.0)
        if self.schedule == "cosine":
            progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        elif self.schedule != "linear":
            raise ValueError(f"Unknown sample weight schedule: {self.schedule}")

        easy_weight = self.easy_start + (self.easy_end - self.easy_start) * progress
        hard_weight = self.hard_start + (self.hard_end - self.hard_start) * progress
        return easy_weight, hard_weight

def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def split_path(dataset_dir: str, split_name: str) -> str:
    return os.path.join(dataset_dir, f"{normalize_split_name(split_name)}.jsonl")

def existing_splits(dataset_dir: str, split_names: Iterable[str], required: bool) -> list[str]:
    found = []
    missing = []
    for split in split_names:
        split_name = normalize_split_name(split)
        if split_name.lower() in {"", "none", "null", "skip"}:
            continue
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

def load_splits(dataset_dir: str, split_names: Iterable[str]) -> list[dict]:
    records = []
    for split in split_names:
        split_name = normalize_split_name(split)
        rows = read_jsonl(split_path(dataset_dir, split_name))
        for row in rows:
            row = dict(row)
            row["_split"] = split_name
            records.append(row)
        logger.info("Loaded %s rows from split %s", len(rows), split_name)
    return records

@dataclass(frozen=True)
class PPASample:
    rtl: str
    clock: float
    values: np.ndarray
    family: str
    group: str
    design_name: str
    split: str

def build_samples(
    records: Iterable[dict],
    metrics: list[str],
    rtl_key: str,
    clock_key: str,
    family_key: str | None,
    clock: float | None,
) -> list[PPASample]:
    samples = []
    skipped = 0
    for row in records:
        if clock is not None and not clock_filter(row, clock):
            continue
        rtl = row.get(rtl_key)
        if not rtl:
            skipped += 1
            continue
        try:
            clock_value = float(row[clock_key])
            values = np.asarray([metric_value(row, metric) for metric in metrics], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        family = infer_family(row, family_key)
        samples.append(
            PPASample(
                rtl=str(rtl),
                clock=clock_value,
                values=values,
                family=family,
                group=f"{family}@@clock={clock_value:.9g}",
                design_name=stable_design_name(row),
                split=str(row.get("_split", "")),
            )
        )
    logger.info("Built %s valid samples (%s skipped)", len(samples), skipped)
    return samples

def limit_samples(
    samples: list[PPASample],
    max_samples: int | None,
    seed: int,
    strategy: str,
) -> list[PPASample]:
    if max_samples is None or len(samples) <= max_samples:
        return samples
    if strategy == "first":
        return samples[:max_samples]
    if strategy == "random":
        rng = random.Random(seed)
        return rng.sample(samples, max_samples)
    raise ValueError(f"Unknown sample limit strategy: {strategy}")

def format_prompt(sample: PPASample) -> str:
    return f"# Clock: {sample.clock:.9g}ns\n# RTL:\n{sample.rtl}"

def comparison_labels(lhs: np.ndarray, rhs: np.ndarray, margin: float) -> np.ndarray:
    lhs, rhs = np.broadcast_arrays(lhs, rhs)
    labels = np.full(lhs.shape, 1, dtype=np.int64)
    denom = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1e-12)
    rel_delta = (lhs - rhs) / denom
    labels[rel_delta > margin] = 2
    labels[rel_delta < -margin] = 0
    return labels

def family_pair_indices(samples: list[PPASample]) -> list[tuple[int, int]]:
    grouped = defaultdict(list)
    for idx, sample in enumerate(samples):
        grouped[sample.group].append(idx)
    pairs = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        for lhs in indices:
            for rhs in indices:
                if lhs != rhs:
                    pairs.append((lhs, rhs))
    return pairs

class PairwisePPADataset(Dataset):
    def __init__(
        self,
        samples: list[PPASample],
        tokenizer,
        max_input_len: int,
        margin: float,
        num_pairs: int,
        pair_mode: str,
        seed: int,
        sample_weight: float = 1.0,
        sample_kind: int = SAMPLE_KIND_EASY,
    ):
        if pair_mode not in {"random", "family", "family_exhaustive"}:
            raise ValueError(f"Unknown pair_mode: {pair_mode}")
        if len(samples) < 2:
            raise ValueError("PairwisePPADataset needs at least two valid samples.")
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.margin = margin
        self.num_pairs = num_pairs
        self.pair_mode = pair_mode
        self.seed = seed
        self.sample_weight = sample_weight
        self.sample_kind = sample_kind
        self.pair_indices: list[tuple[int, int]] = []

        self.groups: list[list[int]] = []
        if pair_mode in {"family", "family_exhaustive"}:
            by_group = defaultdict(list)
            for idx, sample in enumerate(samples):
                by_group[sample.group].append(idx)
            self.groups = [indices for indices in by_group.values() if len(indices) >= 2]
            if not self.groups:
                raise ValueError("No family/clock groups with at least two samples.")
            if pair_mode == "family_exhaustive":
                self.pair_indices = family_pair_indices(samples)
                if not self.pair_indices:
                    raise ValueError("No exhaustive family pairs could be built.")
                if self.num_pairs <= 0:
                    self.num_pairs = len(self.pair_indices)
            logger.info(
                "Family pair sampler: %s groups, mean group size %.1f, mode=%s",
                len(self.groups),
                sum(len(g) for g in self.groups) / len(self.groups),
                pair_mode,
            )

    def __len__(self) -> int:
        return self.num_pairs

    def _choose_pair(self, idx: int) -> tuple[int, int]:
        if self.pair_mode == "family_exhaustive":
            return self.pair_indices[idx % len(self.pair_indices)]

        rng = random.Random(self.seed + idx)
        if self.pair_mode == "random":
            lhs = rng.randrange(len(self.samples))
            rhs = rng.randrange(len(self.samples) - 1)
            if rhs >= lhs:
                rhs += 1
            return lhs, rhs

        group = self.groups[rng.randrange(len(self.groups))]
        lhs_pos = rng.randrange(len(group))
        rhs_pos = rng.randrange(len(group) - 1)
        if rhs_pos >= lhs_pos:
            rhs_pos += 1
        return group[lhs_pos], group[rhs_pos]

    def __getitem__(self, idx: int) -> dict:
        lhs_idx, rhs_idx = self._choose_pair(idx)
        lhs = self.samples[lhs_idx]
        rhs = self.samples[rhs_idx]
        left = self.tokenizer(
            format_prompt(lhs),
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
        )
        right = self.tokenizer(
            format_prompt(rhs),
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
        )
        labels = comparison_labels(lhs.values, rhs.values, self.margin)
        return {
            "input_ids_a": left["input_ids"],
            "attention_mask_a": left["attention_mask"],
            "input_ids_b": right["input_ids"],
            "attention_mask_b": right["attention_mask"],
            "labels": labels,
            "sample_weight": self.sample_weight,
            "sample_kind": self.sample_kind,
        }

class PairwiseEmbeddingDataset(Dataset):
    def __init__(
        self,
        samples: list[PPASample],
        embeddings: torch.Tensor,
        margin: float,
        num_pairs: int,
        pair_mode: str,
        seed: int,
        sample_weight: float = 1.0,
        sample_kind: int = SAMPLE_KIND_EASY,
    ):
        self.pair_dataset = PairwisePPADataset(
            samples=samples,
            tokenizer=None,
            max_input_len=0,
            margin=margin,
            num_pairs=num_pairs,
            pair_mode=pair_mode,
            seed=seed,
            sample_weight=sample_weight,
            sample_kind=sample_kind,
        )
        self.samples = samples
        self.embeddings = embeddings.float().cpu()
        self.margin = margin
        self.sample_weight = sample_weight
        self.sample_kind = sample_kind

    def __len__(self) -> int:
        return len(self.pair_dataset)

    def __getitem__(self, idx: int) -> dict:
        lhs_idx, rhs_idx = self.pair_dataset._choose_pair(idx)
        lhs = self.samples[lhs_idx]
        rhs = self.samples[rhs_idx]
        return {
            "embedding_a": self.embeddings[lhs_idx],
            "embedding_b": self.embeddings[rhs_idx],
            "labels": comparison_labels(lhs.values, rhs.values, self.margin),
            "sample_weight": self.sample_weight,
            "sample_kind": self.sample_kind,
        }

class SinglePPADataset(Dataset):
    def __init__(self, samples: list[PPASample], tokenizer, max_input_len: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        encoded = self.tokenizer(
            format_prompt(sample),
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
        )
        encoded["sample_idx"] = idx
        return encoded

class PairwiseCollator:
    def __init__(self, tokenizer):
        self.left_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        self.right_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features: list[dict]) -> dict:
        left = [
            {"input_ids": f["input_ids_a"], "attention_mask": f["attention_mask_a"]}
            for f in features
        ]
        right = [
            {"input_ids": f["input_ids_b"], "attention_mask": f["attention_mask_b"]}
            for f in features
        ]
        left_batch = self.left_collator(left)
        right_batch = self.right_collator(right)
        return {
            "input_ids_a": left_batch["input_ids"],
            "attention_mask_a": left_batch["attention_mask"],
            "input_ids_b": right_batch["input_ids"],
            "attention_mask_b": right_batch["attention_mask"],
            "labels": torch.tensor(np.stack([f["labels"] for f in features]), dtype=torch.long),
            "sample_weight": torch.tensor([f["sample_weight"] for f in features], dtype=torch.float32),
            "sample_kind": torch.tensor([f["sample_kind"] for f in features], dtype=torch.long),
        }

class PairwiseEmbeddingCollator:
    def __call__(self, features: list[dict]) -> dict:
        return {
            "embedding_a": torch.stack([f["embedding_a"] for f in features]),
            "embedding_b": torch.stack([f["embedding_b"] for f in features]),
            "labels": torch.tensor(np.stack([f["labels"] for f in features]), dtype=torch.long),
            "sample_weight": torch.tensor([f["sample_weight"] for f in features], dtype=torch.float32),
            "sample_kind": torch.tensor([f["sample_kind"] for f in features], dtype=torch.long),
        }

class SingleCollator(DataCollatorWithPadding):
    def __call__(self, features: list[dict]) -> dict:
        sample_idx = [f.pop("sample_idx") for f in features]
        batch = super().__call__(features)
        batch["sample_idx"] = torch.tensor(sample_idx, dtype=torch.long)
        return batch

class PairwisePPAClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_metrics: int,
        hidden_dropout: float,
        head_hidden_size: int | None = None,
        head_num_layers: int = 2,
        metric_weights: list[float] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.num_metrics = num_metrics
        self.hidden_size = get_backbone_hidden_size(backbone)
        self.pad_token_id = int(getattr(backbone.config, "pad_token_id", 0) or 0)
        head_hidden_size = head_hidden_size or self.hidden_size
        
        layers = []
        layers.append(nn.LayerNorm(self.hidden_size * 4))
        layers.append(nn.Dropout(hidden_dropout))
        
        if head_num_layers == 1:
            layers.append(nn.Linear(self.hidden_size * 4, num_metrics * len(CLASS_NAMES)))
        else:
            layers.append(nn.Linear(self.hidden_size * 4, head_hidden_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(hidden_dropout))
            
            for _ in range(head_num_layers - 2):
                layers.append(nn.Linear(head_hidden_size, head_hidden_size))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(hidden_dropout))
                
            layers.append(nn.Linear(head_hidden_size, num_metrics * len(CLASS_NAMES)))
            
        self.classifier = nn.Sequential(*layers)
        if metric_weights is None:
            metric_weights = [1.0] * num_metrics
        self.register_buffer(
            "metric_weights",
            torch.tensor(metric_weights, dtype=torch.float32),
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        last_token_indices = attention_mask.shape[1] - 1 - torch.argmax(
            attention_mask.flip(dims=[1]), dim=1
        )
        return hidden[torch.arange(input_ids.shape[0], device=input_ids.device), last_token_indices]

    def _pad_to_length(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pad_amount = target_length - input_ids.shape[1]
        if pad_amount <= 0:
            return input_ids, attention_mask
        return (
            F.pad(input_ids, (0, pad_amount), value=self.pad_token_id),
            F.pad(attention_mask, (0, pad_amount), value=0),
        )

    def encode_pair(
        self,
        input_ids_a: torch.Tensor,
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,
        attention_mask_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_length = max(input_ids_a.shape[1], input_ids_b.shape[1])
        input_ids_a, attention_mask_a = self._pad_to_length(input_ids_a, attention_mask_a, max_length)
        input_ids_b, attention_mask_b = self._pad_to_length(input_ids_b, attention_mask_b, max_length)
        input_ids = torch.cat([input_ids_a, input_ids_b], dim=0)
        attention_mask = torch.cat([attention_mask_a, attention_mask_b], dim=0)
        embeddings = self.encode(input_ids, attention_mask)
        return embeddings.chunk(2, dim=0)

    def compare_embeddings(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        features = torch.cat([lhs, rhs, lhs - rhs, lhs * rhs], dim=-1)
        head_param = next(self.classifier.parameters(), None)
        if head_param is not None:
            features = features.to(device=head_param.device, dtype=head_param.dtype)
        logits = self.classifier(features)
        return logits.view(lhs.shape[0], self.num_metrics, len(CLASS_NAMES))

    def forward(
        self,
        input_ids_a: torch.Tensor | None = None,
        attention_mask_a: torch.Tensor | None = None,
        input_ids_b: torch.Tensor | None = None,
        attention_mask_b: torch.Tensor | None = None,
        embedding_a: torch.Tensor | None = None,
        embedding_b: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        sample_kind: torch.Tensor | None = None,
    ) -> dict:
        if embedding_a is not None and embedding_b is not None:
            lhs, rhs = embedding_a, embedding_b
        else:
            if input_ids_a is None or attention_mask_a is None or input_ids_b is None or attention_mask_b is None:
                raise ValueError("Either embeddings or both tokenized inputs must be provided.")
            lhs, rhs = self.encode_pair(input_ids_a, attention_mask_a, input_ids_b, attention_mask_b)
        logits = self.compare_embeddings(lhs, rhs)
        loss = None
        if labels is not None:
            per_metric_loss = F.cross_entropy(
                logits.reshape(-1, len(CLASS_NAMES)),
                labels.reshape(-1),
                reduction="none",
            ).view(labels.shape)
            # Apply metric weights registered as buffer
            weighted_metric_loss = per_metric_loss * self.metric_weights.to(per_metric_loss.device)
            per_sample_loss = weighted_metric_loss.mean(dim=1)
            if sample_weight is not None:
                per_sample_loss = per_sample_loss * sample_weight.to(per_sample_loss.device)
            loss = per_sample_loss.mean()
        return {"loss": loss, "logits": logits}

    def save_pretrained(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.backbone.save_pretrained(output_dir)
        torch.save(
            {
                "classifier": self.classifier.state_dict(),
                "num_metrics": self.num_metrics,
                "hidden_size": self.hidden_size,
                "metric_weights": self.metric_weights.cpu(),
            },
            os.path.join(output_dir, "pairwise_classifier_head.pt"),
        )

class PairwiseTrainer(Trainer):
    def __init__(self, *args, tokenizer_to_save=None, sample_weight_schedule: SampleWeightSchedule | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer_to_save = tokenizer_to_save
        self.sample_weight_schedule = sample_weight_schedule

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sample_kind = inputs.pop("sample_kind", None)
        if self.sample_weight_schedule is not None and sample_kind is not None:
            max_steps = int(getattr(self.state, "max_steps", 0) or getattr(self.args, "max_steps", 0) or 1)
            easy_weight, hard_weight = self.sample_weight_schedule.weights(self.state.global_step, max_steps)
            sample_kind = sample_kind.to(model.device if hasattr(model, "device") else inputs["labels"].device)
            inputs["sample_weight"] = torch.where(
                sample_kind == SAMPLE_KIND_HARD,
                torch.full_like(sample_kind, hard_weight, dtype=torch.float32),
                torch.full_like(sample_kind, easy_weight, dtype=torch.float32),
            )
        outputs = model(**inputs)
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        self.model.save_pretrained(output_dir)
        if self.tokenizer_to_save is not None:
            self.tokenizer_to_save.save_pretrained(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model

        # Load PEFT / LoRA adapter weights if applicable
        from peft import PeftModel
        unwrapped_model = self.accelerator.unwrap_model(model)

        if isinstance(unwrapped_model.backbone, PeftModel):
            loaded = False
            for filename in ["adapter_model.safetensors", "adapter_model.bin"]:
                path = os.path.join(resume_from_checkpoint, filename)
                if os.path.exists(path):
                    logger.info("Loading LoRA adapter weights from %s", path)
                    if filename.endswith(".safetensors"):
                        from safetensors.torch import load_file
                        state_dict = load_file(path, device="cpu")
                    else:
                        state_dict = torch.load(path, map_location="cpu")

                    # load_state_dict with strict=False because state_dict only contains adapter weights
                    unwrapped_model.backbone.load_state_dict(state_dict, strict=False)
                    loaded = True
                    break
            if not loaded:
                logger.warning("No adapter weights found in %s", resume_from_checkpoint)

        # Load classifier head weights
        head_path = os.path.join(resume_from_checkpoint, "pairwise_classifier_head.pt")
        if os.path.exists(head_path):
            logger.info("Loading pairwise classifier head weights from %s", head_path)
            state = torch.load(head_path, map_location="cpu")
            unwrapped_model.classifier.load_state_dict(state["classifier"])
            if "metric_weights" in state and hasattr(unwrapped_model, "metric_weights"):
                unwrapped_model.register_buffer("metric_weights", state["metric_weights"].to(unwrapped_model.metric_weights.device))
        else:
            logger.warning("No pairwise classifier head weights found in %s", resume_from_checkpoint)

@torch.no_grad()
def evaluate_random_accuracy(
    model: PairwisePPAClassifier,
    samples: list[PPASample],
    tokenizer,
    args,
    split: str,
    device: torch.device,
    embeddings: torch.Tensor | None = None,
) -> dict:
    eval_pair_mode = "random"
    num_pairs = min(args.eval_random_pairs, max(1, len(samples) * max(len(samples) - 1, 1)))
    if args.exhaustive_eval_pairs:
        exhaustive_pairs = family_pair_indices(samples)
        if exhaustive_pairs:
            eval_pair_mode = "family_exhaustive"
            num_pairs = len(exhaustive_pairs)
    if embeddings is None:
        dataset = PairwisePPADataset(
            samples=samples,
            tokenizer=tokenizer,
            max_input_len=args.max_input_len,
            margin=args.margin,
            num_pairs=num_pairs,
            pair_mode=eval_pair_mode,
            seed=args.seed + 1009,
        )
        collator = PairwiseCollator(tokenizer)
    else:
        dataset = PairwiseEmbeddingDataset(
            samples=samples,
            embeddings=embeddings,
            margin=args.margin,
            num_pairs=num_pairs,
            pair_mode=eval_pair_mode,
            seed=args.seed + 1009,
        )
        collator = PairwiseEmbeddingCollator()
    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers,
    )
    correct = np.zeros(len(args.metrics), dtype=np.int64)
    total = np.zeros(len(args.metrics), dtype=np.int64)
    label_counts = Counter()
    for batch in tqdm(dataloader, desc=f"Random pair accuracy {split}"):
        labels = batch.pop("labels")
        batch.pop("sample_weight", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch)["logits"].detach().cpu()
        preds = logits.argmax(dim=-1).numpy()
        labels_np = labels.numpy()
        correct += (preds == labels_np).sum(axis=0)
        total += labels_np.shape[0]
        label_counts.update(labels_np.reshape(-1).tolist())

    row = {
        "split": split,
        "pair_mode": eval_pair_mode,
        "n_pairs": int(total[0]) if len(total) else 0,
        "overall_accuracy": float(correct.sum() / max(total.sum(), 1)),
    }
    for idx, metric in enumerate(args.metrics):
        row[f"{metric}_accuracy"] = float(correct[idx] / max(total[idx], 1))
    for class_idx, name in enumerate(CLASS_NAMES):
        row[f"label_{name}"] = int(label_counts.get(class_idx, 0))
    return row

@torch.no_grad()
def encode_samples(
    model: PairwisePPAClassifier,
    samples: list[PPASample],
    tokenizer,
    args,
    device: torch.device,
) -> torch.Tensor:
    dataset = SinglePPADataset(samples, tokenizer, args.max_input_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.embed_batch_size,
        shuffle=False,
        collate_fn=SingleCollator(tokenizer=tokenizer),
        num_workers=args.dataloader_num_workers,
    )
    embeddings = [None] * len(samples)
    for batch in tqdm(dataloader, desc="Encoding modules"):
        sample_idx = batch.pop("sample_idx").tolist()
        batch = {k: v.to(device) for k, v in batch.items()}
        encoded = model.encode(**batch).detach().float().cpu()
        for row_idx, embedding in zip(sample_idx, encoded):
            embeddings[row_idx] = embedding
    return torch.stack(embeddings, dim=0)

@torch.no_grad()
def score_group_pairs(
    model: PairwisePPAClassifier,
    embeddings: torch.Tensor,
    indices: list[int],
    args,
    device: torch.device,
) -> np.ndarray:
    group_emb = embeddings[indices]
    n = group_emb.shape[0]
    scores = torch.zeros(n, len(args.metrics), dtype=torch.float32)
    flat_lhs = torch.arange(n).repeat_interleave(n)
    flat_rhs = torch.arange(n).repeat(n)
    for start in range(0, flat_lhs.numel(), args.compare_batch_size):
        end = min(start + args.compare_batch_size, flat_lhs.numel())
        lhs = group_emb[flat_lhs[start:end]].to(device)
        rhs = group_emb[flat_rhs[start:end]].to(device)
        logits = model.compare_embeddings(lhs, rhs)
        probs = torch.softmax(logits.float(), dim=-1)
        expected = probs[..., 2] - probs[..., 0]
        same_item = flat_lhs[start:end] == flat_rhs[start:end]
        if same_item.any():
            expected[same_item.to(expected.device)] = 0.0
        target_rows = flat_lhs[start:end]
        scores.index_add_(0, target_rows.cpu(), expected.detach().cpu())
    # Self-comparisons are included in the n^2 pass above, but should not
    # influence ranking scores.
    return scores.numpy()

def safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    value = fn(x, y).statistic
    return float(value) if not np.isnan(value) else float("nan")

@torch.no_grad()
def evaluate_family_kendall(
    model: PairwisePPAClassifier,
    samples: list[PPASample],
    tokenizer,
    args,
    split: str,
    device: torch.device,
) -> list[dict]:
    embeddings = encode_samples(model, samples, tokenizer, args, device)
    grouped = defaultdict(list)
    for idx, sample in enumerate(samples):
        grouped[sample.group].append(idx)

    rows = []
    per_pair_correct = np.zeros(len(args.metrics), dtype=np.int64)
    per_pair_total = np.zeros(len(args.metrics), dtype=np.int64)
    for group, indices in tqdm(grouped.items(), desc=f"Family Kendall {split}"):
        if len(indices) < args.min_family_samples:
            continue
        scores = score_group_pairs(model, embeddings, indices, args, device)
        values = np.stack([samples[i].values for i in indices]).astype(np.float64)
        families = [samples[i].family for i in indices]
        clocks = [samples[i].clock for i in indices]
        family = families[0]
        clock = clocks[0]
        n = len(indices)

        lhs_vals = values[:, None, :]
        rhs_vals = values[None, :, :]
        labels = comparison_labels(
            lhs_vals,
            rhs_vals,
            args.margin,
        ).reshape(n, n, len(args.metrics))
        group_emb = embeddings[indices]
        preds = []
        flat_lhs = torch.arange(n).repeat_interleave(n)
        flat_rhs = torch.arange(n).repeat(n)
        for start in range(0, flat_lhs.numel(), args.compare_batch_size):
            end = min(start + args.compare_batch_size, flat_lhs.numel())
            logits = model.compare_embeddings(
                group_emb[flat_lhs[start:end]].to(device),
                group_emb[flat_rhs[start:end]].to(device),
            )
            preds.append(logits.argmax(dim=-1).detach().cpu().numpy())
        pred_labels = np.concatenate(preds, axis=0).reshape(n, n, len(args.metrics))
        pair_mask = ~np.eye(n, dtype=bool)
        per_pair_correct += (pred_labels[pair_mask] == labels[pair_mask]).sum(axis=0)
        per_pair_total += labels[pair_mask].shape[0]

        for metric_idx, metric in enumerate(args.metrics):
            gt = values[:, metric_idx]
            pred_score = scores[:, metric_idx]
            rows.append(
                {
                    "split": split,
                    "family": family,
                    "clock": clock,
                    "group": group,
                    "metric": metric,
                    "n": n,
                    "spearman_r": safe_corr(spearmanr, pred_score, gt),
                    "kendall_tau": safe_corr(kendalltau, pred_score, gt),
                }
            )

    summary_rows = []
    for metric_idx, metric in enumerate(args.metrics):
        metric_rows = [r for r in rows if r["metric"] == metric]
        kt = [r["kendall_tau"] for r in metric_rows if not math.isnan(r["kendall_tau"])]
        sp = [r["spearman_r"] for r in metric_rows if not math.isnan(r["spearman_r"])]
        summary_rows.append(
            {
                "split": split,
                "family": "__summary__",
                "clock": "",
                "group": "",
                "metric": metric,
                "n": sum(r["n"] for r in metric_rows),
                "spearman_r": float(np.mean(sp)) if sp else float("nan"),
                "kendall_tau": float(np.mean(kt)) if kt else float("nan"),
                "pair_accuracy": float(per_pair_correct[metric_idx] / max(per_pair_total[metric_idx], 1)),
            }
        )
    return summary_rows + rows

def write_rows(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
