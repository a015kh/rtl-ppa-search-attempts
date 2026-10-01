"""Fine-tune a Gemma 4 pairwise PPA comparison classifier.

This entrypoint replaces string-valued PPA regression with a three-way
comparison target for every metric:

* 0: lhs is less than rhs
* 1: lhs is almost equal to rhs, within ``--margin``
* 2: lhs is greater than rhs

The backbone encodes the two RTL modules independently so each side receives
the full context budget. The pooled embeddings are concatenated and classified
by a small MLP. Evaluation caches per-module embeddings inside each family and
then runs the pair head over all in-family pairs for fast Kendall-tau scoring.
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
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

DEFAULT_METRICS = ["area", "delay", "total_power"]
DEFAULT_EASY_TRAIN_SPLITS = ["train"]
DEFAULT_HARD_TRAIN_SPLITS = ["test_aug_100", "test_aug_200", "test_pyra_200"]
DEFAULT_RANDOM_EVAL_SPLITS = ["test", "test_pluto", "test_realbench", "test_rtl_opt"]
DEFAULT_FAMILY_EVAL_SPLITS = ["test_aug_100", "test_ds_verilog_eval", "test_ppa_feedback"]
CLASS_NAMES = ["less", "almost", "greater"]


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
        )
        self.samples = samples
        self.embeddings = embeddings.float().cpu()
        self.margin = margin
        self.sample_weight = sample_weight

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
        }


class PairwiseEmbeddingCollator:
    def __call__(self, features: list[dict]) -> dict:
        return {
            "embedding_a": torch.stack([f["embedding_a"] for f in features]),
            "embedding_b": torch.stack([f["embedding_b"] for f in features]),
            "labels": torch.tensor(np.stack([f["labels"] for f in features]), dtype=torch.long),
            "sample_weight": torch.tensor([f["sample_weight"] for f in features], dtype=torch.float32),
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
        metric_weights: list[float] | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.num_metrics = num_metrics
        self.hidden_size = get_backbone_hidden_size(backbone)
        self.pad_token_id = int(getattr(backbone.config, "pad_token_id", 0) or 0)
        head_hidden_size = head_hidden_size or self.hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size * 4),
            nn.Dropout(hidden_dropout),
            nn.Linear(self.hidden_size * 4, head_hidden_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(head_hidden_size, num_metrics * len(CLASS_NAMES)),
        )
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
    def __init__(self, *args, tokenizer_to_save=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer_to_save = tokenizer_to_save

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
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


def load_backbone(args, tokenizer_source: str) -> nn.Module:
    quantization_config = None
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return AutoModel.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        quantization_config=quantization_config,
        device_map="auto" if args.device_map_auto else None,
    )


def build_model(args) -> tuple[PairwisePPAClassifier, AutoTokenizer]:
    adapter_dir = args.eval_model_dir or args.model_name
    has_adapter = os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
    has_full_checkpoint = os.path.exists(os.path.join(adapter_dir, "config.json"))
    eval_only = args.do_eval and not args.do_train

    tokenizer_source = adapter_dir if (eval_only and (has_adapter or has_full_checkpoint)) else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone_source = args.model_name
    if eval_only and has_full_checkpoint and not has_adapter:
        backbone_source = adapter_dir
    backbone = load_backbone(args, backbone_source)

    if args.use_lora:
        if eval_only and has_adapter:
            logger.info("Loading LoRA adapter from %s", adapter_dir)
            backbone = PeftModel.from_pretrained(backbone, adapter_dir, is_trainable=False)
        else:
            targets = resolve_lora_target_modules(backbone, args.lora_target_modules)
            logger.info("Using LoRA target modules: %s", targets)
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=targets,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=None,
            )
            backbone = get_peft_model(backbone, lora_config)
            if hasattr(backbone, "print_trainable_parameters"):
                backbone.print_trainable_parameters()

    model = PairwisePPAClassifier(
        backbone=backbone,
        num_metrics=len(args.metrics),
        hidden_dropout=args.head_dropout,
        head_hidden_size=args.head_hidden_size,
        metric_weights=getattr(args, "metric_weights", None),
    )
    head_path = os.path.join(adapter_dir, "pairwise_classifier_head.pt")
    if eval_only and os.path.exists(head_path):
        logger.info("Loading pairwise classifier head from %s", head_path)
        state = torch.load(head_path, map_location="cpu")
        model.classifier.load_state_dict(state["classifier"])
        if "metric_weights" in state and hasattr(model, "metric_weights"):
            model.register_buffer("metric_weights", state["metric_weights"].to(model.metric_weights.device))
    elif eval_only:
        raise FileNotFoundError("Pair classification requires pairwise_classifier_head.pt in the exact model directory.")
    return model, tokenizer


def class_counter(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(labels.reshape(-1).tolist())
    return {CLASS_NAMES[i]: int(counts.get(i, 0)) for i in range(len(CLASS_NAMES))}


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


def build_training_args(args) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, "logs"),
        eval_strategy="no",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="none" if args.no_wandb else "wandb",
        run_name=args.run_name or os.path.basename(args.output_dir.rstrip(os.sep)),
        remove_unused_columns=False,
        use_cpu=args.cpu,
    )


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Process margin (broadcast/validate)
    num_metrics = len(args.metrics)
    if len(args.margin) == 1:
        args.margin = np.full(num_metrics, args.margin[0], dtype=np.float32)
    elif len(args.margin) == num_metrics:
        args.margin = np.asarray(args.margin, dtype=np.float32)
    else:
        raise ValueError(
            f"Length of --margin ({len(args.margin)}) must be 1 or equal to number of metrics ({num_metrics})."
        )

    # Process metric weights
    if args.metric_weights is None:
        args.metric_weights = [1.0] * num_metrics
    elif len(args.metric_weights) != num_metrics:
        raise ValueError(
            f"Length of --metric_weights ({len(args.metric_weights)}) must match number of metrics ({num_metrics})."
        )

    with open(os.path.join(args.output_dir, "pair_classifier_args.json"), "w", encoding="utf-8") as f:
        # Save JSON with float/list versions instead of numpy arrays
        saved_args = dict(vars(args))
        saved_args["margin"] = args.margin.tolist()
        json.dump(saved_args, f, indent=2)

    easy_train_splits = existing_splits(args.dataset_dir, args.easy_train_splits, required=args.do_train)
    hard_train_splits = existing_splits(args.dataset_dir, args.hard_train_splits, required=args.do_train)
    random_eval_splits = existing_splits(args.dataset_dir, args.random_eval_splits, required=False)
    family_eval_splits = existing_splits(args.dataset_dir, args.family_eval_splits, required=False)

    model, tokenizer = build_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if not args.device_map_auto:
        model.to(device)
    model.train(args.do_train)

    train_easy_eval_samples = []
    train_hard_eval_samples = []
    train_easy_eval_embeddings = None
    train_hard_eval_embeddings = None
    if args.do_train:
        train_parts = []
        if easy_train_splits:
            easy_samples = build_samples(
                load_splits(args.dataset_dir, easy_train_splits),
                args.metrics,
                args.rtl_key,
                args.clock_key,
                args.family_key,
                args.clock,
            )
            easy_samples = limit_samples(
                easy_samples,
                args.max_easy_samples,
                args.seed,
                args.sample_limit_strategy,
            )
            train_easy_eval_samples = easy_samples
            easy_pairs = args.easy_pairs or max(len(easy_samples), 1)
            if args.precompute_train_embeddings:
                was_training = model.training
                model.eval()
                easy_embeddings = encode_samples(model, easy_samples, tokenizer, args, device)
                model.train(was_training)
                train_easy_eval_embeddings = easy_embeddings
                train_parts.append(
                    PairwiseEmbeddingDataset(
                        easy_samples,
                        easy_embeddings,
                        args.margin,
                        easy_pairs,
                        pair_mode="random",
                        seed=args.seed,
                        sample_weight=args.easy_weight,
                    )
                )
            else:
                train_parts.append(PairwisePPADataset(
                    easy_samples,
                    tokenizer,
                    args.max_input_len,
                    args.margin,
                    easy_pairs,
                    pair_mode="random",
                    seed=args.seed,
                    sample_weight=args.easy_weight,
                ))
        if hard_train_splits:
            hard_samples = build_samples(
                load_splits(args.dataset_dir, hard_train_splits),
                args.metrics,
                args.rtl_key,
                args.clock_key,
                args.family_key,
                args.clock,
            )
            hard_samples = limit_samples(
                hard_samples,
                args.max_hard_samples,
                args.seed + 17,
                args.sample_limit_strategy,
            )
            train_hard_eval_samples = hard_samples
            hard_pairs = args.hard_pairs or max(len(hard_samples), 1)
            hard_pair_mode = "family_exhaustive" if args.exhaustive_train_pairs else "family"
            if args.exhaustive_train_pairs and args.hard_pairs is None:
                hard_pairs = len(family_pair_indices(hard_samples))
            if args.precompute_train_embeddings:
                was_training = model.training
                model.eval()
                hard_embeddings = encode_samples(model, hard_samples, tokenizer, args, device)
                model.train(was_training)
                train_hard_eval_embeddings = hard_embeddings
                train_parts.append(
                    PairwiseEmbeddingDataset(
                        hard_samples,
                        hard_embeddings,
                        args.margin,
                        hard_pairs,
                        pair_mode=hard_pair_mode,
                        seed=args.seed + 17,
                        sample_weight=args.hard_weight,
                    )
                )
            else:
                train_parts.append(PairwisePPADataset(
                    hard_samples,
                    tokenizer,
                    args.max_input_len,
                    args.margin,
                    hard_pairs,
                    pair_mode=hard_pair_mode,
                    seed=args.seed + 17,
                    sample_weight=args.hard_weight,
                ))
        if not train_parts:
            raise ValueError("No training samples were selected. Check train split arguments.")
        train_dataset = ConcatDataset(train_parts)
        logger.info("Training on %s sampled pairs", len(train_dataset))
        trainer = PairwiseTrainer(
            model=model,
            args=build_training_args(args),
            train_dataset=train_dataset,
            data_collator=PairwiseEmbeddingCollator() if args.precompute_train_embeddings else PairwiseCollator(tokenizer),
            tokenizer_to_save=tokenizer,
        )
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
        trainer.save_model(args.output_dir)

    if args.do_eval:
        model.eval()
        if args.bf16 and device.type == "cuda" and not args.device_map_auto:
            model.to(dtype=torch.bfloat16)

        random_rows = []
        if args.eval_train_samples and train_easy_eval_samples:
            random_rows.append(
                evaluate_random_accuracy(
                    model,
                    train_easy_eval_samples,
                    tokenizer,
                    args,
                    "train_easy_overfit",
                    device,
                    embeddings=train_easy_eval_embeddings,
                )
            )
        if args.eval_train_samples and train_hard_eval_samples:
            random_rows.append(
                evaluate_random_accuracy(
                    model,
                    train_hard_eval_samples,
                    tokenizer,
                    args,
                    "train_hard_overfit",
                    device,
                    embeddings=train_hard_eval_embeddings,
                )
            )
        for split in random_eval_splits:
            samples = build_samples(
                load_splits(args.dataset_dir, [split]),
                args.metrics,
                args.rtl_key,
                args.clock_key,
                args.family_key,
                args.clock,
            )
            samples = limit_samples(
                samples,
                args.max_eval_samples,
                args.seed + 1009,
                args.sample_limit_strategy,
            )
            if len(samples) >= 2:
                random_rows.append(evaluate_random_accuracy(model, samples, tokenizer, args, split, device))
        write_rows(os.path.join(args.output_dir, "random_pair_accuracy.csv"), random_rows)
        if random_rows:
            logger.info("Random pair accuracy summary:\n%s", random_rows)

        family_rows = []
        if args.eval_train_samples and train_hard_eval_samples:
            family_rows.extend(
                evaluate_family_kendall(
                    model,
                    train_hard_eval_samples,
                    tokenizer,
                    args,
                    "train_hard_overfit",
                    device,
                )
            )
        for split in family_eval_splits:
            samples = build_samples(
                load_splits(args.dataset_dir, [split]),
                args.metrics,
                args.rtl_key,
                args.clock_key,
                args.family_key,
                args.clock,
            )
            samples = limit_samples(
                samples,
                args.max_eval_samples,
                args.seed + 2029,
                args.sample_limit_strategy,
            )
            if len(samples) >= args.min_family_samples:
                family_rows.extend(evaluate_family_kendall(model, samples, tokenizer, args, split, device))
        write_rows(os.path.join(args.output_dir, "family_pair_kendall.csv"), family_rows)
        summary = [row for row in family_rows if row["family"] == "__summary__"]
        write_rows(os.path.join(args.output_dir, "family_pair_kendall_summary.csv"), summary)
        if summary:
            logger.info("Family Kendall summary:\n%s", summary)


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False, description="Train Gemma 4 as a pairwise PPA comparison classifier.")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--easy_train_splits", nargs="+", default=DEFAULT_EASY_TRAIN_SPLITS)
    parser.add_argument("--hard_train_splits", nargs="+", default=DEFAULT_HARD_TRAIN_SPLITS)
    parser.add_argument("--random_eval_splits", nargs="+", default=DEFAULT_RANDOM_EVAL_SPLITS)
    parser.add_argument("--family_eval_splits", nargs="+", default=DEFAULT_FAMILY_EVAL_SPLITS)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--rtl_key", type=str, default="rtl")
    parser.add_argument("--clock_key", type=str, default="clock")
    parser.add_argument("--family_key", type=str, default=None)
    parser.add_argument("--clock", type=float, default=None)
    parser.add_argument("--margin", type=float, nargs="+", default=[0.03])
    parser.add_argument("--metric_weights", type=float, nargs="+", default=None)
    parser.add_argument("--max_easy_samples", type=int, default=None)
    parser.add_argument("--max_hard_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--sample_limit_strategy", choices=["first", "random"], default="first")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--eval_model_dir", type=str, default=None)
    parser.add_argument("--max_input_len", type=int, default=2048)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--head_hidden_size", type=int, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--device_map_auto", action="store_true")

    parser.add_argument("--easy_pairs", type=int, default=None)
    parser.add_argument("--hard_pairs", type=int, default=None)
    parser.add_argument("--precompute_train_embeddings", action="store_true")
    parser.add_argument("--exhaustive_train_pairs", action="store_true")
    parser.add_argument("--easy_weight", type=float, default=1.0)
    parser.add_argument("--hard_weight", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--embed_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", action="store_false", dest="bf16")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--eval_random_pairs", type=int, default=4096)
    parser.add_argument("--exhaustive_eval_pairs", action="store_true")
    parser.add_argument("--min_family_samples", type=int, default=10)
    parser.add_argument("--compare_batch_size", type=int, default=4096)
    parser.add_argument("--eval_train_samples", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
