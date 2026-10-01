"""Fine-tune T5Gemma as a pairwise PPA comparison classifier.

This is the T5Gemma counterpart of ``finetune_ppa_pair_classifier.py``.  The
data, labels, and evaluation metrics are shared with the Gemma-4 classifier,
but the backbone is a seq2seq T5Gemma encoder.  Each RTL module is encoded
independently, pooled, and compared with a small per-metric three-way head:

* 0: lhs is less than rhs
* 1: lhs is almost equal to rhs, within ``--margin``
* 2: lhs is greater than rhs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import ConcatDataset
from transformers import AutoTokenizer, TrainingArguments

from finetune_ppa import apply_lora, lora_target_modules
from finetune_ppa_latent_losses import infer_hidden_size
from t5gemma_pair_helpers import (
    CLASS_NAMES,
    DEFAULT_EASY_TRAIN_SPLITS,
    DEFAULT_FAMILY_EVAL_SPLITS,
    DEFAULT_HARD_TRAIN_SPLITS,
    DEFAULT_METRICS,
    DEFAULT_RANDOM_EVAL_SPLITS,
    SAMPLE_KIND_EASY,
    SAMPLE_KIND_HARD,
    PairwiseCollator,
    PairwiseEmbeddingCollator,
    PairwiseEmbeddingDataset,
    PairwisePPADataset,
    PairwiseTrainer,
    SampleWeightSchedule,
    build_samples,
    encode_samples,
    evaluate_family_kendall,
    evaluate_random_accuracy,
    existing_splits,
    family_pair_indices,
    limit_samples,
    load_splits,
    write_rows,
)
from omnipred import T5GemmaModel, T5GemmaModelConfig


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def resolve_checkpoint(path: str | None) -> str | None:
    if not path:
        return None
    return path


def has_tokenizer(path: str | None) -> bool:
    return bool(path) and os.path.exists(os.path.join(path, "tokenizer_config.json"))


def has_adapter(path: str | None) -> bool:
    return bool(path) and os.path.exists(os.path.join(path, "adapter_config.json"))


def has_full_model(path: str | None) -> bool:
    return bool(path) and os.path.exists(os.path.join(path, "config.json"))


class T5GemmaPairwiseClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_metrics: int,
        hidden_dropout: float,
        head_hidden_size: int | None = None,
        metric_weights: list[float] | None = None,
        pooling: str = "mean",
    ):
        super().__init__()
        if pooling not in {"mean", "last"}:
            raise ValueError(f"Unknown pooling mode: {pooling}")
        self.backbone = backbone
        self.config = backbone.config
        self.generation_config = getattr(backbone, "generation_config", None)
        self.num_metrics = num_metrics
        self.hidden_size = infer_hidden_size(backbone.config)
        self.pooling = pooling
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
        self.register_buffer("metric_weights", torch.tensor(metric_weights, dtype=torch.float32))

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        if self.pooling == "last":
            last_token_indices = attention_mask.shape[1] - 1 - torch.argmax(
                attention_mask.flip(dims=[1]), dim=1
            )
            return hidden[torch.arange(input_ids.shape[0], device=input_ids.device), last_token_indices]

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

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
    ) -> dict[str, torch.Tensor | None]:
        del sample_kind
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
            weighted_metric_loss = per_metric_loss * self.metric_weights.to(per_metric_loss.device)
            per_sample_loss = weighted_metric_loss.mean(dim=1)
            if sample_weight is not None:
                per_sample_loss = per_sample_loss * sample_weight.to(per_sample_loss.device)
            loss = per_sample_loss.mean()
        return {"loss": loss, "logits": logits}

    def save_pretrained(self, output_dir: str, safe_serialization: bool = True, **kwargs: Any):
        os.makedirs(output_dir, exist_ok=True)
        self.backbone.save_pretrained(output_dir, safe_serialization=safe_serialization, **kwargs)
        torch.save(
            {
                "classifier": self.classifier.state_dict(),
                "num_metrics": self.num_metrics,
                "hidden_size": self.hidden_size,
                "metric_weights": self.metric_weights.detach().cpu(),
                "pooling": self.pooling,
            },
            os.path.join(output_dir, "pairwise_classifier_head.pt"),
        )


def load_pair_head(model: T5GemmaPairwiseClassifier, model_dir: str | None):
    head_path = os.path.join(model_dir or "", "pairwise_classifier_head.pt")
    if not os.path.exists(head_path):
        raise FileNotFoundError("Pair classification requires pairwise_classifier_head.pt in the exact model directory.")
    logger.info("Loading pairwise classifier head from %s", head_path)
    state = torch.load(head_path, map_location="cpu")
    model.classifier.load_state_dict(state["classifier"])
    if "metric_weights" in state:
        model.register_buffer("metric_weights", state["metric_weights"].to(model.metric_weights.device))


def build_model(args) -> tuple[T5GemmaPairwiseClassifier, AutoTokenizer]:
    eval_only = args.do_eval and not args.do_train
    eval_dir = resolve_checkpoint(args.eval_model_dir or args.base_model_dir) if eval_only else None
    train_source = resolve_checkpoint(args.base_model_dir) if args.base_model_dir else args.encoder_model_name
    model_source = eval_dir if eval_only and (has_adapter(eval_dir) or has_full_model(eval_dir)) else train_source

    if has_adapter(model_source):
        if not args.encoder_model_name:
            raise ValueError("Adapter loading requires an explicit --encoder_model_name for its underlying base model.")
        load_source = args.encoder_model_name
        adapter_dir = model_source
    else:
        load_source = model_source
        adapter_dir = None

    model_kwargs: dict[str, Any] = {}
    if not args.cpu:
        model_kwargs["torch_dtype"] = torch.bfloat16 if args.bf16 else torch.float32
    if args.device_map_auto:
        model_kwargs["device_map"] = "auto"

    config = T5GemmaModelConfig(
        model_name=load_source,
        max_input_len=args.max_input_len,
        max_target_len=8,
        model_kwargs=model_kwargs,
    )
    t5gemma = T5GemmaModel(config=config)

    tokenizer_source = model_source if has_tokenizer(model_source) else load_source
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, model_max_length=args.max_input_len)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    t5gemma.tokenizer = tokenizer
    t5gemma._normalize_generation_config()

    if adapter_dir is not None:
        logger.info("Loading LoRA adapter from %s", adapter_dir)
        t5gemma.model = PeftModel.from_pretrained(
            t5gemma.model,
            adapter_dir,
            is_trainable=bool(args.do_train and args.use_lora),
        )
    elif args.use_lora and args.do_train:
        logger.info(
            "Enabling LoRA fine-tuning: r=%s alpha=%s dropout=%s targets=%s",
            args.lora_rank,
            args.lora_alpha,
            args.lora_dropout,
            lora_target_modules(args),
        )
        t5gemma = apply_lora(t5gemma, args)

    model = T5GemmaPairwiseClassifier(
        backbone=t5gemma.model,
        num_metrics=len(args.metrics),
        hidden_dropout=args.head_dropout,
        head_hidden_size=args.head_hidden_size,
        metric_weights=args.metric_weights,
        pooling=args.pooling,
    )
    if eval_only or args.init_pair_head:
        load_pair_head(model, model_source)
    return model, tokenizer


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

    num_metrics = len(args.metrics)
    if len(args.margin) == 1:
        args.margin = np.full(num_metrics, args.margin[0], dtype=np.float32)
    elif len(args.margin) == num_metrics:
        args.margin = np.asarray(args.margin, dtype=np.float32)
    else:
        raise ValueError(
            f"Length of --margin ({len(args.margin)}) must be 1 or equal to number of metrics ({num_metrics})."
        )

    if args.metric_weights is None:
        args.metric_weights = [1.0] * num_metrics
    elif len(args.metric_weights) != num_metrics:
        raise ValueError(
            f"Length of --metric_weights ({len(args.metric_weights)}) must match number of metrics ({num_metrics})."
        )

    args.easy_weight_start = args.easy_weight if args.easy_weight_start is None else args.easy_weight_start
    args.easy_weight_end = args.easy_weight if args.easy_weight_end is None else args.easy_weight_end
    args.hard_weight_start = args.hard_weight if args.hard_weight_start is None else args.hard_weight_start
    args.hard_weight_end = args.hard_weight if args.hard_weight_end is None else args.hard_weight_end
    sample_weight_schedule = SampleWeightSchedule(
        schedule=args.sample_weight_schedule,
        easy_start=args.easy_weight_start,
        easy_end=args.easy_weight_end,
        hard_start=args.hard_weight_start,
        hard_end=args.hard_weight_end,
    )

    saved_args = dict(vars(args))
    saved_args["margin"] = args.margin.tolist()
    with open(os.path.join(args.output_dir, "t5gemma_pair_classifier_args.json"), "w", encoding="utf-8") as f:
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
            easy_samples = limit_samples(easy_samples, args.max_easy_samples, args.seed, args.sample_limit_strategy)
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
                        sample_kind=SAMPLE_KIND_EASY,
                    )
                )
            else:
                train_parts.append(
                    PairwisePPADataset(
                        easy_samples,
                        tokenizer,
                        args.max_input_len,
                        args.margin,
                        easy_pairs,
                        pair_mode="random",
                        seed=args.seed,
                        sample_weight=args.easy_weight,
                        sample_kind=SAMPLE_KIND_EASY,
                    )
                )

        if hard_train_splits:
            hard_samples = build_samples(
                load_splits(args.dataset_dir, hard_train_splits),
                args.metrics,
                args.rtl_key,
                args.clock_key,
                args.family_key,
                args.clock,
            )
            hard_samples = limit_samples(hard_samples, args.max_hard_samples, args.seed + 17, args.sample_limit_strategy)
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
                        sample_kind=SAMPLE_KIND_HARD,
                    )
                )
            else:
                train_parts.append(
                    PairwisePPADataset(
                        hard_samples,
                        tokenizer,
                        args.max_input_len,
                        args.margin,
                        hard_pairs,
                        pair_mode=hard_pair_mode,
                        seed=args.seed + 17,
                        sample_weight=args.hard_weight,
                        sample_kind=SAMPLE_KIND_HARD,
                    )
                )

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
            sample_weight_schedule=sample_weight_schedule,
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
            samples = limit_samples(samples, args.max_eval_samples, args.seed + 1009, args.sample_limit_strategy)
            if len(samples) >= 2:
                random_rows.append(evaluate_random_accuracy(model, samples, tokenizer, args, split, device))
        write_rows(os.path.join(args.output_dir, "random_pair_accuracy.csv"), random_rows)

        family_rows = []
        if args.eval_train_samples and train_hard_eval_samples:
            family_rows.extend(
                evaluate_family_kendall(model, train_hard_eval_samples, tokenizer, args, "train_hard_overfit", device)
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
            samples = limit_samples(samples, args.max_eval_samples, args.seed + 2029, args.sample_limit_strategy)
            if len(samples) >= args.min_family_samples:
                family_rows.extend(evaluate_family_kendall(model, samples, tokenizer, args, split, device))
        write_rows(os.path.join(args.output_dir, "family_pair_kendall.csv"), family_rows)
        summary = [row for row in family_rows if row["family"] == "__summary__"]
        write_rows(os.path.join(args.output_dir, "family_pair_kendall_summary.csv"), summary)


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False, description="Train T5Gemma as a pairwise PPA comparison classifier.")
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

    parser.add_argument("--encoder_model_name", type=str, default=None, help="Explicit underlying base model when loading an adapter.")
    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--eval_model_dir", type=str, default=None)
    parser.add_argument("--max_input_len", type=int, default=2048)
    parser.add_argument("--pooling", choices=["mean", "last"], default="mean")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    parser.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--head_hidden_size", type=int, default=None)
    parser.add_argument("--device_map_auto", action="store_true")
    parser.add_argument("--init_pair_head", action="store_true")

    parser.add_argument("--easy_pairs", type=int, default=None)
    parser.add_argument("--hard_pairs", type=int, default=None)
    parser.add_argument("--precompute_train_embeddings", action="store_true")
    parser.add_argument("--exhaustive_train_pairs", action="store_true")
    parser.add_argument("--easy_weight", type=float, default=1.0)
    parser.add_argument("--hard_weight", type=float, default=1.0)
    parser.add_argument("--sample_weight_schedule", choices=["constant", "linear", "cosine"], default="constant")
    parser.add_argument("--easy_weight_start", type=float, default=None)
    parser.add_argument("--easy_weight_end", type=float, default=None)
    parser.add_argument("--hard_weight_start", type=float, default=None)
    parser.add_argument("--hard_weight_end", type=float, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--embed_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
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
