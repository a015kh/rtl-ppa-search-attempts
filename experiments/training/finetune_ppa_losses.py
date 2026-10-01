"""Fine-tune T5Gemma for PPA prediction with configurable token losses.

This script is a loss-experiment variant of ``finetune_ppa.py``. It keeps the
same dataset, LoRA, callback, and CSV evaluation flow, while making the training
loss composable. The first added loss is Number Token Loss (``ntloss``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

from finetune_ppa import (
    apply_lora,
    configure_dataset,
    load_model_for_eval,
    lora_target_modules,
)
from omnipred import T5GemmaModel, T5GemmaModelConfig
from utils import T5GemmaPPACallback

from token_losses import compose_token_losses, make_number_token_loss

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class PPALossTrainer(Trainer):
    """Trainer that combines normal seq2seq CE with optional number-token loss."""

    def __init__(
        self,
        *args,
        tokenizer,
        enabled_losses: list[str],
        ce_weight: float,
        ntloss_weight: float,
        ntloss_digit_level: bool,
        ntloss_reweigh: bool,
        ntloss_squash_factor: float | None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.enabled_losses = set(enabled_losses)
        self.ce_weight = ce_weight
        self.ntloss_weight = ntloss_weight
        self._loss_sums = defaultdict(float)
        self._loss_count = 0

        self.ntloss_fn = None
        if "ntloss" in self.enabled_losses:
            vocab_size = getattr(self.model.config, "vocab_size", None)
            self.ntloss_fn = make_number_token_loss(
                tokenizer=tokenizer,
                vocab_size=vocab_size,
                digit_level=ntloss_digit_level,
                reweigh=ntloss_reweigh,
                squash_factor=ntloss_squash_factor,
            )

    def _remember(self, **values: torch.Tensor):
        self._loss_count += 1
        for key, value in values.items():
            self._loss_sums[key] += float(value.detach().float().cpu())

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        labels = inputs["labels"]
        logits = outputs.logits
        ce_loss = outputs.loss if outputs.loss is not None else logits.new_tensor(0.0)
        total_loss, nt_loss = compose_token_losses(
            logits, labels, ce_loss, self.enabled_losses,
            self.ce_weight, self.ntloss_weight, self.ntloss_fn,
        )

        self._remember(loss=total_loss, ce_loss=ce_loss, ntloss=nt_loss)
        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: dict[str, float], *args, **kwargs):
        if self._loss_count:
            logs = dict(logs)
            for key, value in self._loss_sums.items():
                logs[f"train/{key}"] = value / self._loss_count
            self._loss_sums.clear()
            self._loss_count = 0
        return super().log(logs, *args, **kwargs)


def resolve_model_name_or_path(args) -> str:
    if args.base_model_dir:
        model_name_or_path = args.base_model_dir
        logger.info("Loading base model from %s", model_name_or_path)
        return model_name_or_path
    logger.info("Loading model from HF hub: %s", args.encoder_model_name)
    return args.encoder_model_name


def build_training_args(args) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        do_eval=True,
        eval_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        torch_compile=False,
        run_name=os.path.split(args.output_dir.rstrip(os.sep))[-1],
    )


@torch.no_grad()
def evaluate_model(model, tokenizer, dataloader, split_name: str, args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device)
    model.eval()

    all_relative_maes: list[float] = []
    per_target_relative_maes: list[list[float]] = []
    max_relative_maes: list[float] = []
    gt_cols: list[list[float]] = []
    pred_cols: list[list[float]] = []
    design_names: list[str] = []
    clocks: list[float] = []
    sample_offset = 0
    total_samples = 0
    total_decode_fail = 0

    for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
        batch_size = batch["input_ids"].shape[0]
        dataset_samples = getattr(dataloader.dataset, "samples", [])
        batch_samples = dataset_samples[sample_offset : sample_offset + batch_size]
        batch_design_names = [sample.get("design_name", "") for sample in batch_samples]
        batch_clocks = [sample.get("clock", "") for sample in batch_samples]
        if len(batch_design_names) < batch_size:
            batch_design_names.extend([""] * (batch_size - len(batch_design_names)))
        if len(batch_clocks) < batch_size:
            batch_clocks.extend([""] * (batch_size - len(batch_clocks)))
        sample_offset += batch_size

        batch = {k: v.to(device) for k, v in batch.items()}
        output_ids = model.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=args.max_target_len,
            num_return_sequences=args.num_samples,
            do_sample=(not args.greedy and args.num_samples > 1),
            pad_token_id=tokenizer.pad_token_id,
        ).view(batch_size, args.num_samples, -1)

        labels = batch["labels"].detach().cpu()
        labels[labels == -100] = tokenizer.pad_token_id
        ground_truth_strs = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

        for sample_output_ids, gt_str, design_name, clock in zip(
            output_ids.detach().cpu().tolist(),
            ground_truth_strs,
            batch_design_names,
            batch_clocks,
        ):
            try:
                gt_floats = [float(v.strip()) for v in gt_str.strip().split("\n") if v.strip()]
            except ValueError:
                total_decode_fail += 1
                continue
            if not gt_floats:
                total_decode_fail += 1
                continue

            decoded_per_draw = []
            for draw in tokenizer.batch_decode(sample_output_ids, skip_special_tokens=True):
                try:
                    floats = [float(v.strip()) for v in draw.strip().split("\n") if v.strip()]
                except ValueError:
                    continue
                if len(floats) == len(gt_floats):
                    decoded_per_draw.append(floats)
            if not decoded_per_draw:
                total_decode_fail += 1
                continue

            pred_floats = [
                float(np.median([draw[i] for draw in decoded_per_draw]))
                for i in range(len(gt_floats))
            ]

            if not gt_cols:
                gt_cols = [[] for _ in gt_floats]
                pred_cols = [[] for _ in pred_floats]

            sample_rel_maes = []
            for i, (pred, gt) in enumerate(zip(pred_floats, gt_floats)):
                rel_mae = abs(pred - gt) / (abs(gt) + 1e-10)
                all_relative_maes.append(rel_mae)
                sample_rel_maes.append(rel_mae)
                if i >= len(per_target_relative_maes):
                    per_target_relative_maes.append([])
                per_target_relative_maes[i].append(rel_mae)
                gt_cols[i].append(gt)
                pred_cols[i].append(pred)

            if sample_rel_maes:
                max_relative_maes.append(max(sample_rel_maes))
            design_names.append(design_name)
            clocks.append(clock)
            total_samples += 1

    if gt_cols:
        csv_data = {"design_name": design_names, "clock": clocks}
        for i, (gts, preds) in enumerate(zip(gt_cols, pred_cols)):
            name = args.metrics[i] if i < len(args.metrics) else f"metric{i}"
            csv_data[f"gt_{name}"] = gts
            csv_data[f"pred_{name}"] = preds
        csv_path = os.path.join(args.output_dir, f"predictions_ground_truth_{split_name}.csv")
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        logger.info("Saved predictions to %s", csv_path)

    if total_samples == 0:
        return {
            "avg_acc_at_10_percent": 0,
            "avg_relative_mae": 1.0,
            "decode_fail": total_decode_fail,
        }

    arr_all = np.array(all_relative_maes)
    arr_max = np.array(max_relative_maes)
    result = {
        "avg_relative_mae": float(arr_all.mean()),
        "avg_acc_at_5_percent": round(float(np.mean(arr_all < 0.05)) * 100, 2),
        "avg_acc_at_10_percent": round(float(np.mean(arr_all < 0.1)) * 100, 2),
        "avg_acc_at_20_percent": round(float(np.mean(arr_all < 0.2)) * 100, 2),
        "all_metrics_acc_at_5_percent": round(float(np.mean(arr_max < 0.05)) * 100, 2),
        "all_metrics_acc_at_10_percent": round(float(np.mean(arr_max < 0.1)) * 100, 2),
        "all_metrics_acc_at_20_percent": round(float(np.mean(arr_max < 0.2)) * 100, 2),
        "total_samples": total_samples,
        "decode_fail": total_decode_fail,
    }
    for i, target_maes in enumerate(per_target_relative_maes):
        arr = np.array(target_maes)
        name = args.metrics[i] if i < len(args.metrics) else f"metric{i}"
        result[f"{name}_acc_at_5_percent"] = round(float(np.mean(arr < 0.05)) * 100, 2)
        result[f"{name}_acc_at_10_percent"] = round(float(np.mean(arr < 0.1)) * 100, 2)
        result[f"{name}_acc_at_20_percent"] = round(float(np.mean(arr < 0.2)) * 100, 2)
    return result


def train(args):
    torch.manual_seed(args.seed)

    if not args.losses:
        raise ValueError("At least one loss must be enabled with --losses.")
    unknown_losses = sorted(set(args.losses) - {"ce", "ntloss"})
    if unknown_losses:
        raise ValueError(f"Unknown loss(es): {', '.join(unknown_losses)}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "finetune_ppa_losses_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    model_name_or_path = resolve_model_name_or_path(args)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, model_max_length=args.max_input_len)

    train_dataset, eval_dataset, test_datasets, test_datasets_full = configure_dataset(args, tokenizer)

    config = T5GemmaModelConfig(
        model_name=model_name_or_path,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
    )
    model = T5GemmaModel(config=config)
    if args.use_lora:
        logger.info(
            "Enabling LoRA fine-tuning: r=%s alpha=%s dropout=%s targets=%s",
            args.lora_rank,
            args.lora_alpha,
            args.lora_dropout,
            lora_target_modules(args),
        )
        model = apply_lora(model, args)

    data_collator = DataCollatorForSeq2Seq(
        model=model.model,
        tokenizer=tokenizer,
        max_length=args.max_input_len,
        padding="longest",
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Number of trainable parameters: %s", num_params)
    logger.info(
        "Enabled losses: %s (ce_weight=%s, ntloss_weight=%s)",
        ", ".join(args.losses),
        args.ce_weight,
        args.ntloss_weight,
    )

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

    trainer = PPALossTrainer(
        model=model.model,
        args=build_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        enabled_losses=args.losses,
        ce_weight=args.ce_weight,
        ntloss_weight=args.ntloss_weight,
        ntloss_digit_level=args.ntloss_digit_level,
        ntloss_reweigh=args.ntloss_reweigh,
        ntloss_squash_factor=args.ntloss_squash_factor,
    )

    if args.do_train:
        generation_callback = T5GemmaPPACallback(
            trainer=trainer,
            datasets=test_datasets,
            tokenizer=tokenizer,
            num_samples=args.num_samples,
            model=model,
            greedy=args.greedy,
            metrics=args.metrics,
        )
        trainer.add_callback(generation_callback)

        if args.resume_from_checkpoint:
            if not args.checkpoint_path:
                raise ValueError("Resuming requires an explicit --checkpoint_path.")
            trainer.train(resume_from_checkpoint=args.checkpoint_path)
        else:
            trainer.train()
        trainer.save_model(args.output_dir)

    if args.do_eval:
        torch.cuda.empty_cache()
        # Evaluate the just-trained model, or the exact checkpoint supplied by the user.
        eval_model = model if args.do_train else load_model_for_eval(
            args.eval_model_dir or args.base_model_dir, config=config, args=args
        )
        results = {}
        for split_name, test_dataset in test_datasets_full.items():
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                collate_fn=data_collator,
            )
            logger.info("Evaluating %s ...", split_name)
            result = evaluate_model(eval_model, tokenizer, test_dataloader, split_name, args)
            logger.info("Results for %s: %s", split_name, result)
            results[split_name] = result

        results_df = pd.DataFrame(results).T
        results_path = os.path.join(args.output_dir, "evaluation_results.csv")
        if os.path.exists(results_path):
            previous_results_df = pd.read_csv(results_path, index_col=0)
            all_columns = previous_results_df.columns.union(results_df.columns)
            results_df = results_df.reindex(columns=all_columns)
            previous_results_df = previous_results_df.reindex(columns=all_columns)
            for split_name, row in results_df.iterrows():
                previous_results_df.loc[split_name] = row
            results_df = previous_results_df
        results_df.to_csv(results_path, index=True)
        print(results_df)


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False, 
        description="Train T5Gemma for PPA prediction with configurable losses."
    )

    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["area", "delay", "static_power", "switching_power"],
    )
    parser.add_argument("--rtl_key", type=str, default="rtl")
    parser.add_argument("--clock_key", type=str, default="clock")
    parser.add_argument("--num_digits", type=int, default=4)
    parser.add_argument("--training_ratio", type=float, default=1.0)
    parser.add_argument("--clock", type=float, default=None)
    parser.add_argument("--train_splits", type=str, nargs="+", default=["train"])
    parser.add_argument("--validation_splits", type=str, nargs="+", default=["valid"])
    parser.add_argument("--callback_eval_splits", type=str, nargs="+", default=["valid"])
    parser.add_argument("--callback_eval_max_samples", type=int, default=128)
    parser.add_argument("--eval_splits", type=str, nargs="+", default=None)

    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--eval_model_dir", type=str, default=None)
    parser.add_argument("--encoder_model_name", type=str, default=None, help="Explicit underlying base model when loading an adapter.")
    parser.add_argument("--max_input_len", type=int, default=2048)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    parser.add_argument("--losses", type=str, nargs="+", default=["ce", "ntloss"])
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--ntloss_weight", type=float, default=0.3)
    parser.add_argument("--ntloss_digit_level", action="store_true", default=True)
    parser.add_argument("--no_ntloss_digit_level", action="store_false", dest="ntloss_digit_level")
    parser.add_argument("--ntloss_reweigh", action="store_true", default=True)
    parser.add_argument("--no_ntloss_reweigh", action="store_false", dest="ntloss_reweigh")
    parser.add_argument("--ntloss_squash_factor", type=float, default=None)

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--wandb_project", type=str, default="text2text_regression_losses")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=16)
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
    train(parse_args())
