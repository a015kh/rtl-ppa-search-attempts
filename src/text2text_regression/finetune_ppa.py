"""finetune_ppa.py – Training script for multi-metric PPA prediction.

Predicts multiple PPA metrics (area, delay, static_power, switching_power)
simultaneously for a single (RTL, clock) sample. The label is the metric
values joined by newlines in scientific notation.
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments)
from transformers.trainer_utils import get_last_checkpoint

from omnipred import T5GemmaModel, T5GemmaModelConfig
from train import clock_filter
from ppa_evaluation import evaluate_model
from utils import MultiMetricSeq2SeqDataset, T5GemmaPPACallback

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def load_hf_dataset(path):
    files = sorted(f for f in os.listdir(path) if f.endswith(".jsonl"))
    datasets = {}
    for file_name in files:
        split_name = file_name.replace(".jsonl", "")
        data_file = os.path.join(path, file_name)
        with open(data_file, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        datasets[split_name] = Dataset.from_list(records)
    return DatasetDict(datasets)


def normalize_split_name(split_name: str) -> str:
    return os.path.splitext(os.path.basename(split_name))[0]


def configure_dataset(args, tokenizer: AutoTokenizer):
    regression_dataset = load_hf_dataset(args.dataset_dir)
    if args.clock is not None:
        regression_dataset = regression_dataset.filter(lambda x: clock_filter(x, args.clock))
    metrics = args.metrics  # e.g. ["area", "delay", "static_power", "switching_power"]

    def make_dataset(hf_dataset, max_samples=None):
        return MultiMetricSeq2SeqDataset(
            tokenizer=tokenizer,
            dataset=hf_dataset,
            max_input_len=args.max_input_len,
            max_target_len=args.max_target_len,
            metrics=metrics,
            rtl_key=args.rtl_key,
            clock_key=args.clock_key,
            num_digits=args.num_digits,
            max_samples=max_samples,
        )

    # Training splits – support one or more named splits
    train_split_names = [normalize_split_name(s) for s in args.train_splits]
    missing = sorted(set(train_split_names) - set(regression_dataset.keys()))
    if missing:
        available = ", ".join(sorted(regression_dataset.keys()))
        raise ValueError(
            f"Unknown train split(s): {', '.join(missing)}. "
            f"Available splits: {available}"
        )
    train_hf_parts = [regression_dataset[s] for s in train_split_names]
    train_hf = concatenate_datasets(train_hf_parts) if len(train_hf_parts) > 1 else train_hf_parts[0]
    if args.training_ratio < 1.0:
        n = int(len(train_hf) * args.training_ratio)
        train_hf = train_hf.shuffle(seed=42).select(range(n))
    train_dataset = make_dataset(train_hf)
    validation_split_names = [normalize_split_name(s) for s in args.validation_splits]
    missing = sorted(set(validation_split_names) - set(regression_dataset.keys()))
    if missing:
        available = ", ".join(sorted(regression_dataset.keys()))
        raise ValueError(
            f"Unknown validation split(s): {', '.join(missing)}. "
            f"Available splits: {available}"
        )
    validation_hf_parts = [regression_dataset[s] for s in validation_split_names]
    validation_hf = (
        concatenate_datasets(validation_hf_parts)
        if len(validation_hf_parts) > 1
        else validation_hf_parts[0]
    )
    eval_dataset = make_dataset(validation_hf, max_samples=1024)

    # Evaluation subsets. Callback evaluation runs during training, so keep it
    # intentionally small and separate from final --do_eval coverage.
    callback_eval_keys = [normalize_split_name(s) for s in args.callback_eval_splits]
    missing_callback_keys = sorted(set(callback_eval_keys) - set(regression_dataset.keys()))
    if missing_callback_keys:
        available_keys = ", ".join(sorted(regression_dataset.keys()))
        raise ValueError(
            f"Unknown callback eval split(s): {', '.join(missing_callback_keys)}. "
            f"Available splits: {available_keys}"
        )

    full_evaluate_keys = list(dict.fromkeys(
        train_split_names
        + validation_split_names
        + [k for k in regression_dataset.keys() if k.startswith("test")]
    ))
    if args.eval_splits:
        requested_keys = [normalize_split_name(split) for split in args.eval_splits]
        missing_keys = sorted(set(requested_keys) - set(regression_dataset.keys()))
        if missing_keys:
            available_keys = ", ".join(sorted(regression_dataset.keys()))
            raise ValueError(
                f"Unknown eval split(s): {', '.join(missing_keys)}. "
                f"Available splits: {available_keys}"
            )
        full_evaluate_keys = requested_keys

    test_datasets = {}
    test_datasets_full = {}
    for key in callback_eval_keys:
        test_datasets[key] = make_dataset(
            regression_dataset[key],
            max_samples=args.callback_eval_max_samples,
        )

    for key in full_evaluate_keys:
        max_eval = 1024 if key in train_split_names else None
        test_datasets_full[key] = make_dataset(regression_dataset[key], max_samples=max_eval)

    return train_dataset, eval_dataset, test_datasets, test_datasets_full


def lora_target_modules(args):
    if len(args.lora_target_modules) == 1 and args.lora_target_modules[0] == "all-linear":
        return "all-linear"
    return args.lora_target_modules


def apply_lora(model: T5GemmaModel, args):
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules(args),
        bias=args.lora_bias,
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model.model = get_peft_model(model.model, lora_config)
    if hasattr(model.model, "print_trainable_parameters"):
        model.model.print_trainable_parameters()
    return model


def load_model_for_eval(checkpoint_path: str, config: T5GemmaModelConfig, args):
    if not args.use_lora:
        return T5GemmaModel.from_pretrained(checkpoint_path, config=config)

    model = T5GemmaModel(config=config)
    model.model = PeftModel.from_pretrained(model.model, checkpoint_path)
    return model


def train(args):
    torch.manual_seed(42)

    # Resolve model source: --base_model_dir takes precedence over --encoder_model_name.
    # If base_model_dir points to a training output directory, load the latest checkpoint.
    if args.base_model_dir:
        checkpoint = get_last_checkpoint(args.base_model_dir) if os.path.isdir(args.base_model_dir) else None
        model_name_or_path = checkpoint or args.base_model_dir
        logger.info("Loading base model from %s", model_name_or_path)
    else:
        model_name_or_path = args.encoder_model_name
        logger.info("Loading model from HF hub: %s", model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, model_max_length=args.max_input_len
    )

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
    logger.info(f"Number of trainable parameters: {num_params}")

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

    distributed_args = {}
    if args.fsdp:
        distributed_args["fsdp"] = args.fsdp
    if args.fsdp_config:
        distributed_args["fsdp_config"] = args.fsdp_config

    training_args = TrainingArguments(
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
        bf16=True,
        dataloader_num_workers=args.dataloader_num_workers,
        warmup_ratio=0.1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        torch_compile=False,
        run_name=os.path.split(args.output_dir)[-1],
        **distributed_args,
    )

    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    if args.do_train:
        generation_callback = T5GemmaPPACallback(
            trainer=trainer,
            datasets=test_datasets,
            tokenizer=tokenizer,
            num_samples=args.callback_num_samples,
            model=model,
            greedy=args.greedy,
            metrics=args.metrics,
        )
        trainer.add_callback(generation_callback)

        if args.resume_from_checkpoint:
            checkpoint_dir = args.checkpoint_path or args.output_dir
            checkpoint = get_last_checkpoint(checkpoint_dir) or checkpoint_dir
            trainer.train(resume_from_checkpoint=checkpoint)
        else:
            trainer.train()

    if args.do_eval:
        torch.cuda.empty_cache()
        evaluation_checkpoint = args.checkpoint_path or args.output_dir
        if os.path.exists(evaluation_checkpoint):
            last_checkpoint = get_last_checkpoint(evaluation_checkpoint) or evaluation_checkpoint
            model = load_model_for_eval(last_checkpoint, config=config, args=args)
        else:
            raise FileNotFoundError(f"Evaluation checkpoint does not exist: {evaluation_checkpoint}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info("Starting evaluation…")
        metrics = args.metrics


        results = {}
        for split_name, test_dataset in test_datasets_full.items():
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                collate_fn=data_collator,
                num_workers=args.dataloader_num_workers,
            )
            logger.info(f"Evaluating {split_name} …")
            result = evaluate_model(model, test_dataloader, split_name, tokenizer=tokenizer, device=device, metrics=metrics, num_samples=args.num_samples, output_dir=args.output_dir)
            logger.info(f"Results for {split_name}: {result}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train T5Gemma to predict multiple PPA metrics simultaneously."
    )
    # Dataset
    parser.add_argument("--dataset_dir",   type=str, required=True)
    parser.add_argument("--metrics",       type=str, nargs="+",
                        default=["area", "delay", "static_power", "switching_power"],
                        help="Ordered PPA metrics to predict (space-separated).")
    parser.add_argument("--rtl_key",       type=str, default="rtl")
    parser.add_argument("--clock_key",     type=str, default="clock")
    parser.add_argument("--num_digits",    type=int, default=4)
    parser.add_argument("--training_ratio", type=float, default=1.0)
    parser.add_argument("--clock", type=float, default=None)
    parser.add_argument("--train_splits", type=str, nargs="+", default=["train"],
                        help="One or more split names to use for training (e.g. train test_aug_100 test_pyra_200).")
    parser.add_argument("--validation_splits", type=str, nargs="+", default=["valid"],
                        help="One or more splits used for Trainer validation during training.")
    parser.add_argument("--eval_splits", type=str, nargs="+", default=None,
                        help="Optional split names or JSONL filenames to evaluate, e.g. test_deepseek_ref.jsonl.")
    parser.add_argument("--callback_eval_splits", type=str, nargs="+", default=["valid"],
                        help="Small split list evaluated by the W&B generation callback during training.")
    parser.add_argument("--callback_eval_max_samples", type=int, default=128,
                        help="Maximum samples per callback eval split during training.")
    parser.add_argument("--callback_num_samples", type=int, default=1,
                        help="Decode samples per example for callback eval during training.")

    # Model
    parser.add_argument("--base_model_dir", type=str, default=None,
                        help="Path to a pretrained checkpoint directory to start from. "
                             "Takes precedence over --encoder_model_name when set. "
                             "get_last_checkpoint is applied automatically.")
    parser.add_argument("--encoder_model_name", type=str,
                        default="google/t5gemma-s-s-prefixlm",
                        help="HF hub model name or local path, used when --base_model_dir is not set.")
    parser.add_argument("--max_input_len",  type=int, default=2048)
    parser.add_argument("--max_target_len", type=int, default=128,
                        help="4 floats ~ 48 chars; 64 tokens is plenty, default 128 for safety.")
    parser.add_argument("--use_lora", action="store_true",
                        help="Fine-tune LoRA adapters instead of all model weights.")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_bias", type=str, default="none",
                        choices=["none", "all", "lora_only"])
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"],
                        help='Module names to adapt, or "all-linear" for all linear layers.')

    # Training
    parser.add_argument("--output_dir",  type=str, default="./output")
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--wandb_project", type=str, default="text2text_regression")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size",  type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=float, default=20)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--save_steps",   type=int, default=2000)
    parser.add_argument("--eval_steps",   type=int, default=2000)
    parser.add_argument("--num_samples",  type=int, default=16)
    parser.add_argument("--dataloader_num_workers", type=int, default=16,
                        help="Number of DataLoader worker processes per distributed rank.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable Trainer gradient checkpointing.")
    parser.add_argument("--ddp_find_unused_parameters", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Override DDP find_unused_parameters.")
    parser.add_argument("--fsdp", type=str, default="",
                        help='Optional Trainer FSDP mode, e.g. "full_shard auto_wrap".')
    parser.add_argument("--fsdp_config", type=str, default=None,
                        help="Optional Trainer FSDP config as JSON or a config path.")

    # Flags
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval",  action="store_true")
    parser.add_argument("--greedy",   action="store_true")
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)

    args = parser.parse_args()
    train(args)
