import json
import logging
import os
import pickle

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset as HFDataset
from datasets import load_dataset, concatenate_datasets, DatasetDict
from omnipred import OmniPred, OmniPredConfig, OmniPredRegressor, P10Tokenizer
from safetensors.torch import load_file, save_model
from torch.utils.data import ConcatDataset, Dataset, random_split
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments, get_scheduler
from transformers.optimization import Adafactor
from transformers.trainer_utils import get_last_checkpoint
from utils import (
    GenerationWandbCallback,
    OmniPredDataCollator,
    OmniPredDataCollatorWithFloat,
    HFDatasetWrapper,
    Text2TextRegressionDataset,
)

def clock_filter(data, clock: float):
    if clock is None:
        return True
    clock = float(clock)
    if data['clock'] is None:
        return False
    synthesis_clock = float(data['clock'])
    if synthesis_clock == clock:
        return True
    return False

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def load_hf_dataset(path):
    files = os.listdir(path)
    files = sorted(files)
    # Map file names to their paths
    # train.jsonl -> train, valid.jsonl -> eval
    data_files = {}
    for f in files:
        if f.endswith(".jsonl"):
            file_name = f.replace(".jsonl", "")
            data_files[file_name] = f
    dataset = load_dataset(path, data_files=data_files)
    return dataset

def load_json_dataset(
    args,
    encoder_tokenizer: AutoTokenizer,
    decoder_tokenizer: P10Tokenizer,
    train_file=None,
    test_file=None,
    metric=None,
):
    dataset = Text2TextRegressionDataset(
        example_file=train_file if train_file else args.train_file,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        max_length=args.max_input_len,
        max_samples=args.max_train_samples,
        metric=metric,
    )

    train_dataset, eval_dataset = random_split(
        dataset,
        [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)],
        generator=torch.Generator().manual_seed(42),
    )

    test_datasets = {}
    for split in ["train", "test1", "test2", "test3"]:
        test_datasets[split] = Text2TextRegressionDataset(
            example_file=test_file if test_file else args.test_file,
            encoder_tokenizer=encoder_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            max_length=args.max_input_len,
            max_samples=args.max_test_samples,
            split=split,
            metric=metric,
        )
    return train_dataset, eval_dataset, test_datasets


def configure_dataset(
    args, encoder_tokenizer: AutoTokenizer, decoder_tokenizer: P10Tokenizer
):
    regression_dataset = load_hf_dataset(args.dataset_dir)
    if args.clock is not None:
        regression_dataset = regression_dataset.filter(lambda x: clock_filter(x, args.clock))
    train_dataset = regression_dataset["train"]
    if args.training_ratio < 1.0:
        train_size = int(len(train_dataset) * args.training_ratio)
        train_dataset = train_dataset.shuffle(seed=42).select([i for i in list(range(train_size))])
    eval_dataset = regression_dataset["valid"]
    train_dataset = HFDatasetWrapper(
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        dataset=train_dataset,
        max_length=args.max_input_len,
        metric=args.metric,
        sdc_name=args.sdc_name,
        rtl_key=args.rtl_key,
    )
    eval_dataset = HFDatasetWrapper(
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        dataset=eval_dataset,
        max_length=args.max_input_len,
        metric=args.metric,
        sdc_name=args.sdc_name,
        rtl_key=args.rtl_key,
    )
    test_datasets = {}
    train_for_testing = regression_dataset["train"].shuffle(seed=42).select(range(1024))
    test_datasets["train"] = HFDatasetWrapper(
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        dataset=train_for_testing,
        max_length=args.max_input_len,
        metric=args.metric,
        sdc_name=args.sdc_name,
        rtl_key=args.rtl_key,
    )
    eval_for_testing = regression_dataset["valid"].shuffle(seed=42).select(range(1024))
    test_datasets["valid"] = HFDatasetWrapper(
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        dataset=eval_for_testing,
        max_length=args.max_input_len,
        metric=args.metric,
        sdc_name=args.sdc_name,
        rtl_key=args.rtl_key,
    )
    for key in regression_dataset.keys():
        if key.startswith("test"):
            test_datasets[key] = HFDatasetWrapper(
                encoder_tokenizer=encoder_tokenizer,
                decoder_tokenizer=decoder_tokenizer,
                dataset=regression_dataset[key],
                max_length=args.max_input_len,
                metric=args.metric,
                sdc_name=args.sdc_name,
                rtl_key=args.rtl_key,
            )


    return train_dataset, eval_dataset, test_datasets


def train(args):
    torch.manual_seed(42)  # For reproducibility

    # Load tokenizers
    encoder_tokenizer = AutoTokenizer.from_pretrained(
        args.encoder_model_name, model_max_length=args.max_input_len
    )
    decoder_tokenizer = P10Tokenizer(num_digits=args.num_digits)

    # Create dataset
    train_dataset, eval_dataset, test_datasets = configure_dataset(
        args, encoder_tokenizer, decoder_tokenizer
    )

    # Create model configuration
    config = OmniPredConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_input_len=args.max_input_len,
        encoder_vocab_size=encoder_tokenizer.vocab_size,
        decoder_vocab_size=len(decoder_tokenizer),
        num_digits=args.num_digits,
    )

    # Initialize model
    model = OmniPred(
        config=config,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
    )
    data_collator = OmniPredDataCollator(
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        max_length=args.max_input_len,  # Set max_length for padding
    )
    # report trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_params}")

    # Initialize WandB
    os.environ["WANDB_PROJECT"] = args.wandb_project
    # os.environ["WANDB_DISABLED"] = "true"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

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
        bf16=True,  # Enable mixed precision training
        dataloader_num_workers=8,
        warmup_ratio=0.1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        torch_compile=True,
        run_name=os.path.split(args.output_dir)[-1],
    )

    # # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    if args.do_train:
        # Add custom callback for generation logging
        generation_callback = GenerationWandbCallback(
            trainer=trainer,
            datasets=test_datasets,
            tokenizer=decoder_tokenizer,
            num_samples=args.num_samples,
        )
        trainer.add_callback(generation_callback)

        # Train the model
        trainer.train()

    if args.do_eval:
        # load the model from the checkpoint
        if os.path.exists(args.output_dir):
            last_checkpoint = get_last_checkpoint(args.output_dir)
            state_dict = load_file(os.path.join(last_checkpoint, "model.safetensors"))
            model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning(
                f"No checkpoint found in {args.output_dir}. Skipping evaluation."
            )
            return

        # Test the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        logger.info("Starting evaluation...")

        @torch.no_grad()
        def evaluate_model(
            model,
            dataloader,
            tokenizer: P10Tokenizer,
            split_name: str,
            plot_results=True,
        ):
            model.eval()
            epoch_mae = []
            epoch_relative_mae = []
            y_values = []
            y_predictions = []
            total_samples = 0
            y_min, y_max = float("inf"), float("-inf")

            for batch in tqdm(dataloader, desc=f"Evaluating {split_name}"):
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass
                # outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)
                if args.num_samples > 1:
                    output_ids = model.decode(
                        batch,
                        num_samples=args.num_samples,
                    )
                else:
                    output_ids = model.decode(batch).squeeze()

                labels = batch["labels"].cpu()
                ground_truth = torch.zeros(
                    (labels.shape[0], labels.shape[1] + 1), dtype=torch.long
                )
                ground_truth[:, 1:] = labels
                ground_truth[:, 0] = tokenizer.bos_token_id
                ground_truth_str = tokenizer.batch_decode(ground_truth.tolist())
                ground_truth_floats = [float(gt) for gt in ground_truth_str]

                output_ids = output_ids.cpu().tolist()

                for sample_output_ids, sample_ground_truth in zip(
                    output_ids, ground_truth_floats
                ):
                    # Update min and max for normalization
                    y_min = min(y_min, sample_ground_truth)
                    y_max = max(y_max, sample_ground_truth)
                    # Decode the output IDs to strings
                    # try:
                    # ipdb.set_trace()
                    if args.num_samples > 1:
                        output_str = tokenizer.batch_decode(sample_output_ids)
                        # get the median of the output floats
                        output_float = float(np.median([float(os) for os in output_str]))
                        # output_float = float(np.mean([float(os) for os in output_str]))
                    else:
                        output_str = tokenizer.decode(sample_output_ids)
                        output_float = float(output_str)

                    # except Exception as e:
                    #     logger.info(f"Error decoding output IDs: {e}")
                    #     continue

                    y_values.append(sample_ground_truth)
                    y_predictions.append(output_float)

                    # Calculate normalized mean absolute error
                    mae = abs(output_float - sample_ground_truth)
                    relative_mae = mae / (sample_ground_truth + 1e-10)
                    epoch_mae.append(mae)
                    epoch_relative_mae.append(relative_mae)
                    total_samples += 1

            # Calculate average MAE for the epoch
            avg_mae = sum(epoch_mae) / len(epoch_mae)
            relative_mae = np.array(epoch_relative_mae)
            avg_relative_mae = float(relative_mae.mean())
            acc_at_5_percent = round(
                float(np.mean(np.array(relative_mae) < 0.05)) * 100, 2
            )
            acc_at_10_percent = round(
                float(np.mean(np.array(relative_mae) < 0.1)) * 100, 2
            )
            acc_at_20_percent = round(
                float(np.mean(np.array(relative_mae) < 0.2)) * 100, 2
            )

            r_squared = round(float(np.corrcoef(y_values, y_predictions)[0, 1] ** 2), 2)
            logger.info(f"R-squared: {r_squared}")

            # add the regression line y' = ay + b
            a, b = np.polyfit(y_values, y_predictions, 1)
            logger.info(f"Regression line: y' = {a:.4f}y + {b:.4f}")

            if plot_results:
                plt.scatter(y_values, y_predictions, alpha=0.5)
                plt.plot(y_values, y_values, label="y=x", linestyle="--")
                plt.xlabel("Ground Truth")
                plt.ylabel("Predictions")
                plt.xscale("log")
                plt.yscale("log")
                metric_name = os.path.split(args.output_dir)[-1]
                plt.title(f"Predictions vs Ground Truth ({metric_name})")
                plt.savefig(
                    os.path.join(
                        args.output_dir, f"predictions_vs_ground_truth_{split_name}.png"
                    )
                )
                plt.clf()

            # Save the predictions and ground truth to a CSV file
            results_df = pd.DataFrame(
                {"ground_truth": y_values, "predictions": y_predictions}
            )
            results_df.to_csv(
                os.path.join(
                    args.output_dir, f"predictions_ground_truth_{split_name}.csv"
                )
            )

            # if args.synthesis_dataset:
            #     # For synthesis datasets, we can also plot the function
            #     plt.plot(x, y_predictions, label='Predictions', color='blue')
            #     plt.xlabel("Ground Truth")
            #     plt.ylabel("Predictions")
            #     plt.xscale("log")
            #     plt.yscale("log")
            #     plt.title(f"Function Plot for {split_name}")
            #     plt.legend()
            #     plt.savefig(os.path.join(args.output_dir, f"function_plot_{split_name}.png"))
            #     plt.clf()

            # Normalize the MAE
            if y_max - y_min > 0:
                normalized_mae = avg_mae / (y_max - y_min)
            else:
                normalized_mae = avg_mae
            return {
                "avg_mae": avg_mae,
                "normalized_mae": normalized_mae,
                "relative_mae": avg_relative_mae,
                "acc_at_5_percent": acc_at_5_percent,
                "acc_at_10_percent": acc_at_10_percent,
                "acc_at_20_percent": acc_at_20_percent,
            }

        results = {}
        for split_name, test_dataset in test_datasets.items():
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                collate_fn=data_collator,
            )
            logger.info(f"Evaluating {split_name} dataset...")
            result = evaluate_model(
                model, test_dataloader, decoder_tokenizer, split_name
            )
            logger.info(f"Results for {split_name}: {result}")
            results[split_name] = result

        # Save results to a CSV file
        results_df = pd.DataFrame(results).T
        results_df.to_csv(
            os.path.join(args.output_dir, "evaluation_results.csv"), index=True
        )
        print(results_df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a text-to-text regression model."
    )
    parser.add_argument(
        "--train_file", type=str, default="metrex_dataset/train/area.json"
    )
    parser.add_argument(
        "--test_file", type=str, default="metrex_dataset/test/area.json"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="metrex_dataset",
        help="Directory containing the dataset files.",
    )
    parser.add_argument(
        "--test_dataset_dir",
        type=str,
        default="data/metrex_regression",
        help="Directory containing the test dataset files.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="area",
        help="The prediction target of Verilog module",
    )
    parser.add_argument(
        "--sdc_name",
        type=str,
        default="SDC",
        help="The SDC constraint to use for evaluation.",
    )
    parser.add_argument(
        "--rtl_key",
        type=str,
        default="rtl",
        help="The key for RTL code in the dataset. Some are rtl other are code",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use from the dataset.",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use from the evaluation dataset.",
    )
    parser.add_argument(
        "--encoder_model_name",
        type=str,
        default="google-t5/t5-base",
        help="Pretrained encoder model name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the model and tokenizer.",
    )
    parser.add_argument(
        "--num_digits",
        type=int,
        default=4,
        help="Number of digits for the P10 tokenizer.",
    )
    parser.add_argument(
        "--d_model", type=int, default=512, help="Dimension of the model."
    )
    parser.add_argument(
        "--nhead", type=int, default=8, help="Number of attention heads."
    )
    parser.add_argument(
        "--num_encoder_layers", type=int, default=2, help="Number of encoder layers."
    )
    parser.add_argument(
        "--num_decoder_layers", type=int, default=2, help="Number of decoder layers."
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=2048,
        help="Dimension of the feedforward network.",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=2048,
        help="Maximum input length for the encoder.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate during decoding.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate for training."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer."
    )
    parser.add_argument(
        "--logging_dir", type=str, default="./logs", help="Directory for logging."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="text2text_regression",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="omnipred_regression",
        help="WandB run name.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="Number of steps between model saves.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=5000,
        help="Number of steps between evaluations.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to perform training."
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to perform evaluation during training.",
    )
    parser.add_argument(
        "--merge_instructions",
        action="store_true",
        help="Whether to merge instructions with input text.",
    )
    parser.add_argument(
        "--synthesis_dataset",
        action="store_true",
        help="Whether to use synthesis dataset.",
    )
    parser.add_argument(
        "--hf_dataset",
        action="store_true",
        help="Whether to use JSONL format for Hugging Face dataset.",
    )
    parser.add_argument(
        "--merge_dataset",
        action="store_true",
        help="Whether to merge dataset.",
    )
    parser.add_argument(
        "--length_testing_dataset",
        action="store_true",
        help="Whether to use length testing dataset.",
    )
    parser.add_argument(
        "--training_ratio",
        type=float,
        default=1.0,
        help="Ratio of training data to use.",
    )
    parser.add_argument(
        "--clock",
        type=float,
        default=None,
        help="Filter dataset by synthesis clock value.",
    )
    args = parser.parse_args()

    train(args)
