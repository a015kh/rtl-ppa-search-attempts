import json
import logging
from dataclasses import dataclass
from datasets import load_dataset
from typing import Any
import ipdb

import numpy as np
import torch
from datasets import Dataset as HfDataset
from omnipred import P10Tokenizer
from omnipred.modeling_omnipred import OmniPred, OmniPredRegressor
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)



INSTRUCTION_TEMPLATE = "The design is synthesized under the constraints {sdc_name}. Please predicts the {metric}. The RTL code is as follows:\n{rtl_key}\n"
PREDICT_INSTRUCTION_TEMPLATE = "Please first predicts {target1} then predicts {target2}"
TRUNCATION_TEMPLATE = "Some parts of the design are truncated due to length limit."

def format_float_scientific(value: float, num_digits: int) -> str:
    """Format a float in scientific notation with a fixed number of significant digits."""
    return np.format_float_scientific(value, precision=num_digits, unique=False, trim='k')

class Text2TextRegressionDataset(Dataset):
    def __init__(
        self,
        example_file,
        encoder_tokenizer: AutoTokenizer,
        decoder_tokenizer: P10Tokenizer,
        max_length: int | None = None,
        max_samples: int | None = None,
        split: str = None,
        metric: str = None,
    ):
        with open(example_file, "r") as f:
            examples = json.load(f)

        if split is not None:
            examples = examples[split]

        self.max_length = max_length
        self.examples = examples
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.metric = metric
        self.instruction = f"Predict the {metric} of the Verilog module"

        if max_samples is not None:
            self.examples = self.examples[:max_samples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = example["x"]
        if self.metric is not None:
            input_text = f"{self.instruction} {input_text}"
        target_text = str(
            example["y"]
        )  # Ensure target is a string, originally it might be a float or int

        # Tokenize the input and target text

        inputs = self.encoder_tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        targets = self.decoder_tokenizer(target_text)

        return inputs, targets

class Text2TextRegressionDatasetWithFloat(Dataset):
    def __init__(
        self,
        examples: list[dict[str, Any]],
        encoder_tokenizer: AutoTokenizer,
        max_length: int | None = None,
        max_samples: int | None = None,
        split: str = None,
        metric: str = None,
        stats: tuple[float, float] | None = None,
    ):
        self.examples = examples
        self.encoder_tokenizer = encoder_tokenizer
        self.max_length = max_length
        self.metric = metric
        self.instruction = f"Predict the {metric} of the Verilog module"

        if max_samples is not None:
            self.examples = self.examples[:max_samples]

        if stats is not None:
            self.mean, self.std = stats
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.examples)

    def set_stats(self, mean: float, std: float):
        """Set the mean and std for normalization."""
        self.mean = mean
        self.std = std

    def normalize(self):
        """Normalize the target values using the mean and std."""
        if self.std is not None:
            for example in self.examples:
                example["y"] = (float(example["y"]) - self.mean) / self.std
        else:
            y = [float(example["y"]) for example in self.examples]
            self.mean = np.mean(y)
            self.std = np.std(y)
            for example in self.examples:
                example["y"] = (float(example["y"]) - self.mean) / self.std
        return self.mean, self.std

    def denormalize(self, value):
        """Denormalize a value using the mean and std."""
        return value * self.std + self.mean

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = example["x"]
        if self.metric is not None:
            input_text = f"{self.instruction} {input_text}"
        target = float(example["y"])

        # Tokenize the input and target text

        inputs = self.encoder_tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )

        return inputs, target


class HFDatasetWrapper(Dataset):
    def __init__(
        self,
        encoder_tokenizer: AutoTokenizer,
        decoder_tokenizer: P10Tokenizer,
        dataset: HfDataset = None,
        path: str = None,
        max_length: int | None = None,
        metric: str | None = None,
        sdc_name: str = "SDC",
        rtl_key: str = "rtl",
    ):

        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_length = max_length
        self.metric = metric
        self.sdc_name = sdc_name
        self.rtl_key = rtl_key
        if path is not None:
            self.examples = HfDataset.load_from_disk(path)
        else:
            self.examples = dataset

            def get_input_prompt(example):
                if self.sdc_name not in example:
                    
                    return example[self.rtl_key]
                # return example['rtl']
                return f"The design is synthesized under the constraints {example[self.sdc_name]}. The RTL code is as follows:\n{example[self.rtl_key]}\nPlease predicts the {self.metric}:"
            # ipdb.set_trace()

            example = self.examples[0]
            if self.sdc_name not in example:
                logger.info("No SDC constraints found in the dataset.")
            else:
                logger.info(f"Using SDC constraints: {self.sdc_name}")

            self.examples = self.examples.map(
                lambda x: {
                    "x": get_input_prompt(x),
                    "y": str(x[f"{self.metric}"])
                }, remove_columns=self.examples.column_names)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = example["x"]
        target_text = example["y"]

        inputs = self.encoder_tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        targets = self.decoder_tokenizer(target_text)

        return inputs, targets


class HFDatasetWrapperSeq2Seq(Dataset):
    """
    Returns dicts compatible with transformers.DataCollatorForSeq2Seq:
      {
        "input_ids": ...,
        "attention_mask": ...,
        "labels": ...
      }

    - No padding here. Let DataCollatorForSeq2Seq pad dynamically.
    - Uses ONE tokenizer for both source and target (recommended).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset: HfDataset = None,
        path: str | None = None,
        max_input_len: int | None = None,
        max_target_len: int | None = None,
        metric: str | None = None,
        sdc_name: str = "SDC",
        rtl_key: str | list[str] = "rtl",
        num_digits: int = 4,
    ):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.metric = metric
        self.sdc_name = sdc_name
        if isinstance(rtl_key, str):
            self.rtl_keys = [rtl_key]
        else:
            self.rtl_keys = rtl_key
        self.num_digits = num_digits

        if path is not None:
            self.examples = HfDataset.load_from_disk(path)
        else:
            self.examples = dataset

        if self.examples is None:
            raise ValueError("You must provide either `dataset` or `path`.")

        ex0 = self.examples[0]
        if self.sdc_name not in ex0:
            logger.info("No SDC constraints found in the dataset.")
        else:
            logger.info(f"Using SDC constraints: {self.sdc_name}")

        if self.metric is None:
            raise ValueError("`metric` must be provided to build target strings.")

        def get_input_prompts(batch):
            new_x = []
            new_y = []
            
            batch_size = len(batch[next(iter(batch))])
            
            for i in range(batch_size):
                sdc = batch[self.sdc_name][i] if self.sdc_name in batch else None
                try:
                    metric_val = batch[f"{self.metric}_value"][i]
                except KeyError:
                    metric_val = batch[f"{self.metric}"][i]
                    
                formatted_y = format_float_scientific(metric_val, self.num_digits)
                
                for key in self.rtl_keys:
                    content = batch[key][i]
                    if sdc:
                        # Auto-detect if it's RTL or Netlist or something else
                        key_desc = "RTL code" if key == "rtl" else f"{key}"
                        prompt = (
                            f"The design is synthesized under the constraints {sdc}. "
                            f"The {key_desc} is as follows:\n{content}\n"
                            f"Please predicts the {self.metric}:"
                        )
                    else:
                         prompt = f"{content}"
                         
                    new_x.append(prompt)
                    new_y.append(formatted_y)
            return {"x": new_x, "y": new_y}

        # Keep strings only; tokenize in __getitem__ (or pre-tokenize with .map if you prefer)
        self.examples = self.examples.map(
            get_input_prompts,
            batched=True,
            remove_columns=self.examples.column_names,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        src_text = ex["x"]
        tgt_text = ex["y"]

        # Source tokens
        model_inputs = self.tokenizer(
            src_text,
            truncation=True,
            max_length=self.max_input_len,
            padding=False,  # <-- collator will pad
        )

        # Target tokens (labels)
        # Preferred API: text_target=... (works for seq2seq tokenizers)
        labels = self.tokenizer(
            text_target=tgt_text,
            truncation=True,
            max_length=self.max_target_len - 1,
            padding=False,  # <-- collator will pad
        )["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            labels = labels + [self.tokenizer.eos_token_id]

        model_inputs["labels"] = labels
        return model_inputs

@dataclass
class T5GemmaDataCollator:
    tokenizer: PreTrainedTokenizer
    max_input_len: int
    max_target_len: int
    padding: PaddingStrategy = PaddingStrategy.MAX_LENGTH
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        label_name = "label" if "label" in features[0] else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0] else None
        non_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]
    
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_input_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.int)

        return batch

@dataclass
class OmniPredDataCollator:
    encoder_tokenizer: PreTrainedTokenizer
    decoder_tokenizer: P10Tokenizer
    model: OmniPred | None = None
    padding: PaddingStrategy = PaddingStrategy.MAX_LENGTH
    max_length: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        x_text = [f[0] for f in features]
        y_text = [f[1] for f in features]
        # Encode the input text
        x = self.encoder_tokenizer.pad(
            x_text,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Encode the target text
        y = self.decoder_tokenizer.pad(
            y_text,
            padding=self.padding,
            return_tensors="pt",
        )
        # Prepare the decoder input IDs
        decoder_input_ids = y.input_ids.clone()
        x["decoder_input_ids"] = decoder_input_ids[:, :-1]  # Exclude last token
        x["labels"] = decoder_input_ids[:, 1:].clone()  # Shifted
        x["labels"][
            x["labels"] == self.decoder_tokenizer.pad_token_id
        ] = -100  # Ignore padding in loss
        return x


class OmniPredDataCollatorWithFloat(OmniPredDataCollator):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        x_text = [f[0] for f in features]
        y_float = [f[1] for f in features]
        # Encode the input text
        x = self.encoder_tokenizer.pad(
            x_text,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Convert float targets to tensor
        y = torch.tensor(y_float, dtype=torch.float32).unsqueeze(
            1
        )  # Add dimension for batch size

        x["labels"] = y
        return x


class NetlistTokenizedDataset(Dataset):
    def __init__(self,
        encoder_tokenizer: AutoTokenizer,
        decoder_tokenizer: P10Tokenizer,
        dataset: HfDataset,
        max_length: int | None = None,
        metric: str | None = None,
        input_ids: str = "input_ids",
        attention_mask: str = "attention_mask",
    ):
        super().__init__()
        self.dataset = dataset
        self.metric = metric
        self.max_length = max_length
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.instruction = f"Please predicts the {self.metric}:"
        self.instruction_tokenized = encoder_tokenizer(
            self.instruction,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer

        def get_input_prompt(example):
            return INSTRUCTION_TEMPLATE.format(
                sdc_name=example["SDC"],
                metric=self.metric,
                rtl_key=example["rtl"]
            )

        remove_columns=self.dataset.column_names
        remove_columns.remove(self.input_ids)
        remove_columns.remove(self.attention_mask)

        self.dataset = self.dataset.map(
        lambda x: {
            "x": get_input_prompt(x),
            "y": str(x[f"{self.metric}"])
        }, remove_columns=remove_columns)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        input_text = example["x"]
        target_text = example["y"]
        auxiliary_input_ids = example[self.input_ids]
        auxiliary_attention_mask = example[self.attention_mask]
        inputs = self.encoder_tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        targets = self.decoder_tokenizer(target_text)
        # ipdb.set_trace()
        return {
            "inputs": inputs,
            "targets": targets,
            "auxiliary_input_ids": auxiliary_input_ids,
            "auxiliary_attention_mask": auxiliary_attention_mask,
        }


def pad_batch_1d(seqs, pad_value, max_length=None, dtype=torch.long):
    # seqs: List[List[int]]
    if max_length is None:
        max_length = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_length), pad_value, dtype=dtype)
    for i, s in enumerate(seqs):
        s = s[:max_length]
        out[i, :len(s)] = torch.tensor(s, dtype=dtype)
    return out

@dataclass
class NetlistDataCollator:
    encoder_tokenizer: PreTrainedTokenizer
    decoder_tokenizer: P10Tokenizer
    padding: PaddingStrategy = PaddingStrategy.MAX_LENGTH
    max_length: int | None = None
    float_left: bool = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        
        x_text = [f["inputs"] for f in features]
        y_text = [f["targets"] for f in features]
        # z_text = [{"input_ids": f["auxiliary_input_ids"], "attention_mask": f["auxiliary_attention_mask"]} for f in features]
        z_input_ids = [f["auxiliary_input_ids"] for f in features]
        z_attention_mask = [f["auxiliary_attention_mask"] for f in features]
        # Encode the input text
        x = self.encoder_tokenizer.pad(
            x_text,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Encode the target text
        y = self.decoder_tokenizer.pad(
            y_text,
            padding=self.padding,
            return_tensors="pt",
        )
        z = {
            "input_ids": pad_batch_1d(z_input_ids, pad_value=self.encoder_tokenizer.pad_token_id, max_length=self.max_length),
            "attention_mask": pad_batch_1d(z_attention_mask, pad_value=0, max_length=self.max_length),
        }
        x["auxiliary_input_ids"] = z["input_ids"]
        x["auxiliary_attention_mask"] = z["attention_mask"]
        x["targets"] = y.input_ids
        # Prepare the decoder input IDs
        # decoder_input_ids = y.input_ids.clone()
        # if self.float_left:
        #     x["decoder_input_ids"] = torch.cat(
        #         [decoder_input_ids, z["input_ids"][:, :-1]], dim=1
        #     )
        #     float_label = decoder_input_ids[:, 1:].clone()
        #     auxiliary_label = z["input_ids"].clone()
        #     float_label[float_label == self.decoder_tokenizer.pad_token_id] = -100
        #     auxiliary_label[auxiliary_label == self.encoder_tokenizer.pad_token_id] = -100
        #     x["labels"] = torch.cat(
        #         [float_label, auxiliary_label], dim=1
        #     )
        # else:
        #     x["decoder_input_ids"] = torch.cat(
        #         [z["input_ids"], decoder_input_ids[:, :-1]], dim=1
        #     )
        #     auxiliary_label = z["input_ids"][:, 1:].clone()
        #     float_label = decoder_input_ids.clone()
        #     auxiliary_label[auxiliary_label == self.encoder_tokenizer.pad_token_id] = -100
        #     float_label[float_label == self.decoder_tokenizer.pad_token_id] = -100
        #     x["labels"] = torch.cat(
        #         [auxiliary_label, float_label], dim=1
        #     )

        return x

class TokenizedNetlistPPADataset(Dataset):
    def __init__(self, 
            dataset, 
            mode="netlist", 
            metric="area", 
            tokenizer=None,
            max_input_len=2048,
            max_target_len=2048,
            num_digits=4,
            test=False,
        ):
        self.dataset = dataset
        self.mode = mode # 'netlist' or 'ppa'
        self.metric = metric
        self.tokenizer = tokenizer
        self.num_digits = num_digits
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.truncation_input_ids = self.tokenizer(
            TRUNCATION_TEMPLATE,
            max_length=self.max_target_len,
            truncation=True,
        )["input_ids"]
        self.test = test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]

        target_text = format_float_scientific(ex[f"{self.metric}_value"], self.num_digits)
        target_inputs = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            truncation=True,
        )
        input_ids = ex[f"rtl_{self.metric}_input_ids"]
        attention_mask = ex[f"rtl_{self.metric}_attention_mask"]
        
        if self.mode == "netlist":
            # Netlist generation
            # Targets are already tokenized in 'netlist_labels' usually.
            auxiliary = ex["netlist_labels"]

        elif self.mode == "ppa":            
            auxiliary = ex[f"{self.metric}_auxiliary_input_ids"]
        
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented.")
                
        r = np.random.random()
        if self.test:
            additional_instruction = PREDICT_INSTRUCTION_TEMPLATE.format(
                target1=self.metric,
                target2="netlist" if self.mode == "netlist" else "Synthesis report"
            )
            labels = target_inputs["input_ids"]
        elif r < 0.5:
            additional_instruction = PREDICT_INSTRUCTION_TEMPLATE.format(
                target1=self.metric,
                target2="netlist" if self.mode == "netlist" else "Synthesis report"
            )
            # EOS between target and netlist is designed to separate the two parts. Originally used "\n", but EOS helps in decoding.
            labels = target_inputs["input_ids"] + [self.tokenizer.eos_token_id] + auxiliary
            labels = labels[:self.max_target_len]
        else:
            additional_instruction = PREDICT_INSTRUCTION_TEMPLATE.format(
                target1="netlist" if self.mode == "netlist" else "Synthesis report",
                target2=self.metric
            )
            # add the truncation notice if needed
            expected_len = len(auxiliary) + 1 + len(target_inputs["input_ids"])
            if expected_len > self.max_target_len:
                auxiliary = auxiliary[:self.max_target_len - 1 - len(target_inputs["input_ids"]) - len(self.truncation_input_ids)] + self.truncation_input_ids
            labels = auxiliary + [self.tokenizer.eos_token_id] + target_inputs["input_ids"]
            labels = labels[:self.max_target_len]

        additional_input = self.tokenizer(additional_instruction)
        input_ids = additional_input["input_ids"] + input_ids
        attention_mask = additional_input["attention_mask"] + attention_mask
        input_ids = input_ids[:self.max_input_len]
        attention_mask = attention_mask[:self.max_input_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
            
class NetlistTSDataset(Dataset):
    def __init__(self, 
            dataset, 
            metric="area", 
            tokenizer=None,
            max_input_len=2048,
            max_target_len=2048,
            num_digits=4,
            test=False,
        ):
        self.dataset = dataset
        self.metric = metric
        self.tokenizer = tokenizer
        self.num_digits = num_digits
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.test = test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]

        target_text = format_float_scientific(ex[f"{self.metric}_value"], self.num_digits)
        # RTL (Student)
        inputs = ex["rtl"]
        
        # Netlist (Teacher)
        # Assumes netlist_labels are provided in the dataset
        auxiliary = ex["netlist"]

        return {
            "inputs": inputs,
            "labels": target_text,
            "auxiliary": auxiliary,
        }

@dataclass
class T5GemmaTSDataCollator:
    tokenizer: PreTrainedTokenizer
    max_input_len: int
    max_target_len: int
    padding: PaddingStrategy = PaddingStrategy.MAX_LENGTH
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # ipdb.set_trace()

        # Extract features
        inputs = [f["inputs"] for f in features]
        labels = [f["labels"] for f in features]
        auxiliary = [f["auxiliary"] for f in features]

        # Pad input_ids and attention_mask (RTL)
        batch = self.tokenizer(
            inputs,
            padding=self.padding,
            truncation=True,
            max_length=self.max_input_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Pad auxiliary_input_ids and auxiliary_attention_mask (Netlist)
        aux_batch = self.tokenizer(
            auxiliary,
            padding=self.padding,
            truncation=True,
            max_length=self.max_input_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Pad labels (Targets)
        # Usually -100 for pad token in labels
        labels_batch = self.tokenizer(
            text_target=labels,
            padding=False,
            truncation=True,
            max_length=self.max_target_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        # ipdb.set_trace()
        labels_tensor = labels_batch["input_ids"]
        labels_tensor[labels_tensor == self.tokenizer.pad_token_id] = -100

        return {
            "inputs": batch,
            "labels": labels_tensor,  # Used by Trainer for logging
            "auxiliary": aux_batch,
        }


class MultiClockSeq2SeqDataset(Dataset):
    """Seq2Seq dataset that predicts `num_clocks` metric values simultaneously.

    Each training sample corresponds to one RTL design that must have been
    synthesised under at least `num_clocks` distinct clock constraints.
    Designs with fewer clock entries are silently discarded.

    The model input is the RTL code with a generic instruction (no specific
    clock is mentioned).  The label is the `num_clocks` metric values sorted
    by ascending clock period and concatenated with "\\n", each formatted in
    scientific notation.

    Args:
        tokenizer: A HuggingFace tokenizer used for both source and target.
        dataset:   A HfDataset with columns for RTL text, the clock field,
                   and the metric field.
        path:      Alternative to `dataset` — load from disk.
        max_input_len:  Maximum encoder token length (None = no limit).
        max_target_len: Maximum decoder token length (None = no limit).
        metric:    Column name for the synthesis metric (e.g. "area").
                   The dataset must contain either ``{metric}`` or
                   ``{metric}_value`` columns.
        rtl_key:   Column name for the RTL source text (default "rtl").
        clock_key: Column name for the clock / SDC constraint (default "SDC").
        num_clocks: Number of clock values required per design (default 6).
        num_digits: Significant digits for scientific-notation formatting.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset: HfDataset = None,
        path: str | None = None,
        max_input_len: int | None = None,
        max_target_len: int | None = None,
        metric: str | None = None,
        rtl_key: str = "rtl",
        clock_key: str = "clock",
        num_clocks: int = 6,
        num_digits: int = 4,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        if metric is None:
            raise ValueError("`metric` must be provided.")

        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.metric = metric
        self.rtl_key = rtl_key
        self.clock_key = clock_key
        self.num_clocks = num_clocks
        self.num_digits = num_digits

        if path is not None:
            raw = HfDataset.load_from_disk(path)
        elif dataset is not None:
            raw = dataset
        else:
            raise ValueError("Provide either `dataset` or `path`.")

        # ------------------------------------------------------------------ #
        # Group rows by RTL text; collect (clock_float, metric_float) pairs.  #
        # Grouping always uses the "rtl" column regardless of rtl_key, because #
        # netlists are unique per row and cannot be meaningfully grouped.       #
        # ------------------------------------------------------------------ #
        # groups: rtl_text -> {clock: metric, ...}
        # inputs: rtl_text -> training input text (row[rtl_key])
        groups: dict[str, dict[float, float]] = {}
        inputs: dict[str, str] = {}  # rtl_text -> row[rtl_key]

        metric_col = f"{metric}_value" if f"{metric}_value" in raw.column_names else metric
        # ipdb.set_trace()

        for row in raw:
            # Always group by RTL so designs with the same RTL but different
            # clock constraints are collected together. When rtl_key=="netlist"
            # we still need a stable group key — use "rtl" for that.
            group_key = row["rtl"]
            try:
                clock_val = float(row[clock_key])
                metric_val = float(row[metric_col])
            except (TypeError, ValueError, KeyError):
                continue

            if group_key not in groups:
                groups[group_key] = {}
                inputs[group_key] = row[rtl_key]  # training input (rtl or netlist)
            # Keep the first-seen metric value for each unique clock
            if clock_val not in groups[group_key]:
                groups[group_key][clock_val] = metric_val

        # ------------------------------------------------------------------ #
        # Build samples: sort clocks, take first num_clocks, filter < num_clocks
        # ------------------------------------------------------------------ #
        self.samples: list[dict] = []
        for group_key, clock_metric in groups.items():
            if len(clock_metric) < num_clocks:
                logger.debug(
                    f"Discarding design with only {len(clock_metric)} clocks "
                    f"(need {num_clocks})."
                )
                continue
            sorted_pairs = sorted(clock_metric.items())[:num_clocks]
            target_str = "\n".join(
                format_float_scientific(v, num_digits) for _, v in sorted_pairs
            )
            self.samples.append({"x": inputs[group_key], "y": target_str})

        logger.info(
            f"MultiClockSeq2SeqDataset: kept {len(self.samples)} designs "
            f"(each with {num_clocks} clock targets) out of {len(groups)} total."
        )

        # Shuffle and cap AFTER grouping so the limit applies to whole designs,
        # not to individual rows that would break group integrity.
        if max_samples is not None or seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.samples)  # type: ignore[arg-type]
        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
            logger.info(
                f"MultiClockSeq2SeqDataset: capped to {max_samples} designs after shuffle."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        src_text = f"Predict the {self.metric} for the following six clock periods: 1.0ns, 3.0ns, 5.0ns, 8.0ns, 10.0ns, 15.0ns. The RTL code is as follows:\n{sample['x']}\n"
        tgt_text = sample["y"]

        model_inputs = self.tokenizer(
            src_text,
            truncation=True,
            max_length=self.max_input_len,
            padding=False,
        )

        labels = self.tokenizer(
            text_target=tgt_text,
            truncation=True,
            max_length=(self.max_target_len - 1) if self.max_target_len else None,
            padding=False,
        )["input_ids"]

        if self.tokenizer.eos_token_id is not None:
            labels = labels + [self.tokenizer.eos_token_id]

        model_inputs["labels"] = labels
        return model_inputs


# ---------------------------------------------------------------------------
# Multi-metric PPA dataset
# Predicts several PPA metrics simultaneously for one (RTL, clock) sample.
# ---------------------------------------------------------------------------

_DEFAULT_METRICS = ["area", "delay", "static_power", "switching_power"]

# Mapping from user-facing metric name to possible column names in the dataset.
# The first existing column name found is used.
_METRIC_COL_CANDIDATES: dict[str, list[str]] = {
    "area":             ["area_value", "area"],
    "delay":            ["delay_value", "delay"],
    "static_power":     ["static_power_value", "static_power"],
    "switching_power":  ["switching_power_value", "switching_power"],
    "total_power":      ["total_power_value", "total_power"],
    "tns":              ["tns_value", "tns"],
    "wns":              ["wns_value", "wns"],
}


class MultiMetricSeq2SeqDataset(Dataset):
    """Seq2Seq dataset that predicts multiple PPA metrics simultaneously.

    Each sample is a single (RTL, clock) row.  The target label is the
    selected metric values formatted in scientific notation and joined with
    ``"\\n"``.  Rows where any of the requested metrics cannot be parsed as a
    float are silently discarded.

    Args:
        tokenizer:    A HuggingFace tokenizer.
        dataset:      A HfDataset with RTL, clock, and metric columns.
        path:         Alternative to ``dataset`` — load from disk.
        max_input_len:  Maximum encoder token length.
        max_target_len: Maximum decoder token length.
        metrics:      Ordered list of metric names to predict (default:
                      ``["area", "delay", "static_power", "switching_power"]``).
        rtl_key:      Column name for the RTL source text (default ``"rtl"``).
        clock_key:    Column name for the clock period (default ``"clock"``).
        num_digits:   Significant digits for scientific-notation formatting.
        max_samples:  If set, shuffle then cap the dataset to this many samples.
        seed:         Random seed used when ``max_samples`` is set.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset: HfDataset = None,
        path: str | None = None,
        max_input_len: int | None = None,
        max_target_len: int | None = None,
        metrics: list[str] | None = None,
        rtl_key: str = "rtl",
        clock_key: str = "clock",
        num_digits: int = 4,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        self.tokenizer     = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.metrics       = metrics if metrics is not None else _DEFAULT_METRICS
        self.rtl_key       = rtl_key
        self.clock_key     = clock_key
        self.num_digits    = num_digits

        if path is not None:
            raw = HfDataset.load_from_disk(path)
        elif dataset is not None:
            raw = dataset
        else:
            raise ValueError("Provide either `dataset` or `path`.")

        # Resolve column names for each metric once.
        col_map: dict[str, str] = {}
        for m in self.metrics:
            candidates = _METRIC_COL_CANDIDATES.get(m, [m])
            resolved = next((c for c in candidates if c in raw.column_names), None)
            if resolved is None:
                raise ValueError(
                    f"Metric '{m}': none of {candidates} found in dataset columns "
                    f"{raw.column_names}."
                )
            col_map[m] = resolved

        self.metric_col_map = col_map

        # Build samples ---------------------------------------------------
        self.samples: list[dict] = []
        n_skipped = 0
        for row in raw:
            rtl_text = row.get(rtl_key, "")
            if not rtl_text:
                n_skipped += 1
                continue

            try:
                clock_val = float(row[clock_key])
            except (TypeError, ValueError, KeyError):
                n_skipped += 1
                continue

            try:
                metric_floats = [float(row[col_map[m]]) for m in self.metrics]
            except (TypeError, ValueError, KeyError):
                n_skipped += 1
                continue

            target_str = "\n".join(
                format_float_scientific(v, num_digits) for v in metric_floats
            )
            design_name = row.get("design_name", "")
            family_id = row.get("family_id", "")
            self.samples.append({
                "x":           rtl_text,
                "clock":       clock_val,
                "design_name": "" if design_name is None else str(design_name),
                "family_id":   "" if family_id is None else str(family_id),
                "y":           target_str,
            })

        logger.info(
            f"MultiMetricSeq2SeqDataset: {len(self.samples)} valid samples "
            f"({n_skipped} skipped). Metrics: {self.metrics}."
        )

        # Optional shuffle + cap (after building, so cap is on whole samples)
        if max_samples is not None or seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.samples)  # type: ignore[arg-type]
        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
            logger.info(f"MultiMetricSeq2SeqDataset: capped to {max_samples} samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample   = self.samples[idx]
        metrics_str = ", ".join(self.metrics)
        src_text = (
            f"Predict the {metrics_str} for clock period {sample['clock']:.1f}ns. "
            f"The RTL code is as follows:\n{sample['x']}\n"
        )
        tgt_text = sample["y"]

        model_inputs = self.tokenizer(
            src_text,
            truncation=True,
            max_length=self.max_input_len,
            padding=False,
        )

        labels = self.tokenizer(
            text_target=tgt_text,
            truncation=True,
            max_length=(self.max_target_len - 1) if self.max_target_len else None,
            padding=False,
        )["input_ids"]

        if self.tokenizer.eos_token_id is not None:
            labels = labels + [self.tokenizer.eos_token_id]

        model_inputs["labels"] = labels
        return model_inputs
