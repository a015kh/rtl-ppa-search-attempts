# Copyright 2025 Google LLC.
# Modified for RTL/PPA research: model wrappers, numeric outputs and training interfaces.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""T5Gemma wrapper and extension for OmniPred."""

import os
import dataclasses
from typing import Any, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import AutoTokenizer
# from transformers.models.t5_gemma.modeling_t5_gemma import T5GemmaForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, GenerationConfig
from .modeling_omnipred import OmniPredWithNetlistOutput


Tensor = torch.Tensor
IGNORE_TOKEN_ID = -100


@dataclasses.dataclass(frozen=True)
class T5GemmaModelConfig:
    """Configuration for creating a T5Gemma model."""
    model_name: str
    max_input_len: int = 2048
    max_target_len: int = 32
    max_netlist_len: int = 2048 # Added for Netlist support
    model_kwargs: dict[str, Any] | None = None
    tokenizer_kwargs: dict[str, Any] | None = None


class T5GemmaModel(nn.Module):
    """T5Gemma model wrapper."""

    def __init__(self, config: T5GemmaModelConfig):
        super().__init__()
        self.cfg = config

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.cfg.model_name, **(self.cfg.model_kwargs or {})
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, **(self.cfg.tokenizer_kwargs or {})
        )
        # Ensure pad_token_id is set if it's missing (common issue with some HF models)
        if self.tokenizer.pad_token_id is None:
             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self._normalize_generation_config()

    def _normalize_generation_config(self) -> None:
        """Make T5Gemma generation explicit across Transformers versions."""
        pad_token_id = self.tokenizer.pad_token_id
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id

        if pad_token_id is None:
            pad_token_id = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
            self.tokenizer.pad_token_id = pad_token_id

        if bos_token_id is None:
            bos_token_id = pad_token_id

        for cfg in (self.model.config, self.model.generation_config):
            cfg.pad_token_id = pad_token_id
            if getattr(cfg, "eos_token_id", None) is None:
                cfg.eos_token_id = eos_token_id
            if getattr(cfg, "bos_token_id", None) is None:
                cfg.bos_token_id = bos_token_id
            if getattr(cfg, "decoder_start_token_id", None) is None:
                cfg.decoder_start_token_id = bos_token_id


    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor | None = None, # Labels for the main task
        **kwargs
    ) -> Seq2SeqLMOutput:
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    @torch.no_grad()
    def decode(
        self, inputs: dict[str, Tensor], num_samples: int = 1
    ) -> tuple[Tensor, np.ndarray]:
        """Decodes strings."""
        self.model.eval()
        batch_size = inputs['input_ids'].shape[0]

        decoded_tokens = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            generation_config=GenerationConfig.from_model_config(self.model.config),
            max_new_tokens=self.cfg.max_target_len,
            # max_new_tokens=10,
            num_return_sequences=num_samples,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # decoded_texts = self.tokenizer.batch_decode(
        #     decoded_tokens, skip_special_tokens=True
        # )

        # # Remove the starter token and reshape. (B, S, L_decode)
        # # Note: T5Gemma generate usually includes input_ids if decoder_start_token_id is not set distinctively or depending on config.
        # # But mostly it returns just generated part + maybe prompt depending on model type.
        # # For T5/Seq2Seq, it returns sequence starting with pad or start token.
        # # We'll assume standard HF behavior.
        
        # # Check if decoder_start_token exists and strip it if necessary or just return as is.
        # # regress_lm implementation did `decoded_tokens[:, 1:]` implying removal of start token.
        # if decoded_tokens.shape[1] > 1:
        #      decoded_tokens = decoded_tokens[:, 1:]

        decoded_tokens = decoded_tokens.view(batch_size, num_samples, -1)
        # output_strings = np.array(decoded_texts).reshape(batch_size, num_samples)
        return decoded_tokens

    def save_pretrained(self,
        save_directory: str,
        safe_serialization: bool = True,
        **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        # Save the inner HF model (this handles tied weights correctly)
        self.model.save_pretrained(save_directory, safe_serialization=safe_serialization, **kwargs)
        self.tokenizer.save_pretrained(save_directory)

    def state_dict(self, *args, **kwargs):
        # Optional: make state_dict align with inner model if Trainer still calls it
        return self.model.state_dict(*args, **kwargs)


    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        **kwargs,
    ):
        """
        Create wrapper and load inner model/tokenizer from a directory or HF hub id.

        `config` is your T5GemmaModelConfig (or compatible).
        We will override config.model_name to the given path/id for loading.
        """
        # Make a shallow copy / override fields
        obj = cls(config)

        # IMPORTANT: ensure inner model/tokenizer are loaded from the path/id
        # (cls(config) already loads from config.model_name, but we reassign explicitly
        # in case your __init__ changes later or uses different defaults.)
        obj.model = AutoModelForSeq2SeqLM.from_pretrained(
            pretrained_model_name_or_path, **(config.model_kwargs or {})
        )
        obj.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **(config.tokenizer_kwargs or {})
        )

        if obj.tokenizer.pad_token_id is None:
            obj.tokenizer.pad_token_id = obj.tokenizer.eos_token_id
        obj._normalize_generation_config()

        return obj


class T5GemmaWithNetlist(T5GemmaModel):
    """T5Gemma model that includes netlist as auxiliary learning target."""

    def __init__(
        self,
        config: T5GemmaModelConfig,
        float_left: bool = True,
        joint_training: bool = True,
    ):
        super().__init__(config)
        self.float_left = float_left
        self.joint_training = joint_training

    def _generate_decoder_input_embeds_and_labels(
        self,
        targets: Tensor, # Main task labels (e.g. float value tokens)
        auxiliary_input_ids: Tensor, # Netlist tokens
        auxiliary_attention_mask: Tensor,
    ):
        # T5Gemma uses shared embeddings usually, but let's check.
        # self.model.get_decoder() gives the decoder.
        # Embeddings are usually self.model.get_input_embeddings() or self.model.encoder.embed_tokens
        
        # We need to concatenate embeddings because we are mixing float tokens and netlist tokens.
        # Targets (floats) are usually short. Auxiliary (netlist) is long.
        
        # T5Gemma expect `labels` which are token IDs. It handles shifting internally or we provide `decoder_input_ids`.
        # However, we want to construct a SEQUENCE of labels that contains both.
        # So we can just concatenate the IDs and let the model handle embedding lookup if we pass `input_ids` (as labels) 
        # But wait, T5Gemma is an Encoder-Decoder. 
        # The `labels` argument to forward() is the target sequence.
        # The `input_ids` argument is the source sequence (encoder input).
        
        # RegressLM logic:
        # Encoder Input: Text description / constraints.
        # Decoder Output: (Float/Value) + (Netlist)
        
        # In OmniPredWithNetlist:
        # decoder_input_embeds are constructed manually.
        # Here we can probably just construct `labels` (target token IDs) and let T5Gemma compute loss.
        # T5 uses `labels` to automatically create `decoder_input_ids` (shifted right).
        
        # Construct combined labels
        
        if self.joint_training:
            self.float_left = torch.rand(1).item() < 0.5

        # Prepare float labels (targets)
        # targets shape: (B, L_float)
        # auxiliary_input_ids shape: (B, L_netlist)
        
        # We need to respect max length constraints if necessary, or truncation.
        
        # Note: OmniPredWithNetlist logic constructs embeddings because it might be mixing things differently or using custom embeddings?
        # Actually it uses `self.tgt_tok_emb(decoder_input_ids)` and `self.src_tok_emb(auxiliary_input_ids)`.
        # T5Gemma likely shares embeddings or has them defined.
        
        # If we can just concatenate TOKEN IDs, that's easier and safer for HF models.
        # The only reason to use embeddings directly is if we wanted to mix embeddings from different sources (e.g. different vocabularies).
        # T5Gemma assumes a single vocabulary. If `auxiliary_input_ids` are from the SAME tokenizer, we can just concat IDs.
        # If they are from a DIFFERENT tokenizer, we have a problem.
        
        # Assumption: The user wants to use the SAME tokenizer for everything (T5Gemma tokenizer).
        # In OmniPredWithNetlist, it takes `encoder_tokenizer` and `decoder_tokenizer`.
        # T5Gemma has one tokenizer.
        # So we will assume `targets` and `auxiliary_input_ids` are both tokenized with the T5Gemma tokenizer.
        
        # Let's simple concatenate labels.
        
        # Truncate auxiliary to fit max length if needed? 
        # We assume caller handles reasonable lengths, or we truncate.
        
        if self.float_left:
            # Float then Netlist
            # Float labels
            float_labels = targets.clone()
            # Netlist labels
            aux_labels = auxiliary_input_ids.clone()
            
            # Concat
            # We assume pad tokens are handled correctly (usually -100 for labels in HF)
            # If targets/aux have pad_token_id, we should mask them if not already -100.
            
            # OmniPredWithNetlist handled masking manually.
            # Let's ensure pads are -100.
            
            float_labels[float_labels == self.tokenizer.pad_token_id] = IGNORE_TOKEN_ID
            aux_labels[aux_labels == self.tokenizer.pad_token_id] = IGNORE_TOKEN_ID
            
            combined_labels = torch.cat([float_labels, aux_labels], dim=1)
            
            # Lengths for splitting later
            len_float = float_labels.shape[1]
            len_aux = aux_labels.shape[1]
            
            split_idx = len_float
            return combined_labels, split_idx, True # True for float_left

        else:
            # Netlist then Float
            float_labels = targets.clone()
            aux_labels = auxiliary_input_ids.clone()
            
            float_labels[float_labels == self.tokenizer.pad_token_id] = IGNORE_TOKEN_ID
            aux_labels[aux_labels == self.tokenizer.pad_token_id] = IGNORE_TOKEN_ID
            
            combined_labels = torch.cat([aux_labels, float_labels], dim=1)
            
            len_float = float_labels.shape[1]
            len_aux = aux_labels.shape[1]
            
            split_idx = len_aux
            return combined_labels, split_idx, False # False for float_left

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        targets: Tensor, # Expecting 'labels' argument to be named targets to match OmniPred signature if possible, or we adapt.
        auxiliary_input_ids: Tensor,
        auxiliary_attention_mask: Tensor,
        **kwargs
    ) -> OmniPredWithNetlistOutput:
        
        # Prepare combined labels
        combined_labels, split_idx, is_float_left = self._generate_decoder_input_embeds_and_labels(
            targets, auxiliary_input_ids, auxiliary_attention_mask
        )
        
        # Run model
        # HF T5Gemma forward computes loss on 'labels'
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=combined_labels,
        )
        
        # Outputs: loss, logits, ...
        loss = outputs.loss
        logits = outputs.logits
        
        # Separate losses/logits for reporting
        # logits shape: (B, SeqLen, Vocab)
        # We need to slice it back to main and aux.
        # NOTE: 'labels' in HF are shifted right to create decoder inputs. 
        # The output logits correspond to the prediction of 'labels'.
        # So the sequence length of logits matches 'labels'.
        
        if is_float_left:
            main_logits = logits[:, :split_idx, :]
            aux_logits = logits[:, split_idx:, :]
            
            main_labels = combined_labels[:, :split_idx]
            aux_labels = combined_labels[:, split_idx:]
        else:
            aux_logits = logits[:, :split_idx, :]
            main_logits = logits[:, split_idx:, :]
            
            aux_labels = combined_labels[:, :split_idx]
            main_labels = combined_labels[:, split_idx:]
            
        # Recompute separate losses if needed for metrics
        loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN_ID)
        
        main_loss = loss_fct(main_logits.reshape(-1, main_logits.shape[-1]), main_labels.reshape(-1))
        aux_loss = loss_fct(aux_logits.reshape(-1, aux_logits.shape[-1]), aux_labels.reshape(-1))
        
        # Verify if total loss matches
        # HF averages over total tokens. 
        # Our manual split average might ideally match, but weighting could differ if lengths differ greatly.
        # OmniPredWithNetlist uses (main + aux) / 2.0. 
        # HF uses standard mean reduction.
        # We should probably return the HF loss as 'loss' for backprop to be consistent with model internals, 
        # OR override it with (main+aux)/2 if strictly desired.
        # Given the request is to "include the features... create a new module... named T5GemmaWithNetlist",
        # mimicking OmniPredWithNetlist behavior (even weighting) seems appropriate.
        
        avg_loss = (main_loss + aux_loss) / 2.0
        
        return OmniPredWithNetlistOutput(
            loss=avg_loss,
            logits=logits, # Full logits
            aux_loss=aux_loss,
            aux_logits=aux_logits,
            main_loss=main_loss,
            main_logits=main_logits,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state if hasattr(outputs, 'encoder_last_hidden_state') else None
        )


class T5GemmaTSModel(T5GemmaWithNetlist):
    def __init__(self, config: T5GemmaModelConfig, ts_weight: float = 0.1):
        super().__init__(config)
        self.ts_weight = ts_weight
        

    def forward(
        self,
        inputs: Tensor,
        labels: Tensor,
        auxiliary: Tensor,
        **kwargs
    ) -> Seq2SeqLMOutput:
        
        outputs = self.model(
            **inputs,
            labels=labels,
            return_dict=True,
        )
        aux_outputs = self.model(
            **auxiliary,
            labels=labels,
            return_dict=True,
        )
        rtl_loss = outputs.loss
        netlist_loss = aux_outputs.loss
        criterion = nn.SmoothL1Loss()
        ts_loss = criterion(outputs.encoder_last_hidden_state, aux_outputs.encoder_last_hidden_state)
        total_loss = rtl_loss + ts_loss * self.ts_weight
        # total_loss = rtl_loss
        return OmniPredWithNetlistOutput(
            loss=total_loss,
            aux_loss=netlist_loss,
            aux_logits=aux_outputs.logits,
            main_loss=rtl_loss,
            main_logits=outputs.logits,
            ts_loss=ts_loss,
        )

    @torch.no_grad()
    def decode(
        self, inputs: dict[str, Tensor], num_samples: int = 1
    ) -> tuple[Tensor, np.ndarray]:
        input_batch = inputs["inputs"]
        return super().decode(input_batch, num_samples)
