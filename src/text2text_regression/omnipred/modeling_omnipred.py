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

"""Default PyTorch architecture for a RegressLM."""

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
)
from transformers.tokenization_utils import PreTrainedTokenizer

from .configuration_omnipred import OmniPredConfig
from .tokenization_omnipred import P10Tokenizer

NEG_INF = -1e12  # Used for masking in attention


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)."""

    def __init__(self, d_model: int, max_len: int, base: int = 10000):
        super().__init__()
        # Note: d_model is the head dimension in our case.
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # freqs shape is (max_len, d_model / 2)
        # The cached sin/cos tables should have a feature dimension of d_model / 2
        self.register_buffer(
            "cos_cached", freqs.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", freqs.sin()[None, None, :, :], persistent=False
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[2]  # x has shape (B, n_heads, L, head_dim).
        # Return tensors of shape (1, 1, L, head_dim / 2)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies RoPE to query and key tensors."""
    # q, k have shape (B, H, L, D_h)
    # cos, sin have shape (B, H, L, D_h / 2) after broadcasting

    # Reshape q and k to separate even and odd dimensions
    q_reshaped = q.float().reshape(*q.shape[:-1], -1, 2)
    k_reshaped = k.float().reshape(*k.shape[:-1], -1, 2)
    q_even, q_odd = q_reshaped[..., 0], q_reshaped[..., 1]
    k_even, k_odd = k_reshaped[..., 0], k_reshaped[..., 1]
    # q_even, q_odd, k_even, k_odd have shape (B, H, L, D_h / 2)

    # Apply rotation. All tensors in this operation have a final dim of D_h / 2.
    q_out = torch.stack(
        [q_even * cos - q_odd * sin, q_even * sin + q_odd * cos], -1
    ).flatten(-2)
    k_out = torch.stack(
        [k_even * cos - k_odd * sin, k_even * sin + k_odd * cos], -1
    ).flatten(-2)

    return q_out.type_as(q), k_out.type_as(k)


class RopeTransformerEncoderLayer(nn.Module):
    """A Transformer Encoder Layer with RoPE support."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        if self.head_dim * nhead != self.d_model:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
            )
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attn_dropout_p = dropout  # For F.scaled_dot_product_attention

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.ff_dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def _sa_block(
        self,
        x: torch.Tensor,
        rotary_pos_emb: RotaryPositionalEmbedding,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Self-attention block with RoPE, manual projection, and correct masking."""
        B, L, _ = x.shape  # pylint: disable=invalid-name

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: (B, L, D) -> (B, H, L, D_h)
        q = q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        # ipdb.set_trace()

        # Apply RoPE to q and k
        cos, sin = rotary_pos_emb(q)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Prepare attention mask
        final_attn_mask = (
            key_padding_mask.view(B, 1, 1, L) if key_padding_mask is not None else None
        )

        # Perform attention
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=final_attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(attn_output)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.ff_dropout(self.activation(self.linear1(x))))

    def forward(
        self,
        src: torch.Tensor,
        rotary_pos_emb: RotaryPositionalEmbedding,
        src_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Applies the encoder layer with the Pre-Norm structure."""
        x = src
        x = x + self.dropout1(
            self._sa_block(self.norm1(x), rotary_pos_emb, src_key_padding_mask)
        )
        x = x + self.dropout2(self._ff_block(self.norm2(x)))
        return x


class RopeEncoder(nn.Module):
    """A stack of RoPE-enabled Transformer Encoder Layers."""

    def __init__(
        self,
        encoder_layer: RopeTransformerEncoderLayer,
        num_layers: int,
        norm: nn.LayerNorm | None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, rotary_pos_emb, src_key_padding_mask):
        output = src
        for mod in self.layers:
            output = mod(
                output, rotary_pos_emb, src_key_padding_mask=src_key_padding_mask
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class PositionalEncoding(nn.Module):
    """Default positional encoding."""

    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


SPD_BACKENDS = [
    nn.attention.SDPBackend.FLASH_ATTENTION,
    nn.attention.SDPBackend.MATH,
    nn.attention.SDPBackend.EFFICIENT_ATTENTION,
]


class OmniPred(nn.Module):
    """Encoder-Decoder model in PyTorch."""

    def __init__(
        self,
        config: OmniPredConfig,
        encoder_tokenizer: PreTrainedTokenizer,
        decoder_tokenizer: P10Tokenizer,
    ):
        super().__init__()
        self.config = config
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        d_model = config.d_model
        self.encoder_pad_idx = (
            encoder_tokenizer.pad_token_id if encoder_tokenizer else 0
        )
        nhead = config.nhead
        num_encoder_layers = config.num_encoder_layers
        num_decoder_layers = config.num_decoder_layers
        dim_feedforward = config.dim_feedforward
        dropout = config.dropout
        self.src_tok_emb = nn.Embedding(config.encoder_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(config.decoder_vocab_size, d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # RoPE Encoder for best length generalization.
        self.rotary_pos_emb = RotaryPositionalEmbedding(
            d_model // nhead, max_len=config.max_encoder_len
        )
        encoder_layer = RopeTransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = RopeEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # We use a standard positional encoding and decoder.
        self.decoder_positional_encoding = PositionalEncoding(
            d_model,
            max_len=config.max_decoder_len,
            dropout=dropout,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.generator = nn.Linear(d_model, config.decoder_vocab_size)

        # Decode attributes
        self.register_buffer(
            "decoder_constraint_masks",
            self._create_decoder_constraint_masks(),
            persistent=False,
        )

    def _generate_causal_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.ones((sz, sz), dtype=torch.bool), diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        src_padding_mask = input_ids == self.encoder_pad_idx
        tgt_causal_mask = self._generate_causal_mask(decoder_input_ids.size(1)).to(
            decoder_input_ids.device
        )

        with nn.attention.sdpa_kernel(SPD_BACKENDS):  # Flash attention
            # Encode with RoPE encoder
            memory = self.encoder(
                self.emb_dropout(self.src_tok_emb(input_ids)),
                self.rotary_pos_emb,
                src_key_padding_mask=src_padding_mask,
            )

        # Decode with standard decoder
        decoder_output = self.decoder(
            tgt=self.decoder_positional_encoding(self.tgt_tok_emb(decoder_input_ids)),
            memory=memory,
            tgt_mask=tgt_causal_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        logits = self.generator(decoder_output)
        if labels is not None:
            # Compute loss if labels are provided
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            return Seq2SeqLMOutput(
                loss=loss,
                logits=logits,
                past_key_values=None,
                decoder_hidden_states=None,
                decoder_attentions=None,
                cross_attentions=None,
                encoder_last_hidden_state=memory,
                encoder_hidden_states=None,
                encoder_attentions=None,
            )
        else:
            # If no labels, return logits and memory
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                cross_attentions=None,
                encoder_last_hidden_state=memory,
            )

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes the source sequence using the RoPE encoder."""
        src_padding_mask = src == self.encoder_pad_idx
        src_emb = self.emb_dropout(self.src_tok_emb(src))
        with nn.attention.sdpa_kernel(SPD_BACKENDS):  # Flash attention
            memory = self.encoder(
                src_emb, self.rotary_pos_emb, src_key_padding_mask=src_padding_mask
            )
        return memory, src_padding_mask

    def next_token_logits(
        self,
        current_tgt_seq: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes one step using the standard decoder."""
        tgt_causal_mask = self._generate_causal_mask(current_tgt_seq.size(1)).to(
            current_tgt_seq.device
        )
        tgt = self.decoder_positional_encoding(self.tgt_tok_emb(current_tgt_seq))

        with nn.attention.sdpa_kernel(SPD_BACKENDS):  # Flash attention
            decoder_output_all_steps = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_causal_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.generator(decoder_output_all_steps[:, -1, :])

    def _create_decoder_constraint_masks(self) -> torch.Tensor:
        """Creates masks for decoder constraints."""
        num_float_tokens = self.decoder_tokenizer.num_tokens_per_obj + 1
        vocab_size = self.decoder_tokenizer.vocab_size

        logits_masks = torch.zeros((num_float_tokens, vocab_size), dtype=torch.float)
        for i in range(num_float_tokens):
            for allowed_token in self.decoder_tokenizer.tokens_at_index(i):
                token_id = self.decoder_tokenizer._convert_token_to_id(allowed_token)
                logits_masks[i, token_id] = 1.0
        return logits_masks

    @torch.no_grad()
    def decode(
        self,
        inputs: dict[str, torch.Tensor] | BatchEncoding,
        num_samples: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_input = inputs["input_ids"]  # (B, L_src)
        device = encoder_input.device
        batch_size = encoder_input.shape[0]
        # memory: (B, L_src, D_model), memory_key_padding_mask: (B, L_src)
        memory, memory_key_padding_mask = self.encode(encoder_input)

        if num_samples > 1:
            memory = (
                memory.unsqueeze(1)
                .expand(-1, num_samples, -1, -1)
                .reshape(batch_size * num_samples, memory.size(1), memory.size(2))
            )
            memory_key_padding_mask = (
                memory_key_padding_mask.unsqueeze(1)
                .expand(-1, num_samples, -1)
                .reshape(batch_size * num_samples, memory_key_padding_mask.size(1))
            )

        # Initialize decoder input for the expanded batch, start with <pad>.
        current_tgt_ids = torch.full(
            (batch_size * num_samples, 1),
            self.decoder_tokenizer.bos_token_id,  # Use BOS token as the first token
            dtype=torch.long,
            device=device,
        )

        # Store all generated token IDs for all sequences in the expanded batch
        decode_len = (
            self.decoder_tokenizer.num_tokens_per_obj + 1
        )  # +1 for the end token
        generated_sequences_ids = torch.zeros(
            (batch_size * num_samples, decode_len + 1), dtype=torch.long
        )
        generated_sequences_ids[:, 0] = self.decoder_tokenizer.bos_token_id  # BOS token

        # Batched autoregressive decoding loop
        for step_idx in range(decode_len):
            # Get logits for the next token for all (B * num_samples) sequences
            # Shape: (B*S, V)
            logits = self.next_token_logits(
                current_tgt_ids, memory, memory_key_padding_mask
            )

            # Apply constraints using the pre-computed mask
            curr_mask = self.decoder_constraint_masks[step_idx, :]  # (V,)
            curr_mask = curr_mask.unsqueeze(0)  # (1, V)
            # https://github.com/pytorch/rl/blob/3cf1df0cd1379dfe846b7a255be1cf359caa80e8/torchrl/modules/distributions/discrete.py#L314
            min_real = torch.finfo(logits.dtype).min
            masked_logits = logits.masked_fill(curr_mask == 0, min_real)  # (B*S, V)

            # Apply temperature sampling, 1 token for each of the B*S sequences
            temperature = 1.0
            probs = F.softmax(masked_logits / temperature, dim=-1)
            token_ids = torch.multinomial(probs, num_samples=1)  # (B*S, 1)

            # Store the predicted token IDs
            generated_sequences_ids[:, step_idx + 1] = token_ids.squeeze(-1)
            # ipdb.set_trace()

            # Prepare input for the next step, but only if not the last float token
            if step_idx < decode_len - 1:
                current_tgt_ids = torch.cat([current_tgt_ids, token_ids], dim=1)

        # Reshape outputs back to (B, num_samples, L_decode)
        final_decoded_ids = generated_sequences_ids.view(
            batch_size, num_samples, decode_len + 1
        )

        # Compute equivalent floats.
        # output_floats = torch.zeros((batch_size, num_samples), dtype=float, device=torch.device('cpu'))
        # for b in range(batch_size):
        #     for s_idx in range(num_samples):
        #         # ipdb.set_trace()
        #         float_str = self.decoder_tokenizer.decode(
        #             final_decoded_ids[b, s_idx].tolist()
        #         )

        #         try:
        #             output_floats[b, s_idx] = float(float_str)
        #         except ValueError:
        #             # If decoding fails, set to 0
        #             output_floats[b, s_idx] = 0
        return final_decoded_ids

    def compute_loss_and_metrics(
        self,
        examples: dict[str, torch.Tensor] | BatchEncoding,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        metrics = {}
        logits = self.forward(examples["input_ids"], examples["decoder_input_ids"])
        targets = examples["labels"]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),  # (B * L_decode, V)
            targets.reshape(-1),  # Reshape to (B * L_decode)
        )
        metrics["ce_loss"] = loss.detach()

        if self.z_loss_coef is not None:
            # Calculate z_loss (log-softmax normalization constant).
            log_z = torch.logsumexp(logits, dim=-1)  # (B * L_decode)
            z_loss_per_token = self.z_loss_coef * (log_z**2)

            # Calculate the mean z_loss over the real (non-padded) tokens.
            loss_mask = (targets != self.decoder_tokenizer.pad_token_id).float()
            z_loss = (z_loss_per_token * loss_mask).sum() / loss_mask.sum()
            metrics["z_loss"] = z_loss.detach()
            loss += z_loss

        metrics["loss"] = loss.detach()
        return loss, metrics


class OmniPredRegressor(OmniPred):
    """OmniPred Regressor model."""

    config_class = OmniPredConfig
    base_model_prefix = "omnipred_regressor"

    def __init__(
        self,
        config: OmniPredConfig,
        encoder_tokenizer: PreTrainedTokenizer,
        decoder_tokenizer: P10Tokenizer,
    ):
        super().__init__(config, encoder_tokenizer, decoder_tokenizer)
        self.regressor = nn.Linear(config.d_model, 1)
        self.eos_token_id = self.decoder_tokenizer.eos_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> Seq2SeqLMOutput | BaseModelOutputWithPastAndCrossAttentions:
        """Forward pass for the OmniPred Regressor."""
        memory, memory_key_padding_mask = self.encode(input_ids)
        last_token_idx = torch.sum(~memory_key_padding_mask, dim=1) - 1
        last_token_memory = memory[torch.arange(memory.size(0)), last_token_idx, :]

        # Pass through the regressor
        output = self.regressor(last_token_memory)
        if labels is not None:
            # Compute loss if labels are provided
            loss_fct = nn.MSELoss()
            loss = loss_fct(output, labels)
            return Seq2SeqLMOutput(
                loss=loss,
                logits=output,
                past_key_values=None,
                decoder_hidden_states=None,
                decoder_attentions=None,
                cross_attentions=None,
                encoder_last_hidden_state=memory,
                encoder_hidden_states=None,
                encoder_attentions=None,
            )
        else:
            # If no labels, return logits and memory
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=output,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                cross_attentions=None,
                encoder_last_hidden_state=memory,
            )

    def decode(self, inputs, num_samples=1):
        raise NotImplementedError(
            "OmniPredRegressor does not support decoding. Use the `forward` method for regression tasks."
        )


@dataclass
class OmniPredWithNetlistOutput(Seq2SeqLMOutput):
    """Output class for OmniPredWithNetlist model."""

    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    aux_logits: torch.Tensor | None = None
    main_loss: torch.Tensor | None = None
    main_logits: torch.Tensor | None = None
    ts_loss: torch.Tensor | None = None
    

class OmniPredWithNetlist(OmniPred):
    """OmniPred model that includes netlist as auxiliary learning target."""

    config_class = OmniPredConfig
    base_model_prefix = "omnipred_with_netlist"

    def __init__(
        self,
        config: OmniPredConfig,
        encoder_tokenizer: PreTrainedTokenizer,
        decoder_tokenizer: P10Tokenizer,
        float_left: bool = True,
        joint_training: bool = True,
    ):
        super().__init__(config, encoder_tokenizer, decoder_tokenizer)
        self.float_left = float_left
        self.auxiliary_generator = nn.Linear(config.d_model, config.encoder_vocab_size)
        self.joint_training = joint_training

    def _generate_decoder_input_embeds_and_labels(
        self,
        targets: torch.Tensor,
        auxiliary_input_ids: torch.Tensor,
        auxiliary_attention_mask: torch.Tensor,
    ):
        decoder_input_ids = targets.clone()
        if self.joint_training:
            self.float_left = torch.rand(1).item() < 0.5
        if self.float_left:
            input_embeds = self.tgt_tok_emb(decoder_input_ids)
            auxiliary_length = min(self.config.max_decoder_len - input_embeds.size(1), auxiliary_input_ids.size(1) - 1)
            auxiliary_embeds = self.src_tok_emb(auxiliary_input_ids[:, :auxiliary_length])
            decoder_input_embeds = torch.cat([input_embeds, auxiliary_embeds], dim=1)
            float_label = targets[:, 1:].clone()
            float_label[float_label == self.decoder_tokenizer.pad_token_id] = -100
            auxiliary_label = auxiliary_input_ids[:, :auxiliary_length + 1].clone()
            auxiliary_label[auxiliary_label == self.encoder_tokenizer.pad_token_id] = -100
            labels = torch.cat(
                [float_label, auxiliary_label], dim=1
            )
            tgt_key_padding_mask = torch.cat(
                [
                    (decoder_input_ids[:, 1:] != self.decoder_tokenizer.pad_token_id),
                    auxiliary_attention_mask[:, :auxiliary_length + 1].bool(),
                ], dim=1
            )
        else:
            input_embeds = self.tgt_tok_emb(decoder_input_ids)
            auxiliary_length = min(self.config.max_decoder_len - input_embeds.size(1), auxiliary_input_ids.size(1) - 1)
            auxiliary_embeds = self.src_tok_emb(auxiliary_input_ids[:, :auxiliary_length])
            decoder_input_embeds = torch.cat([auxiliary_embeds, input_embeds], dim=1)
            float_label = targets[:, 1:].clone()
            float_label[float_label == self.decoder_tokenizer.pad_token_id] = -100
            auxiliary_label = auxiliary_input_ids[:, :auxiliary_length + 1].clone()
            auxiliary_label[auxiliary_label == self.encoder_tokenizer.pad_token_id] = -100
            labels = torch.cat(
                [auxiliary_label, float_label], dim=1
            )
            tgt_key_padding_mask = torch.cat(
                [
                    auxiliary_attention_mask[:, :auxiliary_length + 1].bool(),
                    (decoder_input_ids[:, 1:] != self.decoder_tokenizer.pad_token_id),
                ], dim=1
            )
        return decoder_input_embeds, labels, tgt_key_padding_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        auxiliary_input_ids: torch.Tensor,
        auxiliary_attention_mask: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        src_padding_mask = input_ids == self.encoder_pad_idx
        target_length = targets.size(1)
        decoder_input_embeds, labels, tgt_key_padding_mask = self._generate_decoder_input_embeds_and_labels(
            targets, auxiliary_input_ids, auxiliary_attention_mask
        )

        tgt_causal_mask = self._generate_causal_mask(decoder_input_embeds.size(1)).to(
            decoder_input_embeds.device
        )

        with nn.attention.sdpa_kernel(SPD_BACKENDS):  # Flash attention
            # Encode with RoPE encoder
            memory = self.encoder(
                self.emb_dropout(self.src_tok_emb(input_ids)),
                self.rotary_pos_emb,
                src_key_padding_mask=src_padding_mask,
            )

        # ipdb.set_trace()

        # Decode with standard decoder
        decoder_output = self.decoder(
            tgt=self.decoder_positional_encoding(decoder_input_embeds),
            memory=memory,
            tgt_mask=tgt_causal_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        # split decoder output for main and auxiliary tasks
        if self.float_left:
            main_decoder_output = decoder_output[:, : target_length - 1, :]
            aux_decoder_output = decoder_output[:, target_length - 1:, :]
            main_decoder_output = main_decoder_output.contiguous()
            aux_decoder_output = aux_decoder_output.contiguous()
            main_labels = labels[:, : target_length - 1]
            aux_labels = labels[:, target_length - 1 :]
        else:
            main_decoder_output = decoder_output[:, - (target_length - 1):, :]
            aux_decoder_output = decoder_output[:, : - (target_length - 1), :]
            main_decoder_output = main_decoder_output.contiguous()
            aux_decoder_output = aux_decoder_output.contiguous()
            main_labels = labels[:, - (target_length - 1):]
            aux_labels = labels[:, : - (target_length - 1)]

        # ipdb.set_trace()

        logits = self.generator(main_decoder_output)
        aux_logits = self.auxiliary_generator(aux_decoder_output)

        # Compute loss if labels are provided
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        main_loss = loss_fct(logits.reshape(-1, logits.shape[-1]), main_labels.reshape(-1))
        aux_loss = loss_fct(aux_logits.reshape(-1, aux_logits.shape[-1]), aux_labels.reshape(-1))
        loss = (main_loss + aux_loss) / 2.0
        return OmniPredWithNetlistOutput(
            loss=loss,
            logits=logits,
            aux_loss=aux_loss,
            aux_logits=aux_logits,
            main_loss=main_loss,
            main_logits=logits,
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=memory,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )
