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

"""Custom tokenizers for RegressLM.

For full generality, note that the tokens are strings, as we assume the
vocabulary will assign integer IDs.
"""
import re

import ipdb
import numpy as np
import ordered_set
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

DELIMITERS = ("<", ">")
FLOAT_REGEX = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def _to_token(s: str | int) -> str:
    left_d, right_d = DELIMITERS
    return f"{left_d}{s}{right_d}"


def _from_token(token: str) -> str | int:
    left_d, right_d = DELIMITERS
    pattern = f"{left_d}(.*?){right_d}"
    m = re.fullmatch(pattern, token)
    if not m:
        raise ValueError(f"Could not deserialize `{token}`.")
    return m.group(1)


class P10Tokenizer(PreTrainedTokenizer):
    """Uses P10 tokenization from https://arxiv.org/abs/2112.01898.

    A float f can be represented as:

    `s * m * 10^e`

    where:
        s: Positive/Negative sign (+, -)
        m: Mantissa representing leading digits.
        e: Exponent.
    """

    vocab_files_names = {
        "vocab_file": "vocab.json",
    }

    def __init__(
        self,
        num_digits: int = 4,
        exponent_range: int = 10,
        bos_token: str = "<float>",
        eos_token: str = "</float>",
        unk_token: str = "<nan>",
        pad_token: str = "<pad>",
        **kwargs,
    ):

        self.pattern = re.compile(FLOAT_REGEX)

        self.num_digits = num_digits
        self.exponent_range = exponent_range
        self.vocab = self.all_tokens()
        self.vocab = ordered_set.OrderedSet(
            [bos_token, eos_token, unk_token, pad_token] + list(self.vocab)
        )
        self.ids_to_tokens = {i: token for i, token in enumerate(self.vocab)}
        self.tokens_to_ids = {token: i for i, token in enumerate(self.vocab)}

        bos_token = (
            AddedToken(bos_token, lstrip=True, rstrip=False)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=True, rstrip=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=True, rstrip=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(pad_token, lstrip=True, rstrip=False)
            if isinstance(pad_token, str)
            else pad_token
        )

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )
        self.model_max_length = self.num_tokens_per_obj + 2

    ## P10-specific methods
    @property
    def num_tokens_per_obj(self):
        """Number of tokens used to represent each float."""
        return 2 + self.num_digits

    def all_tokens(self) -> ordered_set.OrderedSet[str]:
        """Returns ordered set of all tokens used."""
        out = []
        for i in range(self.num_tokens_per_obj):
            out.extend(self.tokens_at_index(i))
        return ordered_set.OrderedSet(out)

    def tokens_at_index(self, index: int) -> ordered_set.OrderedSet[str]:
        if index < 0 or index > self.num_tokens_per_obj:
            raise ValueError(f"Index {index} out of bounds.")

        if index == 0:  # beginning
            tokens = [_to_token(s) for s in ["+", "-"]]
        elif index == self.num_tokens_per_obj - 1:  # end
            exps = [
                f"E{i}" for i in range(-self.exponent_range, self.exponent_range + 1)
            ]
            tokens = [_to_token(s) for s in exps]
        elif index == self.num_tokens_per_obj:  # end of float
            tokens = ["<float>", "</float>"]
        else:  # middle (digit)
            tokens = [_to_token(s) for s in range(0, 10)]
        return ordered_set.OrderedSet(tokens)

    @property
    def _max_abs_val(self) -> float:
        """Largest representable positive number."""
        return float(self.num_digits * "9") * (10.0**self.exponent_range)

    @property
    def _min_abs_val(self) -> float:
        """Smallest representable positive number."""
        min_mantissa = float("1" + (self.num_digits - 1) * "0")
        return min_mantissa * (10 ** (-self.exponent_range))

    def _round_float(self, f: float) -> float:
        """Rounds float to the closest in-range value."""
        abs_f = abs(f)
        abs_f = min(abs_f, self._max_abs_val)
        if abs_f < self._min_abs_val:
            # Decides whether to move to 0.0 or `min_abs_val`.
            zero_or_min = round(abs_f / self._min_abs_val)
            abs_f = self._min_abs_val * zero_or_min
        return abs_f if f >= 0 else -abs_f

    def to_tokens(self, f: float, /) -> list[str]:
        f = self._round_float(f)
        s = np.format_float_scientific(
            f,
            precision=self.num_digits - 1,
            min_digits=self.num_digits - 1,
            sign=True,
        )
        # We expect numpy to produce scientific notation of the form `+2.123e+4`.
        # It will round for us and ensure leading digit isn't zero, unless the
        # number is zero.
        m = re.fullmatch("([+-])([0-9.]*)e(.*)", s)
        if not m:
            raise RuntimeError(f"Unexpected numpy notation: {s}")
        sign: str = m.group(1)
        digits = list(m.group(2).replace(".", ""))
        exp = int(m.group(3)) - len(digits) + 1 if f else 0

        out = [sign] + digits + [f"E{exp}"]
        return [_to_token(s) for s in out]

    def from_tokens(self, tokens: list[str], /) -> float:
        primitives = [_from_token(t) for t in tokens]

        sign = -1 if primitives[0] == "-" else 1
        mantissa = int("".join(map(str, primitives[1:-1])))
        exp = int("".join(primitives[-1]).lstrip("E"))

        return float(sign * mantissa * 10**exp)

    ## PreTrainedTokenizer methods
    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text: str):
        """
        text: a string containing a float or a sequence of floats.
        Note: Original OmniPred only tokenizes a single float.
        This implementation tokenizes all floats in the text.
        """
        # Implement tokenization logic here
        floats = self.pattern.findall(text)

        if not floats:
            return [
                "<float>",
                "<nan>",
                "</float>",
            ]  # Return unk token if no floats found
        tokens = ["<float>"]  # Start with the BOS token
        # Convert each float to tokens
        for f in floats:
            if f.startswith("+") or f.startswith("-"):
                tokens.extend(self.to_tokens(float(f)))
            else:
                tokens.extend(self.to_tokens(float("+" + f)))  # Ensure sign is included
        tokens.append("</float>")
        return tokens

    def _convert_token_to_id(self, token):
        # return self.vocab.get(token, self.unk_token_id)
        return self.tokens_to_ids.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        # return self.ids_to_tokens.get(index, self.unk_token)
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (strings) into a float.
        `<+> <3> <1> <4> <0> <E-3>` -> "3.14"
        """
        if not tokens:
            return ""
        if tokens[0] != "<float>" or tokens[-1] != "</float>":
            raise ValueError("Tokens must start with <float> and end with </float>.")
        # Remove the <float> and </float> tokens
        tokens = tokens[1:-1]
        # split tokens into groups of digits and exponent
        num_objs = len(tokens) // self.num_tokens_per_obj
        objects = []
        for i in range(num_objs):
            obj_tokens = tokens[
                i * self.num_tokens_per_obj : (i + 1) * self.num_tokens_per_obj
            ]
            objects.append(self.from_tokens(obj_tokens))
        # Join the objects into a string representation
        if len(objects) == 1:
            return str(objects[0])
        else:
            return " ".join(str(obj) for obj in objects)

    def get_vocab(self):
        return self.tokens_to_ids

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        The Vocabulary is not saved because it is built dynamically based on the number of digits and exponent range.
        """
        pass


if __name__ == "__main__":
    tokenizer = P10Tokenizer()
    print(tokenizer.vocab_size)
    # vocab = tokenizer.all_tokens()
    # print("Vocabulary:", vocab)
    # print(tokenizer.get_vocab())
    # a = "3.14"
    # tokens = tokenizer._tokenize(a)
    # print("Tokens for '{}': {}".format(a, tokens))
    # b = "12345"
    # tokens = tokenizer._tokenize(b)
    # print("Tokens for '{}': {}".format(b, tokens))
    # c = "0.1234 -123.4 0.0"
    # tokens = tokenizer._tokenize(c)
    # print("Tokens for '{}': {}".format(c, tokens))
    # print(tokenizer.encode("3.14"))
    # print(tokenizer.decode(tokenizer.encode("3.14")))
    # print(tokenizer.decode(tokenizer.encode(c)))
