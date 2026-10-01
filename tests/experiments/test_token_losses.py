"""Synthetic checks of the historical numeric objective and masked gradients."""

import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/training"))
import token_losses

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_NTLOSS = importlib.util.find_spec("ntloss") is not None
if HAS_TORCH:
    import torch
    import torch.nn.functional as F


class DigitTokenizer:
    """Tiny synthetic vocabulary; every digit already exists, plus text."""

    def __init__(self):
        self.vocab = {str(value): value for value in range(10)}
        self.vocab["text"] = 10

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def add_tokens(self, tokens):
        added = 0
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added


class MissingDependencyTests(unittest.TestCase):
    def test_optional_dependency_reports_installable_extra(self):
        with patch.object(token_losses, "import_module", side_effect=ModuleNotFoundError("ntloss")):
            with self.assertRaisesRegex(ImportError, r"ntloss==0\.3\.1.*experiments"):
                token_losses.make_number_token_loss(tokenizer=DigitTokenizer())


@unittest.skipUnless(HAS_TORCH, "Install the ppa extra for tensor objective checks.")
class CrossEntropyTests(unittest.TestCase):
    def test_ce_only_runs_without_optional_number_loss(self):
        logits = torch.zeros(1, 2, 11, requires_grad=True)
        labels = torch.tensor([[2, -100]])
        ce = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100)
        total, numeric = token_losses.compose_token_losses(logits, labels, ce, {"ce"}, 2.0, 0.3, None)
        self.assertAlmostEqual(total.item(), 2.0 * math.log(11), places=5)
        self.assertEqual(numeric.item(), 0)
        total.backward()
        self.assertEqual(logits.grad[0, 1].abs().sum().item(), 0)


@unittest.skipUnless(HAS_TORCH and HAS_NTLOSS, "Install the experiments extra for numeric loss checks.")
class NumberTokenLossTests(unittest.TestCase):
    def setUp(self):
        self.loss_fn = token_losses.make_number_token_loss(
            tokenizer=DigitTokenizer(), vocab_size=11,
            digit_level=True, reweigh=True, squash_factor=None,
        )

    def objective(self, logits, labels):
        ce = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100)
        return token_losses.compose_token_losses(
            logits, labels, ce, {"ce", "ntloss"}, 1.0, 0.3, self.loss_fn
        )

    def test_uniform_digits_have_expected_weighted_ce_plus_numeric_loss(self):
        logits = torch.zeros(1, 3, 11, requires_grad=True)
        labels = torch.tensor([[2, -100, 8]])
        total, numeric = self.objective(logits, labels)
        # The two digit targets have mean distances 3.1 and 3.7 under uniform
        # digits. Ten of eleven tokens are digits; text mass incurs max-distance
        # reweighting by the historical library's factor 1.01.
        expected_numeric = 3.4 * (10 / 11) + 1.01 * 9 * (1 / 11)
        self.assertAlmostEqual(numeric.item(), expected_numeric, places=5)
        self.assertAlmostEqual(total.item(), math.log(11) + 0.3 * expected_numeric, places=5)
        total.backward()
        self.assertGreater(logits.grad[0, 0].abs().sum().item(), 0)
        self.assertEqual(logits.grad[0, 1].abs().sum().item(), 0)

    def test_masked_logits_cannot_change_either_loss_component(self):
        labels = torch.tensor([[2, -100, 8]])
        logits = torch.zeros(1, 3, 11)
        before = self.objective(logits, labels)
        changed = logits.clone()
        changed[0, 1] = torch.arange(11) * 100.0
        after = self.objective(changed, labels)
        for original, altered in zip(before, after):
            self.assertEqual(original.item(), altered.item())


if __name__ == "__main__":
    unittest.main()
