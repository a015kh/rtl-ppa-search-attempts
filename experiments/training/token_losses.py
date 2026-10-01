"""Composable historical token objective without importing the training runtime.

This preserves the original CE + Number Token Loss calculation. Number Token
Loss is an optional external dependency; its implementation is not vendored.
"""

from importlib import import_module


def make_number_token_loss(**kwargs):
    """Load the optional loss only when it is requested."""
    try:
        loss_class = import_module("ntloss").NTLoss
    except ImportError as exc:
        raise ImportError(
            "Number Token Loss requires ntloss==0.3.1. "
            "Install the experiments extra: "
            "pip install '.[experiments]'."
        ) from exc
    return loss_class(**kwargs)


def compose_token_losses(
    logits, labels, ce_loss, enabled_losses, ce_weight, ntloss_weight, ntloss_fn
):
    """Return the historical weighted total and unweighted numeric component.

    The model supplies CE (including its masked-label behavior). NTL receives
    the same labels and the historical ``ignore_index=-100`` explicitly.
    """
    total_loss = logits.new_tensor(0.0)
    if "ce" in enabled_losses and ce_weight != 0:
        total_loss = total_loss + ce_weight * ce_loss

    nt_loss = logits.new_tensor(0.0)
    if "ntloss" in enabled_losses and ntloss_weight != 0:
        if ntloss_fn is None:
            raise RuntimeError("NTLoss is enabled but was not initialized.")
        nt_loss = ntloss_fn(logits, labels, ignore_index=-100)
        total_loss = total_loss + ntloss_weight * nt_loss
    return total_loss, nt_loss
