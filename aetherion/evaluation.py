"""Exact next-byte negative log likelihood over non-overlapping input windows.

Each target byte (after the first) is scored exactly once. Context is reset at
window boundaries, so results must include the window size for comparability.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


@torch.inference_mode()
def evaluate_tokens(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    block_size: int,
    device: str | torch.device = "cpu",
) -> dict[str, float | int]:
    """Evaluate a one-dimensional tensor of token IDs with teacher forcing."""
    if tokens.ndim != 1 or tokens.numel() < 2:
        raise ValueError("tokens must be a 1D tensor with at least two tokens")
    if not 1 <= block_size <= model.cfg.block_size:
        raise ValueError("block_size must be within [1, model.cfg.block_size]")

    was_training = model.training
    model.eval()
    nll_sum = 0.0
    target_count = 0
    try:
        for start in range(0, tokens.numel() - 1, block_size):
            length = min(block_size, tokens.numel() - 1 - start)
            x = tokens[start : start + length].unsqueeze(0).to(device)
            y = tokens[start + 1 : start + length + 1].to(device)
            logits, _ = model(x)
            nll_sum += F.cross_entropy(logits[0], y, reduction="sum").item()
            target_count += length
    finally:
        model.train(was_training)

    nll = nll_sum / target_count
    return {
        "nll": nll,
        "perplexity": math.exp(nll) if nll < 700 else float("inf"),
        "tokens": target_count,
        "context_window": block_size,
    }
