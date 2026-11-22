"""Deterministic next-token evaluation with exact target accounting."""
import math

import torch
from torch.nn import functional as F


@torch.inference_mode()
def evaluate_tokens(model, tokens: torch.Tensor, *, stride: int | None = None):
    """Score each target exactly once, using up to the configured context.

    A stride equal to context uses adjacent windows; smaller strides retain
    additional left context while scoring only new targets.
    """
    context = model.cfg.block_size
    stride = context if stride is None else stride
    if tokens.ndim != 1 or tokens.dtype != torch.long or tokens.numel() < 2:
        raise ValueError("Expected a one-dimensional long tensor with >= 2 tokens")
    if not 1 <= stride <= context:
        raise ValueError("stride must be between 1 and context size")
    device = next(model.parameters()).device
    was_training = model.training
    total_nll, count = 0.0, 0
    model.eval()
    try:
        for target_start in range(1, tokens.numel(), stride):
            end = min(target_start + stride, tokens.numel())
            start = max(0, end - 1 - context)
            x = tokens[start:end - 1].unsqueeze(0).to(device)
            y = tokens[target_start:end].to(device)
            logits, _ = model(x)
            selected = logits[0, -(end - target_start):].float()
            losses = F.cross_entropy(selected, y, reduction="none")
            total_nll += losses.double().sum().item()
            count += y.numel()
    finally:
        model.train(was_training)
    nll = total_nll / count
    if not math.isfinite(nll) or nll > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
        raise ValueError("Non-finite evaluation metrics")
    return {"evaluated_targets": count, "total_nll": total_nll,
            "mean_nll": nll, "perplexity": math.exp(nll),
            "context_size": context, "stride": stride}
