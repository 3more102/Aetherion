"""Shared optimizer step, fixed-horizon schedule and BF16 autocast."""
from contextlib import nullcontext
import math
import torch


def learning_rate(step, *, peak, warmup=0, horizon=0, min_ratio=0.1):
    """One-indexed optimizer step; horizon=0 preserves constant LR."""
    if step < 1 or peak <= 0 or warmup < 0 or horizon < 0 or not 0 <= min_ratio <= 1:
        raise ValueError("Invalid learning-rate settings")
    if warmup and (not horizon or warmup >= horizon):
        raise ValueError("Warmup must be shorter than a positive schedule horizon")
    if not horizon:
        return peak
    if step > horizon:
        raise ValueError("Step exceeds fixed schedule horizon")
    if warmup and step <= warmup:
        return peak * step / warmup
    # No warmup: first update uses peak LR, final update uses floor.
    start = max(1, warmup)
    fraction = (step - start) / max(1, horizon - start)
    return peak * (min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * fraction)) / 2)


def precision_context(device, precision):
    kind = torch.device(device).type
    if precision == "fp32":
        return nullcontext()
    if precision != "bf16" or kind not in ("cpu", "cuda"):
        raise ValueError("BF16 autocast is supported on CPU and compatible CUDA devices only")
    if kind == "cuda":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("Selected CUDA device does not support BF16")
    return torch.autocast(device_type=kind, dtype=torch.bfloat16)


def optimizer_step(model, optimizer, batch, *, accumulation, device, precision):
    """Average equal-sized microbatch losses; clip once per optimizer update."""
    if accumulation < 1:
        raise ValueError("accumulation must be positive")
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=device)
    for _ in range(accumulation):
        x, y = batch()
        with precision_context(device, precision):
            _, loss = model(x, y)
        loss_sum += loss.detach().float()
        (loss / accumulation).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return (loss_sum / accumulation).item()
