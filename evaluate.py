"""Deterministic held-out next-byte evaluation for Aetherion Micro.

Metrics are computed on the last 10% of bytes, matching train.py.
This is an engineering baseline, not a measure of assistant quality.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from aetherion import Aetherion, ModelConfig
from train import resolve_device


def validation_tokens(raw: bytes) -> torch.Tensor:
    """Use the same 90/10 contiguous byte split as train.py."""
    cutoff = int(0.9 * len(raw))
    held_out = raw[cutoff:]
    if len(held_out) < 2:
        raise ValueError("Validation split must contain at least two bytes")
    return torch.tensor(list(held_out), dtype=torch.long)


@torch.inference_mode()
def evaluate_tokens(
    model: Aetherion, tokens: torch.Tensor, device: str = "cpu"
) -> tuple[float, int]:
    """Evaluate every held-out next-byte target exactly once.

    Reset context at each non-overlapping window; report this segmentation
    policy when comparing runs. No tokens from the training split are used.
    """
    if tokens.ndim != 1 or tokens.numel() < 2:
        raise ValueError("Expected a 1D token sequence of at least length 2")
    model.eval()
    total_nll = 0.0
    total_targets = 0
    for start in range(0, tokens.numel() - 1, model.cfg.block_size):
        window = tokens[start : start + model.cfg.block_size + 1]
        x = window[:-1].unsqueeze(0).to(device)
        y = window[1:].unsqueeze(0).to(device)
        logits, _ = model(x)
        total_nll += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        ).item()
        total_targets += y.numel()
    return total_nll / total_targets, total_targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Aetherion Micro held-out loss")
    parser.add_argument("--checkpoint", default="checkpoints/demo.pt")
    parser.add_argument("--data", default="data/demo.txt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    raw = Path(args.data).read_bytes()
    tokens = validation_tokens(raw)
    device = resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Aetherion(ModelConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    loss, count = evaluate_tokens(model, tokens, device)
    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = None

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "data": str(args.data),
        "data_sha256": hashlib.sha256(raw).hexdigest(),
        "split": "last 10% of UTF-8 bytes; no overlapping windows",
        "checkpoint_step": ckpt.get("step"),
        "checkpoint_seed": ckpt.get("seed"),
        "model_config": ckpt["config"],
        "device": device,
        "heldout_bytes": int(tokens.numel()),
        "evaluated_next_byte_targets": count,
        "cross_entropy_nats_per_byte": round(loss, 6),
        "perplexity_per_byte": round(perplexity, 6) if perplexity is not None else None,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
