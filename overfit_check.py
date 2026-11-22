"""M16: CPU-only, seed-controlled synthetic next-token learning check.

This diagnostic trains a tiny ephemeral model on one repeated toy sequence.
It proves a learning signal, NOT held-out generalization or pretrained quality.
"""
import argparse
import json
import math

import torch
from aetherion import Aetherion, ModelConfig


def overfit_probe(*, steps=40, seed=17):
    if not isinstance(steps, int) or not 1 <= steps <= 500:
        raise ValueError("steps must be between 1 and 500")
    if not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=16, block_size=8, d_model=32,
                      n_heads=2, n_layers=1, dropout=0.0)
    model = Aetherion(cfg).cpu()
    x = torch.arange(1, 9, dtype=torch.long).unsqueeze(0)
    y = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 1]], dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    model.train()

    with torch.no_grad():
        initial = float(model(x, y)[1])
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite toy training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    with torch.no_grad():
        final = float(model(x, y)[1])
    if not math.isfinite(initial) or not math.isfinite(final):
        raise ValueError("Non-finite toy evaluation loss")

    ratio = final / initial
    return {
        "schema_version": 1, "diagnostic": "synthetic_training_overfit",
        "device": "cpu", "torch_version": str(torch.__version__),
        "seed": seed, "optimizer_steps": steps, "tokens_seen": steps * x.numel(),
        "unique_training_sequences": 1,
        "parameters": sum(p.numel() for p in model.parameters()),
        "initial_nll": initial, "final_nll": final, "final_to_initial_ratio": ratio,
        "pass_threshold": 0.5, "passed": final < 0.5 * initial,
        "checkpoint_saved": False,
        "limitations": "One synthetic memorized sequence; not held-out evaluation, generalization, or assistant capability.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    report = overfit_probe(steps=args.steps, seed=args.seed)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
