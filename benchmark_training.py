"""Measure synthetic optimizer updates; no model-quality claims."""
import argparse
from dataclasses import replace
import json
import platform
import time

import torch
from aetherion import Aetherion
from aetherion.presets import get_preset
from aetherion.training import optimizer_step, precision_context
from train import resolve_device


def benchmark(*, preset="micro", device="cpu", precision="fp32", batch_size=1,
              accumulation=1, steps=3, warmup=1, context=None):
    if min(batch_size, accumulation, steps) < 1 or warmup < 0:
        raise ValueError("Invalid benchmark counts")
    device = torch.device(resolve_device(device))
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Benchmark supports CPU and CUDA")
    precision_context(device, precision)
    torch.manual_seed(42)
    cfg = get_preset(preset)
    if context is not None:
        if not 2 <= context <= cfg.block_size:
            raise ValueError("Invalid context override")
        cfg = replace(cfg, block_size=context)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = Aetherion(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    def batch():
        tokens = torch.randint(cfg.vocab_size, (batch_size, cfg.block_size + 1), device=device)
        return tokens[:, :-1], tokens[:, 1:]
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    for _ in range(warmup):
        optimizer_step(model, optimizer, batch, accumulation=accumulation, device=device, precision=precision)
    sync()
    start = time.perf_counter()
    for _ in range(steps):
        loss = optimizer_step(model, optimizer, batch, accumulation=accumulation, device=device, precision=precision)
    sync()
    seconds = time.perf_counter() - start
    tokens = steps * batch_size * accumulation * cfg.block_size
    return {"synthetic": True, "preset": preset, "device": str(device),
            "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or platform.machine(),
            "torch_version": str(torch.__version__), "precision": precision,
            "parameters": sum(p.numel() for p in model.parameters()),
            "context": cfg.block_size, "batch_size": batch_size, "accumulation": accumulation,
            "warmup_updates": warmup, "measured_updates": steps, "tokens": tokens,
            "seconds": seconds, "tokens_per_second": tokens / seconds, "last_loss": loss,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=("micro", "125m"), default="micro")
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulation", type=int, default=1)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--context", type=int)
    print(json.dumps(benchmark(**vars(p.parse_args())), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
