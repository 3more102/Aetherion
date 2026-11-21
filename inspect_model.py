"""Inspect model architecture without allocating CPU/GPU parameter storage.

This is a theoretical parameter-storage estimate, not a VRAM training benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json

import torch

from aetherion import Aetherion
from aetherion.presets import PRESET_NAMES, get_preset


def inspect_preset(name: str, weight_dtype: str = "bf16") -> dict:
    bytes_per_element = {"fp32": 4, "fp16": 2, "bf16": 2}
    if weight_dtype not in bytes_per_element:
        raise ValueError(f"Unsupported weight dtype: {weight_dtype}")
    config = get_preset(name)
    # torch.device("meta") creates tensor metadata only; no giant weight allocation.
    with torch.device("meta"):
        model = Aetherion(config)
    params = sum(p.numel() for p in model.parameters())
    embedding_params = model.embed.weight.numel()
    return {
        "name": name,
        "configuration": asdict(config),
        "parameters": params,
        "embedding_parameters": embedding_params,
        "tied_input_output_embeddings": model.lm_head.weight is model.embed.weight,
        "theoretical_weight_storage_mib": round(
            params * bytes_per_element[weight_dtype] / (1024 ** 2), 2
        ),
        "weight_dtype": weight_dtype,
        "caveat": (
            "Weight storage only: excludes gradients, optimizer states, "
            "activations, KV cache, temporary tensors and runtime overhead."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an Aetherion model preset")
    parser.add_argument("--preset", choices=PRESET_NAMES, default="aetherion-125m")
    parser.add_argument("--weight-dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    args = parser.parse_args()
    print(json.dumps(inspect_preset(args.preset, args.weight_dtype), indent=2))


if __name__ == "__main__":
    main()
