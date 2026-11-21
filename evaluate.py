"""Evaluate an Aetherion checkpoint on held-out next-byte prediction.

By default, reproduces train.py's final-10%-by-byte validation split.
The bundled demo corpus contains repeated text and is NOT a real benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from aetherion import Aetherion, ModelConfig
from aetherion.evaluation import evaluate_tokens
from train import resolve_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate Aetherion Micro")
    parser.add_argument("--checkpoint", default="checkpoints/demo.pt")
    parser.add_argument("--data", default="data/demo.txt")
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Aetherion(ModelConfig(**saved["config"]))
    model.load_state_dict(saved["state_dict"])
    model.to(device)

    raw = Path(args.data).read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    expected_sha = saved.get("dataset_sha256")
    if expected_sha is not None and source_sha != expected_sha:
        parser.error("Dataset SHA-256 does not match checkpoint metadata")

    cutoff = int(0.9 * len(raw))
    selected = {"train": raw[:cutoff], "val": raw[cutoff:], "all": raw}[args.split]
    tokens = torch.tensor(list(selected), dtype=torch.long)
    block_size = args.block_size or model.cfg.block_size
    metrics = evaluate_tokens(model, tokens, block_size=block_size, device=device)
    print(json.dumps({"split": args.split, "dataset_sha256": source_sha, **metrics}, indent=2))


if __name__ == "__main__":
    main()
