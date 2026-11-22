"""Evaluate the exact validation data recorded by a trusted v2 checkpoint."""
import argparse
from contextlib import redirect_stdout
import sys
from hashlib import sha256
import json
from pathlib import Path

import torch

from aetherion import Aetherion, ModelConfig
from aetherion.evaluation import evaluate_tokens
from train import load_bpe_data, load_data, resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/demo.pt")
    parser.add_argument("--data", default="data/demo.txt")
    parser.add_argument("--val-data")
    parser.add_argument("--tokenizer")
    parser.add_argument("--stride", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2 or not isinstance(checkpoint.get("data"), dict):
        parser.error("Evaluation requires a v2 checkpoint with dataset fingerprints")
    recorded = checkpoint["data"]
    encoding = recorded.get("tokenizer_type", "bytes")
    if encoding == "bpe":
        if not args.tokenizer:
            parser.error("BPE checkpoint requires --tokenizer")
        _, val, fingerprints, vocab_size = load_bpe_data(args.data, args.val_data, args.tokenizer)
    elif encoding == "bytes":
        if args.tokenizer:
            parser.error("Byte checkpoint cannot use --tokenizer")
        with redirect_stdout(sys.stderr):
            _, val, fingerprints = load_data(args.data, args.val_data)
        vocab_size = 256
    else:
        parser.error("Unsupported checkpoint tokenizer type")
    if recorded != fingerprints:
        parser.error("Dataset or tokenizer fingerprints differ from checkpoint")
    config = ModelConfig(**checkpoint["config"])
    if config.vocab_size != vocab_size:
        parser.error("Vocabulary size differs from checkpoint")
    model = Aetherion(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(resolve_device(args.device))
    report = evaluate_tokens(model, val, stride=args.stride)
    report.update({
        "schema_version": 1,
        "checkpoint_sha256": sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "checkpoint_step": checkpoint["step"],
        "tokenizer_type": encoding,
        "validation_sha256": recorded["validation_sha256"],
        "tokenizer_sha256": recorded.get("tokenizer_sha256"),
        "split_policy": "explicit_validation_file" if recorded["explicit_validation_file"] else "demo_byte_90_10",
    })
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
