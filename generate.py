"""Generate from a byte-mode or BPE-mode Aetherion Micro checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aetherion import Aetherion, ModelConfig
from aetherion.token_data import sha256_file
from train import resolve_device


def main():
    parser = argparse.ArgumentParser(description="Aetherion Micro text generation")
    parser.add_argument("--checkpoint", default="checkpoints/demo.pt")
    parser.add_argument("--tokenizer", default=None, help="BPE tokenizer JSON used during training")
    parser.add_argument("--prompt", default="Artificial intelligence")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.tokens < 0 or args.temperature <= 0:
        parser.error("--tokens must be >= 0 and --temperature must be > 0")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**checkpoint["config"])
    info = checkpoint.get("tokenizer", {"mode": "byte"})
    mode = info.get("mode", "byte")

    if mode == "bpe":
        if not args.tokenizer:
            parser.error("BPE checkpoint requires --tokenizer FILE")
        if sha256_file(args.tokenizer) != info["sha256"]:
            raise ValueError("Tokenizer SHA-256 hash does not match checkpoint")
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(Path(args.tokenizer)))
        if tokenizer.get_vocab_size() != cfg.vocab_size:
            raise ValueError("Tokenizer vocabulary size differs from checkpoint")
        ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    elif mode == "byte":
        if args.tokenizer:
            parser.error("--tokenizer is only valid for a BPE checkpoint")
        tokenizer = None
        ids = list(args.prompt.encode("utf-8"))
    else:
        raise ValueError(f"Unsupported checkpoint tokenizer mode: {mode}")

    if not ids:
        parser.error("The prompt must contain at least one token")
    if max(ids) >= cfg.vocab_size:
        raise ValueError("Prompt contains an out-of-vocabulary token")

    device = resolve_device(args.device)
    model = Aetherion(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    tokens = torch.tensor([ids], device=device, dtype=torch.long)
    result = model.generate(
        tokens, max_new_tokens=args.tokens, temperature=args.temperature
    )[0].tolist()
    if tokenizer is None:
        print(bytes(result).decode("utf-8", errors="replace"))
    else:
        print(tokenizer.decode(result, skip_special_tokens=True))


if __name__ == "__main__":
    main()
