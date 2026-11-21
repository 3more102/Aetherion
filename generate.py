"""Generate text from an Aetherion Micro checkpoint in byte or BPE mode."""
import argparse

import torch

from aetherion import Aetherion, ModelConfig
from aetherion.encoding import load_bpe, sha256_file
from train import resolve_device


def main():
    parser = argparse.ArgumentParser(description="Aetherion Micro text generation")
    parser.add_argument("--checkpoint", default="checkpoints/demo.pt")
    parser.add_argument("--tokenizer", help="Tokenizer JSON used to train a BPE checkpoint")
    parser.add_argument("--prompt", default="Artificial intelligence")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.tokens < 0 or args.temperature <= 0:
        parser.error("--tokens must be >= 0 and --temperature must be > 0")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**checkpoint["config"])
    encoding = checkpoint.get("encoding", "byte")  # Support older Micro checkpoints.
    tokenizer = None
    if encoding == "bpe":
        if not args.tokenizer:
            parser.error("This checkpoint requires --tokenizer")
        expected = checkpoint.get("tokenizer_sha256")
        if not expected or sha256_file(args.tokenizer) != expected:
            raise ValueError("Tokenizer SHA-256 does not match the training checkpoint")
        tokenizer = load_bpe(args.tokenizer)
        if tokenizer.get_vocab_size() != cfg.vocab_size:
            raise ValueError("Tokenizer vocabulary size differs from checkpoint")
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    elif encoding == "byte":
        if args.tokenizer:
            parser.error("Byte-mode checkpoints must not use --tokenizer")
        prompt_ids = list(args.prompt.encode("utf-8"))
    else:
        raise ValueError(f"Unsupported checkpoint encoding: {encoding}")

    if not prompt_ids:
        parser.error("The prompt must contain at least one token")

    device = resolve_device(args.device)
    model = Aetherion(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    tokens = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    result = model.generate(tokens, max_new_tokens=args.tokens, temperature=args.temperature)
    ids = result[0].tolist()
    if tokenizer is not None:
        print(tokenizer.decode(ids, skip_special_tokens=True))
    else:
        print(bytes(ids).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
