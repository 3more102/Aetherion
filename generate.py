"""Generate from an Aetherion Micro byte or BPE checkpoint."""
import argparse

import torch

from aetherion import Aetherion, ModelConfig
from aetherion.encoding import sha256_file
from train import resolve_device


def encode_prompt(prompt: str, checkpoint: dict, tokenizer_path: str | None):
    """Encode with the checkpoint's recorded tokenizer, never a silent fallback."""
    info = checkpoint.get("tokenizer")
    if not info:
        legacy_hash = checkpoint.get("tokenizer_sha256")
        info = {"mode": "bpe", "sha256": legacy_hash} if legacy_hash else {"mode": "byte"}
    mode = info.get("mode", "byte")
    if mode == "byte":
        if tokenizer_path:
            raise ValueError("Byte checkpoint cannot be used with --tokenizer")
        return list(prompt.encode("utf-8")), None
    if mode != "bpe":
        raise ValueError(f"Unsupported tokenizer mode: {mode}")
    if not tokenizer_path:
        raise ValueError("BPE checkpoint requires --tokenizer")
    if sha256_file(tokenizer_path) != info.get("sha256"):
        raise ValueError("Tokenizer SHA-256 mismatch with checkpoint")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    if tokenizer.get_vocab_size() != checkpoint["config"]["vocab_size"]:
        raise ValueError("Tokenizer vocabulary does not match model embeddings")
    return tokenizer.encode(prompt, add_special_tokens=False).ids, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Aetherion Micro text generation")
    parser.add_argument("--checkpoint", default="checkpoints/demo.pt")
    parser.add_argument("--tokenizer", help="Tokenizer JSON for BPE checkpoints")
    parser.add_argument("--prompt", default="Artificial intelligence")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.tokens < 0 or args.temperature <= 0:
        parser.error("--tokens must be >= 0 and --temperature must be > 0")
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    try:
        prompt_ids, tokenizer = encode_prompt(args.prompt, checkpoint, args.tokenizer)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if not prompt_ids:
        parser.error("The prompt must encode to at least one token")
    model = Aetherion(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    tokens = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    result = model.generate(tokens, max_new_tokens=args.tokens, temperature=args.temperature)
    ids = result[0].tolist()
    if tokenizer is None:
        print(bytes(ids).decode("utf-8", errors="replace"))
    else:
        print(tokenizer.decode(ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
