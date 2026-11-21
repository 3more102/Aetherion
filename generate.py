"""Generate text from byte-level or BPE Aetherion Micro checkpoints."""
import argparse
import torch
from aetherion import Aetherion, ModelConfig
from aetherion.token_stream import file_sha256
from train import resolve_device


def main():
    parser = argparse.ArgumentParser(description="Aetherion Micro text generation")
    parser.add_argument("--checkpoint", default="checkpoints/demo.pt")
    parser.add_argument("--tokenizer", help="BPE tokenizer.json matching the checkpoint")
    parser.add_argument("--prompt", default="Artificial intelligence")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.tokens < 0 or args.temperature <= 0:
        parser.error("--tokens must be >= 0 and --temperature must be > 0")
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    expected_hash = checkpoint.get("tokenizer_sha256")
    if expected_hash and not args.tokenizer:
        parser.error("This checkpoint requires --tokenizer")
    if not expected_hash and args.tokenizer:
        parser.error("This checkpoint uses byte tokens; do not pass --tokenizer")
    tokenizer = None
    if args.tokenizer:
        if file_sha256(args.tokenizer) != expected_hash:
            raise ValueError("Tokenizer SHA-256 mismatch with checkpoint")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(args.tokenizer)
        if tokenizer.get_vocab_size() != checkpoint["config"]["vocab_size"]:
            raise ValueError("Tokenizer vocabulary size differs from checkpoint")

    model = Aetherion(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    if tokenizer is None:
        prompt_ids = list(args.prompt.encode("utf-8"))
    else:
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    if not prompt_ids:
        parser.error("The prompt must contain at least one token")
    tokens = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    result = model.generate(tokens, max_new_tokens=args.tokens, temperature=args.temperature)
    if tokenizer is None:
        print(bytes(result[0].tolist()).decode("utf-8", errors="replace"))
    else:
        print(tokenizer.decode(result[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
