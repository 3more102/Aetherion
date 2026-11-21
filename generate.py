"""Generate bytes from an Aetherion Micro checkpoint."""
import argparse
import torch
from aetherion import Aetherion, ModelConfig
from train import resolve_device


def main():
    parser = argparse.ArgumentParser(description="Aetherion Micro text generation")
    parser.add_argument("--checkpoint", default="checkpoints/demo.pt")
    parser.add_argument("--prompt", default="Artificial intelligence")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokenizer", default=None, help="Required tokenizer.json for BPE checkpoints")
    args = parser.parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Aetherion(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    encoding = checkpoint.get("data", {}).get("tokenizer_type", "bytes")
    if encoding == "bpe":
        if not args.tokenizer:
            parser.error("This checkpoint requires --tokenizer")
        from aetherion.tokenizer_runtime import load_verified_tokenizer
        tokenizer, metadata = load_verified_tokenizer(args.tokenizer)
        if (metadata["tokenizer_sha256"] != checkpoint["data"].get("tokenizer_sha256")
                or tokenizer.get_vocab_size() != model.cfg.vocab_size):
            parser.error("Tokenizer artifact does not match checkpoint")
        token_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
        decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
    elif encoding == "bytes":
        if args.tokenizer:
            parser.error("--tokenizer cannot be used for a byte-level checkpoint")
        token_ids = list(args.prompt.encode("utf-8"))
        decode = lambda ids: bytes(ids).decode("utf-8", errors="replace")
    else:
        parser.error(f"Unsupported tokenizer type: {encoding}")
    if not token_ids:
        parser.error("The prompt must contain at least one character")
    tokens = torch.tensor([token_ids], device=device, dtype=torch.long)
    result = model.generate(tokens, max_new_tokens=args.tokens, temperature=args.temperature)
    print(decode(result[0].tolist()))


if __name__ == "__main__":
    main()
