"""Generate from an Aetherion Micro checkpoint using its matching tokenizer."""
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
    parser.add_argument("--tokenizer-dir", default=None, help="Required for BPE checkpoints")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.tokens < 0:
        parser.error("--tokens must be >= 0")
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**checkpoint["config"])
    model = Aetherion(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    data = checkpoint.get("data", {})
    mode = data.get("tokenizer_type", "bytes")
    if mode == "bpe":
        if not args.tokenizer_dir:
            parser.error("This checkpoint requires --tokenizer-dir")
        from aetherion.tokenizer import AetherionTokenizer
        tokenizer = AetherionTokenizer.load(args.tokenizer_dir)
        if tokenizer.sha256 != data.get("tokenizer_sha256"):
            parser.error("BPE tokenizer checksum differs from checkpoint")
        if tokenizer.vocab_size != cfg.vocab_size:
            parser.error("BPE tokenizer vocabulary size differs from model")
        prompt_ids = tokenizer.encode(args.prompt)
        decode = tokenizer.decode
    elif mode == "bytes":
        if args.tokenizer_dir:
            parser.error("--tokenizer-dir cannot be used with byte-level demo checkpoints")
        if cfg.vocab_size != 256:
            parser.error("Legacy byte-level checkpoint must have a 256-token vocabulary")
        prompt_ids = list(args.prompt.encode("utf-8"))
        decode = lambda tokens: bytes(tokens).decode("utf-8", errors="replace")
    else:
        parser.error(f"Unsupported checkpoint tokenizer type: {mode}")
    if not prompt_ids:
        parser.error("The prompt must contain at least one character")
    ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    result = model.generate(ids, max_new_tokens=args.tokens, temperature=args.temperature)
    print(decode(result[0].tolist()))


if __name__ == "__main__":
    main()
