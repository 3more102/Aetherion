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
    args = parser.parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Aetherion(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    prompt_bytes = args.prompt.encode("utf-8")
    if not prompt_bytes:
        parser.error("The prompt must contain at least one character")
    tokens = torch.tensor([list(prompt_bytes)], device=device, dtype=torch.long)
    result = model.generate(tokens, max_new_tokens=args.tokens, temperature=args.temperature)
    print(bytes(result[0].tolist()).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
