"""Local single-turn CLI for Aetherion SFT checkpoints (not a hosted service)."""
import argparse
import torch

from aetherion import Aetherion, ModelConfig
from aetherion.instruct import TEMPLATE, prompt_ids
from aetherion.tokenizer_runtime import load_verified_tokenizer
from train import resolve_device


def load_chat(checkpoint_path, tokenizer_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    info = checkpoint.get("data", {})
    tokenizer, meta = load_verified_tokenizer(tokenizer_path)
    cfg = ModelConfig(**checkpoint["config"])
    if (info.get("stage") != "sft" or info.get("template") != TEMPLATE
            or info.get("tokenizer_sha256") != meta["tokenizer_sha256"]
            or cfg.vocab_size != tokenizer.get_vocab_size()):
        raise ValueError("Chat requires a matching Aetherion SFT checkpoint and tokenizer")
    model = Aetherion(cfg)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), tokenizer


def reply(model, tokenizer, text, tokens=64, temperature=0.0):
    prefix = prompt_ids(tokenizer, text)
    if tokens < 1 or len(prefix) + tokens > model.cfg.block_size:
        raise ValueError("Prompt plus generation budget exceeds context")
    prompt = torch.tensor([prefix], dtype=torch.long, device=next(model.parameters()).device)
    output = model.generate_cached(prompt, max_new_tokens=tokens, temperature=temperature,
                                   eos_id=tokenizer.token_to_id("<|eos|>"))
    return tokenizer.decode(output[0, len(prefix):].tolist(), skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--prompt")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()
    model, tok = load_chat(args.checkpoint, args.tokenizer, resolve_device(args.device))
    if args.prompt is not None:
        print(reply(model, tok, args.prompt, args.tokens, args.temperature))
        return
    print("Aetherion experimental single-turn chat. No conversation memory. /exit to quit.")
    while True:
        try:
            prompt = input("You: ")
            if prompt == "/exit":
                break
            print("Aetherion:", reply(model, tok, prompt, args.tokens, args.temperature))
        except (EOFError, KeyboardInterrupt):
            break
        except ValueError as exc:
            print("Invalid prompt:", exc)


if __name__ == "__main__":
    main()
