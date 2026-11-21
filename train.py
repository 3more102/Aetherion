"""Train Aetherion Micro on demo bytes or held-out BPE JSONL splits."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from aetherion import Aetherion, ModelConfig
from aetherion.token_data import load_training_splits, sha256_file


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Aetherion Micro training")
    parser.add_argument("--data", default="data/demo.txt", help="Demo text or prepared train.jsonl")
    parser.add_argument("--val-data", default=None, help="Separate validation.jsonl or text")
    parser.add_argument("--tokenizer", default=None, help="Optional trained BPE tokenizer.json")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="checkpoints/demo.pt")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.block_size,
           args.d_model, args.layers, args.heads) < 1 or args.lr <= 0:
        parser.error("steps, batch-size, eval-every, dimensions and lr must be positive")

    tokenizer = None
    tokenizer_meta = {"mode": "byte", "vocab_size": 256}
    if args.tokenizer:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(args.tokenizer)
        tokenizer_meta = {
            "mode": "bpe",
            "vocab_size": tokenizer.get_vocab_size(),
            "sha256": sha256_file(args.tokenizer),
        }

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    train, val, hashes = load_training_splits(args.data, args.val_data, tokenizer)
    if min(train.numel(), val.numel()) < args.block_size + 1:
        raise ValueError("Both splits need at least block-size + 1 tokens")
    cfg = ModelConfig(
        vocab_size=tokenizer_meta["vocab_size"],
        block_size=args.block_size,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
    )
    if max(int(train.max()), int(val.max())) >= cfg.vocab_size:
        raise ValueError("Dataset contains token IDs outside the model vocabulary")

    model = Aetherion(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"train_tokens={train.numel()} validation_tokens={val.numel()} vocab={cfg.vocab_size}")

    # Independent CPU generators make validation sampling reproducible without
    # consuming the training batch RNG stream.
    train_rng = torch.Generator().manual_seed(args.seed + 1)
    val_rng = torch.Generator().manual_seed(args.seed + 2)

    def get_batch(split: torch.Tensor, generator: torch.Generator):
        starts = torch.randint(
            0, split.numel() - cfg.block_size,
            (args.batch_size,), generator=generator,
        ).tolist()
        x = torch.stack([split[i:i + cfg.block_size] for i in starts])
        y = torch.stack([split[i + 1:i + cfg.block_size + 1] for i in starts])
        return x.to(device), y.to(device)

    for step in range(1, args.steps + 1):
        model.train()
        x, y = get_batch(train, train_rng)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.inference_mode():
                vx, vy = get_batch(val, val_rng)
                _, val_loss = model(vx, vy)
            print(f"step={step:4d} train_loss={loss.item():.3f} val_loss={val_loss.item():.3f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(cfg),
            "state_dict": model.state_dict(),
            "step": args.steps,
            "seed": args.seed,
            "data": hashes,
            "tokenizer": tokenizer_meta,
            "tokens_seen": args.steps * args.batch_size * cfg.block_size,
        },
        output,
    )
    print(f"saved_checkpoint={output}")


if __name__ == "__main__":
    main()
