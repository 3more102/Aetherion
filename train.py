"""Train Aetherion Micro with bytes or a verified experimental BPE tokenizer."""
import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from aetherion import Aetherion, ModelConfig
from aetherion.encoding import load_bpe_splits, sha256_file


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_training_data(args):
    """Return train IDs, validation IDs, vocabulary size and provenance."""
    if args.tokenizer:
        if not args.val_data:
            raise ValueError("BPE training requires --val-data validation.jsonl")
        train, val, metadata = load_bpe_splits(args.data, args.val_data, args.tokenizer)
        return train, val, metadata["tokenizer"]["vocab_size"], metadata

    if Path(args.data).suffix == ".jsonl":
        raise ValueError("JSONL input requires --tokenizer and --val-data")
    raw = torch.tensor(list(Path(args.data).read_bytes()), dtype=torch.long)
    if args.val_data:
        if Path(args.data).resolve() == Path(args.val_data).resolve():
            raise ValueError("--data and --val-data must be different files")
        validation = torch.tensor(list(Path(args.val_data).read_bytes()), dtype=torch.long)
        if sha256_file(args.data) == sha256_file(args.val_data):
            raise ValueError("Train and validation data must not have identical content")
        train, val = raw, validation
        policy = "explicit-byte-files"
    else:
        cutoff = int(0.9 * len(raw))
        train, val = raw[:cutoff], raw[cutoff:]
        policy = "last-10-percent-of-bytes (demo only)"

    return train, val, 256, {
        "tokenizer": {"mode": "byte", "vocab_size": 256},
        "dataset": {
            "split_policy": policy,
            "train_sha256": sha256_file(args.data),
            "validation_sha256": sha256_file(args.val_data) if args.val_data else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Aetherion Micro training")
    parser.add_argument("--data", default="data/demo.txt")
    parser.add_argument("--val-data", help="Independent validation text or prepared validation.jsonl")
    parser.add_argument("--tokenizer", help="BPE tokenizer.json and matching .meta.json sidecar")
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
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1:
        parser.error("--steps, --batch-size, and --eval-every must be >= 1")
    if args.block_size < 2:
        parser.error("--block-size must be >= 2")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    try:
        train, val, vocab_size, metadata = load_training_data(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.error(str(exc))
    if min(len(train), len(val)) < args.block_size + 2:
        parser.error("Each split must contain at least block-size+2 token IDs")

    cfg = ModelConfig(vocab_size=vocab_size, block_size=args.block_size,
                      d_model=args.d_model, n_layers=args.layers, n_heads=args.heads)
    model = Aetherion(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,} "
          f"tokenizer={metadata['tokenizer']['mode']} vocabulary={vocab_size}")

    def get_batch(split: torch.Tensor):
        starts = torch.randint(0, len(split) - cfg.block_size, (args.batch_size,))
        x = torch.stack([split[i:i + cfg.block_size] for i in starts])
        y = torch.stack([split[i + 1:i + cfg.block_size + 1] for i in starts])
        return x.to(device), y.to(device)

    for step in range(1, args.steps + 1):
        model.train()
        x, y = get_batch(train)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.inference_mode():
                vx, vy = get_batch(val)
                _, val_loss = model(vx, vy)
            print(f"step={step:4d} train_loss={loss.item():.3f} val_loss={val_loss.item():.3f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": asdict(cfg), "state_dict": model.state_dict(),
        "step": args.steps, "seed": args.seed,
        "tokenizer": metadata["tokenizer"], "dataset": metadata["dataset"],
        "tokenizer_sha256": metadata["tokenizer"].get("sha256"),
        "data_sha256": metadata["dataset"]["train_sha256"],
        "val_data_sha256": metadata["dataset"]["validation_sha256"],
        "train_tokens": len(train), "val_tokens": len(val),
    }, output)
    print(f"saved_checkpoint={output}")


if __name__ == "__main__":
    main()
