"""Train Aetherion Micro on UTF-8 bytes or prepared BPE token IDs."""
import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from aetherion import Aetherion, ModelConfig
from aetherion.token_stream import file_sha256, load_token_stream


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
    parser.add_argument("--data", default="data/demo.txt")
    parser.add_argument("--val-data", help="Independent held-out text/JSONL; required with --tokenizer")
    parser.add_argument("--tokenizer", help="Trained ByteLevel BPE tokenizer.json; omit for legacy bytes")
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
    if args.tokenizer and not args.val_data:
        parser.error("--tokenizer requires independent --val-data")
    if args.val_data and Path(args.data).resolve() == Path(args.val_data).resolve():
        parser.error("--data and --val-data must differ")

    tokenizer = None
    tokenizer_hash = None
    vocab_size = 256
    if args.tokenizer:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(args.tokenizer)
        tokenizer_hash = file_sha256(args.tokenizer)
        vocab_size = tokenizer.get_vocab_size()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    train_hash = file_sha256(args.data)
    data = torch.tensor(load_token_stream(args.data, tokenizer), dtype=torch.long)
    if args.val_data:
        val_hash = file_sha256(args.val_data)
        if train_hash == val_hash:
            raise ValueError("Train/validation data are identical")
        train = data
        val = torch.tensor(load_token_stream(args.val_data, tokenizer), dtype=torch.long)
    else:
        cutoff = int(0.9 * len(data))
        train, val = data[:cutoff], data[cutoff:]
        val_hash = None
    if min(len(train), len(val)) < args.block_size + 2:
        raise ValueError("Train and validation splits each need at least block_size+2 tokens")

    cfg = ModelConfig(vocab_size=vocab_size, block_size=args.block_size,
                      d_model=args.d_model, n_layers=args.layers, n_heads=args.heads)
    model = Aetherion(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,} "
          f"tokenizer={'bpe' if tokenizer else 'byte'} "
          f"train_tokens={len(train)} val_tokens={len(val)}")

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
    torch.save({"config": asdict(cfg), "state_dict": model.state_dict(),
                "step": args.steps, "seed": args.seed,
                "tokenizer_sha256": tokenizer_hash,
                "data_sha256": train_hash, "val_data_sha256": val_hash,
                "train_tokens": len(train), "val_tokens": len(val)}, output)
    print(f"saved_checkpoint={output}")


if __name__ == "__main__":
    main()
