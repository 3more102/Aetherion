"""Train Aetherion Micro from UTF-8 bytes (educational starter)."""
import argparse
from hashlib import sha256
from pathlib import Path

import torch
from aetherion import Aetherion, ModelConfig
from aetherion.checkpoint import load_training_checkpoint, save_training_checkpoint


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_data(train_path: str, val_path: str | None):
    """Read distinct prepared files when provided; preserve demo fallback."""
    train_bytes = Path(train_path).read_bytes()
    if val_path is None:
        cutoff = int(0.9 * len(train_bytes))
        print("WARNING: using a raw 90/10 byte split for demo only; use --val-data for real experiments")
        train_bytes, val_bytes = train_bytes[:cutoff], train_bytes[cutoff:]
    else:
        val_bytes = Path(val_path).read_bytes()
    train_sha = sha256(train_bytes).hexdigest()
    val_sha = sha256(val_bytes).hexdigest()
    return (torch.tensor(list(train_bytes), dtype=torch.long),
            torch.tensor(list(val_bytes), dtype=torch.long),
            {"train_sha256": train_sha, "validation_sha256": val_sha,
             "explicit_validation_file": val_path is not None})


def main():
    parser = argparse.ArgumentParser(description="Aetherion Micro training")
    parser.add_argument("--data", default="data/demo.txt")
    parser.add_argument("--val-data", default=None, help="Separate validation text from prepare_data.py")
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
    parser.add_argument("--resume", default=None, help="Resumable checkpoint from the same data/configuration")
    parser.add_argument("--save-every", type=int, default=0, help="Checkpoint interval in steps (0=final only)")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1:
        parser.error("--steps, --batch-size, and --eval-every must be >= 1")
    if args.save_every < 0:
        parser.error("--save-every must be >= 0")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    train, val, hashes = load_data(args.data, args.val_data)
    if min(len(train), len(val)) < args.block_size + 2:
        raise ValueError("Train and validation data must each contain at least block_size+2 UTF-8 bytes")

    cfg = ModelConfig(block_size=args.block_size, d_model=args.d_model,
                      n_layers=args.layers, n_heads=args.heads)
    model = Aetherion(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    training = {"batch_size": args.batch_size, "eval_every": args.eval_every, "lr": args.lr}
    start_step = 0
    if args.resume:
        start_step = load_training_checkpoint(
            args.resume, model=model, optimizer=optimizer, config=cfg,
            seed=args.seed, data=hashes, training=training, device=device,
        )
        if args.steps <= start_step:
            parser.error(f"--steps must exceed checkpoint step {start_step} (total target steps)")
        print(f"resumed_checkpoint={args.resume} at_step={start_step}")
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")

    def get_batch(split: torch.Tensor):
        starts = torch.randint(0, len(split) - cfg.block_size, (args.batch_size,))
        x = torch.stack([split[i:i + cfg.block_size] for i in starts])
        y = torch.stack([split[i + 1:i + cfg.block_size + 1] for i in starts])
        return x.to(device), y.to(device)

    output = Path(args.output)
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        x, y = get_batch(train)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            # Validation must not advance the training minibatch RNG stream.
            with torch.random.fork_rng(devices=[]), torch.inference_mode():
                vx, vy = get_batch(val)
                _, val_loss = model(vx, vy)
            print(f"step={step:4d} train_loss={loss.item():.3f} val_loss={val_loss.item():.3f}")
        if step == args.steps or (args.save_every and step % args.save_every == 0):
            save_training_checkpoint(
                output, model=model, optimizer=optimizer, config=cfg, step=step,
                seed=args.seed, data=hashes, training=training, device=device,
            )
    print(f"saved_checkpoint={output}")


if __name__ == "__main__":
    main()
