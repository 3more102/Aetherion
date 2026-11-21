"""Train or deterministically resume Aetherion Micro from UTF-8 bytes."""
import argparse
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import torch
from aetherion import Aetherion, ModelConfig
from aetherion.checkpoint import (capture_rng_state, load_resume,
                                  restore_rng_state, save_checkpoint)


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
    hashes = {"train_sha256": sha256(train_bytes).hexdigest(),
              "validation_sha256": sha256(val_bytes).hexdigest(),
              "explicit_validation_file": val_path is not None}
    return (torch.tensor(list(train_bytes), dtype=torch.long),
            torch.tensor(list(val_bytes), dtype=torch.long), hashes)


def run_training(args) -> None:
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1 or args.save_every < 0:
        raise ValueError("steps, batch-size, eval-every must be >= 1 and save-every must be >= 0")
    if args.lr <= 0:
        raise ValueError("Learning rate must be > 0")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    train, val, hashes = load_data(args.data, args.val_data)
    if min(len(train), len(val)) < args.block_size + 2:
        raise ValueError("Train and validation data must each contain at least block_size+2 UTF-8 bytes")

    cfg = ModelConfig(block_size=args.block_size, d_model=args.d_model,
                      n_layers=args.layers, n_heads=args.heads)
    settings = {"batch_size": args.batch_size, "lr": args.lr,
                "eval_every": args.eval_every, "seed": args.seed,
                "optimizer": "AdamW", "weight_decay": 0.1}
    model = Aetherion(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    first_step = 1
    if args.resume:
        ckpt = load_resume(args.resume, config=asdict(cfg), data_hashes=hashes,
                           settings=settings, device=device)
        if args.steps <= ckpt["step"]:
            raise ValueError(f"--steps must exceed checkpoint step ({ckpt['step']})")
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        restore_rng_state(ckpt["rng_state"], device)
        first_step = ckpt["step"] + 1
        print(f"resuming_from_step={ckpt['step']}")
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")

    def get_batch(split: torch.Tensor, generator: torch.Generator | None = None):
        starts = torch.randint(0, len(split) - cfg.block_size, (args.batch_size,), generator=generator)
        x = torch.stack([split[i:i + cfg.block_size] for i in starts])
        y = torch.stack([split[i + 1:i + cfg.block_size + 1] for i in starts])
        return x.to(device), y.to(device)

    def write_checkpoint(step: int) -> None:
        save_checkpoint(args.output, {
            "config": asdict(cfg), "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": capture_rng_state(device),
            "step": step, "seed": args.seed, "data": hashes,
            "training": settings, "device_type": torch.device(device).type,
            "checkpoint_schema": 2,
        })
        print(f"saved_checkpoint={args.output} step={step}")

    for step in range(first_step, args.steps + 1):
        model.train()
        x, y = get_batch(train)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            # A dedicated seed makes validation sampling independent of the
            # training RNG, including extra final-step evaluations on short runs.
            val_rng = torch.Generator().manual_seed(args.seed + 1_000_003 + step)
            with torch.inference_mode():
                vx, vy = get_batch(val, generator=val_rng)
                _, val_loss = model(vx, vy)
            print(f"step={step:4d} train_loss={loss.item():.3f} val_loss={val_loss.item():.3f}")
        if step == args.steps or (args.save_every and step % args.save_every == 0):
            write_checkpoint(step)


def main():
    parser = argparse.ArgumentParser(description="Aetherion Micro training")
    parser.add_argument("--data", default="data/demo.txt")
    parser.add_argument("--val-data", default=None, help="Separate validation text from prepare_data.py")
    parser.add_argument("--steps", type=int, default=200, help="Total training steps, including completed resumed steps")
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
    parser.add_argument("--resume", default=None, help="Complete training checkpoint to continue from")
    parser.add_argument("--save-every", type=int, default=0, help="Periodic checkpoint interval; 0 saves only at the end")
    args = parser.parse_args()
    try:
        run_training(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
