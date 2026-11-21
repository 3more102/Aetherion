"""Train Aetherion Micro on UTF-8 bytes with interruption-safe checkpoints.

The byte-stream demo is an engineering baseline, not evidence of language ability.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import os
from pathlib import Path
import tempfile

import torch
from aetherion import Aetherion, ModelConfig


WEIGHT_DECAY = 0.1
CHECKPOINT_VERSION = 2


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


def atomic_save_checkpoint(state: dict, output: Path) -> None:
    """Replace a checkpoint only after torch.save completes successfully."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, output)
    finally:
        Path(name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Aetherion Micro training")
    parser.add_argument("--data", default="data/demo.txt")
    parser.add_argument("--val-data", default=None, help="Separate validation text from prepare_data.py")
    parser.add_argument("--steps", type=int, default=200,
                        help="Total optimizer steps, including steps before --resume")
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
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save after each N updates (0 means only final checkpoint)")
    parser.add_argument("--resume", default=None, help="Resume full training state from a v2 checkpoint")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1 or args.save_every < 0:
        parser.error("--steps, --batch-size, --eval-every must be >= 1; --save-every >= 0")
    if args.lr <= 0:
        parser.error("--lr must be > 0")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    train, val, hashes = load_data(args.data, args.val_data)
    if min(len(train), len(val)) < args.block_size + 2:
        raise ValueError("Train and validation data must each contain at least block_size+2 UTF-8 bytes")

    cfg = ModelConfig(block_size=args.block_size, d_model=args.d_model,
                      n_layers=args.layers, n_heads=args.heads)
    run_config = {"batch_size": args.batch_size, "learning_rate": args.lr,
                  "weight_decay": WEIGHT_DECAY, "seed": args.seed}
    model = Aetherion(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    sampler = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    start_step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("Resume requires a v2 training checkpoint with optimizer and RNG state")
        if checkpoint.get("config") != asdict(cfg):
            raise ValueError("Model configuration does not match resume checkpoint")
        if checkpoint.get("data") != hashes:
            raise ValueError("Dataset hashes or train/validation split do not match resume checkpoint")
        if checkpoint.get("training_config") != run_config:
            raise ValueError("Training parameters do not match resume checkpoint")
        start_step = checkpoint["step"]
        if not isinstance(start_step, int) or start_step < 1 or start_step >= args.steps:
            raise ValueError("--steps must be larger than the saved completed step")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        sampler.set_state(checkpoint["sampler_rng_state"])
        torch.set_rng_state(checkpoint["cpu_rng_state"])
        if device.startswith("cuda"):
            if "cuda_rng_state_all" not in checkpoint:
                raise ValueError("Checkpoint lacks CUDA RNG state required for CUDA resume")
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        print(f"resumed_from={args.resume} completed_step={start_step}")

    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")

    def get_batch(split: torch.Tensor, generator: torch.Generator):
        starts = torch.randint(0, len(split) - cfg.block_size, (args.batch_size,), generator=generator)
        x = torch.stack([split[i:i + cfg.block_size] for i in starts])
        y = torch.stack([split[i + 1:i + cfg.block_size + 1] for i in starts])
        return x.to(device), y.to(device)

    def save(step: int):
        state = {"checkpoint_version": CHECKPOINT_VERSION,
                 "config": asdict(cfg), "state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "step": step, "seed": args.seed, "data": hashes,
                 "training_config": run_config,
                 "sampler_rng_state": sampler.get_state(),
                 "cpu_rng_state": torch.get_rng_state()}
        if device.startswith("cuda"):
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        atomic_save_checkpoint(state, Path(args.output))
        print(f"saved_checkpoint={args.output} step={step}")

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        x, y = get_batch(train, sampler)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            # Evaluation never changes training RNG, including at a resume boundary.
            eval_sampler = torch.Generator(device="cpu").manual_seed(args.seed + 1000003 * step)
            with torch.inference_mode():
                vx, vy = get_batch(val, eval_sampler)
                _, val_loss = model(vx, vy)
            print(f"step={step:4d} train_loss={loss.item():.3f} val_loss={val_loss.item():.3f}")
        if args.save_every and step % args.save_every == 0:
            save(step)

    if not args.save_every or args.steps % args.save_every:
        save(args.steps)


if __name__ == "__main__":
    main()
