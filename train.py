"""Train Aetherion Micro with exact, auditable checkpoint resumption."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import os
from pathlib import Path
import tempfile

import torch

from aetherion import Aetherion, ModelConfig


CHECKPOINT_VERSION = 2
WEIGHT_DECAY = 0.1


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


def capture_rng(train_generator: torch.Generator, val_generator: torch.Generator,
                device: str) -> dict:
    """Save all RNG streams that can affect the next optimizer update."""
    result = {
        "cpu": torch.get_rng_state(),
        "train_sampler": train_generator.get_state(),
        "val_sampler": val_generator.get_state(),
    }
    if device.startswith("cuda"):
        result["cuda"] = torch.cuda.get_rng_state_all()
    if device.startswith("mps"):
        if not hasattr(torch.mps, "get_rng_state"):
            raise RuntimeError("This PyTorch version cannot checkpoint MPS RNG state")
        result["mps"] = torch.mps.get_rng_state()
    return result


def restore_rng(state: dict, train_generator: torch.Generator,
                val_generator: torch.Generator, device: str) -> None:
    """Restore RNG *after* model construction, which consumes random numbers."""
    for key in ("cpu", "train_sampler", "val_sampler"):
        if key not in state:
            raise ValueError(f"Checkpoint RNG state is missing {key}")
    torch.set_rng_state(state["cpu"])
    train_generator.set_state(state["train_sampler"])
    val_generator.set_state(state["val_sampler"])
    if device.startswith("cuda"):
        cuda_states = state.get("cuda")
        if not cuda_states or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA RNG state is incompatible with available GPUs")
        torch.cuda.set_rng_state_all(cuda_states)
    if device.startswith("mps"):
        if "mps" not in state or not hasattr(torch.mps, "set_rng_state"):
            raise ValueError("Checkpoint MPS RNG state cannot be restored")
        torch.mps.set_rng_state(state["mps"])


def save_checkpoint(path: Path, payload: dict) -> None:
    """Atomically replace a checkpoint: an interrupted write retains the old one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_resume(checkpoint: dict, config: ModelConfig, data_hashes: dict,
                    trainer: dict, requested_steps: int) -> int:
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            "Cannot resume this checkpoint: optimizer/RNG state or checkpoint version "
            "is missing. Old checkpoints still work for generation."
        )
    if checkpoint.get("config") != asdict(config):
        raise ValueError("Cannot resume: model configuration differs from checkpoint")
    if checkpoint.get("data") != data_hashes:
        raise ValueError("Cannot resume: train/validation data hashes or split mode changed")
    if checkpoint.get("trainer") != trainer:
        raise ValueError("Cannot resume: batch size, learning rate, seed, or device changed")
    for name in ("state_dict", "optimizer_state_dict", "rng_state"):
        if name not in checkpoint:
            raise ValueError(f"Cannot resume: checkpoint is missing {name}")
    step = checkpoint.get("step")
    if not isinstance(step, int) or step < 1 or requested_steps <= step:
        raise ValueError("For --resume, --steps must exceed the completed checkpoint step")
    return step


def main() -> None:
    parser = argparse.ArgumentParser(description="Aetherion Micro training")
    parser.add_argument("--data", default="data/demo.txt")
    parser.add_argument("--val-data", default=None, help="Separate validation text from prepare_data.py")
    parser.add_argument("--steps", type=int, default=200, help="Total target optimizer steps, including resumed steps")
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
    parser.add_argument("--resume", default=None, help="Resume from a version-2 training checkpoint")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Atomically save every N steps (0 = final step only)")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1:
        parser.error("--steps, --batch-size, and --eval-every must be >= 1")
    if args.save_every < 0:
        parser.error("--save-every must be >= 0")
    if not 0 < args.lr:
        parser.error("--lr must be > 0")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    train, val, hashes = load_data(args.data, args.val_data)
    if min(len(train), len(val)) < args.block_size + 2:
        raise ValueError("Train and validation data must each contain at least block_size+2 UTF-8 bytes")

    cfg = ModelConfig(block_size=args.block_size, d_model=args.d_model,
                      n_layers=args.layers, n_heads=args.heads)
    model = Aetherion(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    # Validation sampling must not advance the training sampler, even on a
    # partial run's final step. Otherwise uninterrupted/resumed runs diverge.
    train_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    val_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    trainer = {
        "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": WEIGHT_DECAY, "seed": args.seed, "device": device,
    }

    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        start_step = validate_resume(checkpoint, cfg, hashes, trainer, args.steps)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_rng(checkpoint["rng_state"], train_generator, val_generator, device)
        print(f"resumed_from={args.resume} completed_step={start_step}")

    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")

    def get_batch(split: torch.Tensor, generator: torch.Generator):
        starts = torch.randint(0, len(split) - cfg.block_size,
                               (args.batch_size,), generator=generator)
        x = torch.stack([split[i:i + cfg.block_size] for i in starts])
        y = torch.stack([split[i + 1:i + cfg.block_size + 1] for i in starts])
        return x.to(device), y.to(device)

    output = Path(args.output)
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        x, y = get_batch(train, train_generator)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.inference_mode():
                vx, vy = get_batch(val, val_generator)
                _, val_loss = model(vx, vy)
            print(f"step={step:4d} train_loss={loss.item():.3f} val_loss={val_loss.item():.3f}")

        if step == args.steps or (args.save_every and step % args.save_every == 0):
            save_checkpoint(output, {
                "checkpoint_version": CHECKPOINT_VERSION,
                "config": asdict(cfg),
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_state": capture_rng(train_generator, val_generator, device),
                "step": step,
                "seed": args.seed,
                "data": hashes,
                "trainer": trainer,
            })
            print(f"saved_checkpoint={output} step={step}")


if __name__ == "__main__":
    main()
