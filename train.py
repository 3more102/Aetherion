"""Train Aetherion Micro from UTF-8 bytes, with resumable checkpoints."""
from __future__ import annotations
import argparse
import os
import tempfile
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import torch
from aetherion import Aetherion, ModelConfig


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



def _training_config(args: argparse.Namespace, device: str) -> dict:
    return {"batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": 0.1, "device_type": torch.device(device).type}


def _rng_snapshot(batch_generator: torch.Generator) -> dict:
    """Save randomness needed to continue training, not just to reload weights."""
    state = {"torch_cpu": torch.get_rng_state(),
             "train_batch": batch_generator.get_state()}
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state: dict, batch_generator: torch.Generator, device: str) -> None:
    torch.set_rng_state(state["torch_cpu"])
    batch_generator.set_state(state["train_batch"])
    if torch.device(device).type == "cuda":
        if "torch_cuda_all" not in state:
            raise ValueError("Resume checkpoint is missing CUDA RNG state")
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    if torch.device(device).type == "mps":
        if "torch_mps" not in state:
            raise ValueError("Resume checkpoint is missing MPS RNG state")
        torch.mps.set_rng_state(state["torch_mps"])


def save_training_checkpoint(path: Path, model: Aetherion, optimizer: torch.optim.Optimizer,
                             config: ModelConfig, step: int, args: argparse.Namespace,
                             hashes: dict, batch_generator: torch.Generator, device: str) -> None:
    """Atomically replace a resumable checkpoint, keeping generate.py compatibility."""
    payload = {
        "checkpoint_version": 2,
        "config": asdict(config),
        "state_dict": model.state_dict(),
        "step": step,
        "seed": args.seed,
        "data": hashes,
        "training_state": {
            "optimizer": optimizer.state_dict(),
            "run_config": _training_config(args, device),
            "rng": _rng_snapshot(batch_generator),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def restore_training_checkpoint(path: Path, model: Aetherion,
                                optimizer: torch.optim.Optimizer,
                                config: ModelConfig, args: argparse.Namespace,
                                hashes: dict, batch_generator: torch.Generator,
                                device: str) -> int:
    """Reject mismatched runs before modifying weights/optimizer/random state."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_version") != 2 or "training_state" not in checkpoint:
        raise ValueError("Checkpoint is inference-only or too old to resume (requires version 2)")
    if checkpoint.get("config") != asdict(config):
        raise ValueError("Model configuration differs from checkpoint; use the original model flags")
    if checkpoint.get("data") != hashes:
        raise ValueError("Training/validation dataset hashes differ from checkpoint")
    if checkpoint.get("seed") != args.seed:
        raise ValueError("Seed differs from checkpoint")
    trainer = checkpoint["training_state"]
    if trainer.get("run_config") != _training_config(args, device):
        raise ValueError("Training batch size, learning rate or device differs from checkpoint")
    step = checkpoint.get("step")
    if not isinstance(step, int) or step < 1 or step >= args.steps:
        raise ValueError("--steps must exceed the checkpoint's completed step count")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(trainer["optimizer"])
    _restore_rng(trainer["rng"], batch_generator, device)
    return step


def main() -> None:
    parser = argparse.ArgumentParser(description="Aetherion Micro training")
    parser.add_argument("--data", default="data/demo.txt")
    parser.add_argument("--val-data", default=None, help="Separate validation text from prepare_data.py")
    parser.add_argument("--steps", type=int, default=200, help="Total optimizer steps, including any resumed steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None, help="Checkpoint path (defaults to --resume path or checkpoints/demo.pt)")
    parser.add_argument("--resume", default=None, help="Resume training from a version-2 training checkpoint")
    parser.add_argument("--save-every", type=int, default=0, help="Save every N optimizer steps (0: final only)")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_every < 1 or args.save_every < 0:
        parser.error("--steps, --batch-size and --eval-every must be >= 1; --save-every >= 0")
    if args.lr <= 0:
        parser.error("--lr must be > 0")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    train, val, hashes = load_data(args.data, args.val_data)
    if min(len(train), len(val)) < args.block_size + 2:
        raise ValueError("Train and validation data must each contain at least block_size+2 UTF-8 bytes")

    cfg = ModelConfig(block_size=args.block_size, d_model=args.d_model,
                      n_layers=args.layers, n_heads=args.heads)
    model = Aetherion(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(args.seed + 1)
    start_step = 0
    if args.resume:
        start_step = restore_training_checkpoint(Path(args.resume), model, optimizer, cfg,
                                                 args, hashes, train_generator, device)
        print(f"resumed_from={args.resume} completed_steps={start_step}")
    output = Path(args.output or args.resume or "checkpoints/demo.pt")
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,}")

    def get_batch(split: torch.Tensor, generator: torch.Generator):
        starts = torch.randint(0, len(split) - cfg.block_size, (args.batch_size,),
                               generator=generator)
        x = torch.stack([split[i:i + cfg.block_size] for i in starts])
        y = torch.stack([split[i + 1:i + cfg.block_size + 1] for i in starts])
        return x.to(device), y.to(device)

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        x, y = get_batch(train, train_generator)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            # Evaluation uses a per-step RNG, so an extra final evaluation does
            # not change training batches or future validation samples on resume.
            val_generator = torch.Generator(device="cpu")
            val_generator.manual_seed(args.seed + 10_000_000 + step)
            model.eval()
            with torch.inference_mode():
                vx, vy = get_batch(val, val_generator)
                _, val_loss = model(vx, vy)
            print(f"step={step:4d} train_loss={loss.item():.3f} val_loss={val_loss.item():.3f}")
        if (args.save_every and step % args.save_every == 0) or step == args.steps:
            save_training_checkpoint(output, model, optimizer, cfg, step, args,
                                     hashes, train_generator, device)
            print(f"saved_checkpoint={output} step={step}")


if __name__ == "__main__":
    main()
