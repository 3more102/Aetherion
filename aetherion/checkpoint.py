"""Versioned, integrity-checked training checkpoints for Aetherion-Micro.

Checkpoints contain optimizer and RNG state, not just inference weights. Only
load checkpoints from trusted sources: even with weights_only=True, external
files should be treated as untrusted input.
"""
from dataclasses import asdict
import os
from pathlib import Path
import tempfile

import torch


CHECKPOINT_VERSION = 1


def _device_type(device: str) -> str:
    return torch.device(device).type


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config,
    step: int,
    seed: int,
    data: dict,
    training: dict,
    device: str,
) -> None:
    """Atomically persist everything necessary to continue a training run."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "config": asdict(config),
        "state_dict": model.state_dict(),  # also readable by generate.py
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "seed": seed,
        "data": data,
        "training": training,
        "device_type": _device_type(device),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps_rng_state": (
            torch.mps.get_rng_state()
            if _device_type(device) == "mps" and hasattr(torch, "mps")
            and hasattr(torch.mps, "get_rng_state") else None
        ),
        "torch_version": str(torch.__version__),
    }

    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as destination:
            torch.save(checkpoint, destination)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config,
    seed: int,
    data: dict,
    training: dict,
    device: str,
) -> int:
    """Restore a compatible run, or fail before changing model/optimizer state."""
    # Loading to CPU also keeps CPU/CUDA RNG byte tensors in the right format.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Checkpoint lacks resumable state or uses an unsupported version")
    expected = {
        "config": asdict(config),
        "seed": seed,
        "data": data,
        "training": training,
        "device_type": _device_type(device),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Cannot resume: {key} differs from the saved training run")
    required = ("state_dict", "optimizer_state_dict", "step", "torch_rng_state")
    if any(key not in checkpoint for key in required):
        raise ValueError("Checkpoint is missing optimizer, model, step or RNG state")
    if not isinstance(checkpoint["step"], int) or checkpoint["step"] < 1:
        raise ValueError("Checkpoint step must be a positive integer")

    if _device_type(device) == "cuda":
        states = checkpoint.get("cuda_rng_states")
        if states is None or len(states) != torch.cuda.device_count():
            raise ValueError("CUDA RNG state/device count does not match the saved run")
    if _device_type(device) == "mps" and checkpoint.get("mps_rng_state") is None:
        raise ValueError("MPS RNG state is unavailable; cannot guarantee an exact resume")

    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if _device_type(device) == "cuda":
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
    elif _device_type(device) == "mps":
        torch.mps.set_rng_state(checkpoint["mps_rng_state"])
    return checkpoint["step"]
