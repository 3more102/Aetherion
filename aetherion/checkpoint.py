"""Atomic, metadata-checked resumable checkpoints for Aetherion Micro.

Checkpoint files are for locally created, trusted training artifacts; weight-only
loading reduces unsafe deserialization risk but is not a substitute for trust.
"""
import os
import tempfile
from pathlib import Path

import torch
from aetherion.devices import xpu_available

FORMAT_VERSION = 2


def save_training_checkpoint(path, *, model, optimizer, train_rng, config, step, seed, data, hparams):
    """Replace a complete checkpoint atomically, retaining inference compatibility."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "config": config,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "seed": seed,
        "data": data,
        "train_hparams": hparams,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "train_generator_state": train_rng.get_state(),
    }
    if next(model.parameters()).device.type == "xpu":
        payload["xpu_rng_states"] = torch.xpu.get_rng_state_all()
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_training_checkpoint(path, *, model, optimizer, train_rng, config, seed, data, hparams):
    """Restore full training state; refuse incompatible configs, data and old checkpoints."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("Checkpoint lacks resumable training state (requires format_version=2)")
    for field, expected, label in (
        ("config", config, "model config"),
        ("seed", seed, "seed"),
        ("data", data, "dataset fingerprint"),
        ("train_hparams", hparams, "training hyperparameters"),
    ):
        if checkpoint.get(field) != expected:
            raise ValueError(f"Cannot resume: {label} differs from checkpoint")
    required = ("state_dict", "optimizer_state_dict", "torch_rng_state", "train_generator_state", "step")
    if any(key not in checkpoint for key in required):
        raise ValueError("Checkpoint is missing optimizer/RNG/step information")
    step = checkpoint["step"]
    if not isinstance(step, int) or step < 1:
        raise ValueError("Invalid checkpoint training step")
    stored_cuda = checkpoint.get("cuda_rng_states", [])
    if stored_cuda and torch.cuda.is_available() and len(stored_cuda) != torch.cuda.device_count():
        raise ValueError("Cannot resume: CUDA device count differs from checkpoint")
    stored_xpu = checkpoint.get("xpu_rng_states", [])
    on_xpu = next(model.parameters()).device.type == "xpu"
    if on_xpu and (not xpu_available() or len(stored_xpu) != torch.xpu.device_count()):
        raise ValueError("Cannot resume: XPU RNG state or device count differs from checkpoint")
    if stored_xpu and not on_xpu:
        raise ValueError("Cannot resume an XPU checkpoint on another backend")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if stored_cuda and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(stored_cuda)
    if on_xpu:
        torch.xpu.set_rng_state_all(stored_xpu)
    train_rng.set_state(checkpoint["train_generator_state"])
    return step
