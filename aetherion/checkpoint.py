"""Strict, resumable Aetherion-Micro training checkpoints.

Use only trusted checkpoints: weight-only loading prevents arbitrary pickle
objects, but untrusted model artifacts should still not be executed.
"""
from __future__ import annotations

from pathlib import Path
import os
import tempfile

import torch


def capture_rng_state(device: str) -> dict:
    """Capture sampling/dropout RNG for supported devices."""
    state = {"cpu": torch.get_rng_state()}
    kind = torch.device(device).type
    if kind == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif kind == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state: dict, device: str) -> None:
    kind = torch.device(device).type
    torch.set_rng_state(state["cpu"])
    if kind == "cuda":
        if "cuda" not in state:
            raise ValueError("Checkpoint is missing CUDA RNG state")
        torch.cuda.set_rng_state_all(state["cuda"])
    elif kind == "mps":
        if "mps" not in state:
            raise ValueError("Checkpoint is missing MPS RNG state")
        torch.mps.set_rng_state(state["mps"])


def save_checkpoint(path: str | Path, payload: dict) -> None:
    """Atomically replace the destination to avoid half-written checkpoints."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.",
                                         suffix=".tmp", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            torch.save(payload, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
        tmp_path.replace(target)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def load_resume(path: str | Path, *, config: dict, data_hashes: dict,
                settings: dict, device: str) -> dict:
    """Validate exact-match inputs and full training state before resuming."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "optimizer_state", "rng_state", "step", "config", "data", "training"):
        if key not in ckpt:
            raise ValueError(f"Checkpoint lacks '{key}'; legacy inference-only checkpoints cannot resume")
    if ckpt["config"] != config:
        raise ValueError("Model configuration differs from the checkpoint")
    if ckpt["data"] != data_hashes:
        raise ValueError("Train or validation data identity differs from the checkpoint")
    if ckpt["training"] != settings:
        raise ValueError("Training settings differ from the checkpoint")
    if ckpt.get("device_type") != torch.device(device).type:
        raise ValueError("Resume requires the same device type as the checkpoint")
    if not isinstance(ckpt["step"], int) or ckpt["step"] < 1:
        raise ValueError("Invalid completed step in checkpoint")
    return ckpt
