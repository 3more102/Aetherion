"""Atomic, metadata-checked resumable checkpoints for Aetherion Micro.

Checkpoint files are for locally created, trusted training artifacts; weight-only
loading reduces unsafe deserialization risk but is not a substitute for trust.
"""
import os
import tempfile
from hashlib import sha256
from pathlib import Path

import torch
from aetherion.devices import xpu_available

FORMAT_VERSION = 2
LINEAGE_VERSION = 1


def file_sha256(path):
    """Hash a checkpoint in bounded memory without loading its tensors."""
    digest = sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_lineage(lineage, *, step):
    """Validate optional per-snapshot metadata; it is not a signed attestation."""
    if type(step) is not int or step < 1 or type(lineage) is not dict:
        raise ValueError("Invalid checkpoint lineage")
    if set(lineage) != {"version", "run_id", "parent_sha256", "parent_step"}:
        raise ValueError("Invalid checkpoint lineage fields")
    if type(lineage["version"]) is not int or lineage["version"] != LINEAGE_VERSION:
        raise ValueError("Unsupported checkpoint lineage version")
    run_id, digest, parent_step = (
        lineage["run_id"], lineage["parent_sha256"], lineage["parent_step"]
    )
    hexdigits = set("0123456789abcdef")
    if type(run_id) is not str or len(run_id) != 32 or not set(run_id) <= hexdigits:
        raise ValueError("Invalid checkpoint lineage run ID")
    if digest is None:
        if parent_step is not None:
            raise ValueError("Root checkpoint cannot have a parent step")
    elif (type(digest) is not str or len(digest) != 64
          or not set(digest) <= hexdigits
          or type(parent_step) is not int or not 1 <= parent_step < step):
        raise ValueError("Invalid checkpoint lineage parent")
    return dict(lineage)


def save_training_checkpoint(path, *, model, optimizer, train_rng, config, step, seed, data, hparams, overwrite=True, lineage=None):
    """Write checkpoint atomically; opt-in snapshots reject existing paths.

    overwrite=False uses a same-filesystem hard link from the fsynced temp
    file, making destination creation atomic and exclusive. It requires
    a filesystem that supports hard links.
    """
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
    if lineage is not None:
        payload["lineage"] = validate_lineage(lineage, step=step)
    if next(model.parameters()).device.type == "xpu":
        payload["xpu_rng_states"] = torch.xpu.get_rng_state_all()
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temp_name, target)
        else:
            # Atomic create-if-absent: concurrent workers cannot overwrite a snapshot.
            os.link(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_training_checkpoint(path, *, model, optimizer, train_rng, config, seed, data, hparams, lineage_out=None):
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
    if type(step) is not int or step < 1:
        raise ValueError("Invalid checkpoint training step")
    lineage = checkpoint.get("lineage")
    if lineage is not None:
        lineage = validate_lineage(lineage, step=step)
    if lineage_out is not None:
        lineage_out.clear()
        if lineage is not None:
            lineage_out.update(lineage)
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
