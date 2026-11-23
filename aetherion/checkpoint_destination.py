"""M36: Point-in-time disk and collision preflight for guarded 125M pilots.

This does not reserve disk space or eliminate races with concurrent writers.
"""
from __future__ import annotations

import math
from pathlib import Path
import shutil

GIB = 1024 ** 3


def _planned_step(step: int, *, steps: int, interval: int) -> bool:
    return 1 <= step <= steps and (step == steps or (interval > 0 and step % interval == 0))


def check_checkpoint_destination(output: str, *, steps: int, save_every: int | None,
                                 retain_step_checkpoints: bool, min_free_gib: float,
                                 allow_overwrite_output: bool = False,
                                 disk_usage=shutil.disk_usage) -> dict:
    """Fail closed on existing destinations and inadequate *current* free space.

    Disk usage is measured on the nearest existing parent path, which can
    differ from actual capacity at write time. This check never writes files.
    """
    if type(output) is not str or not output.strip():
        raise ValueError("Checkpoint output must be a nonempty path")
    if type(steps) is not int or steps < 1:
        raise ValueError("Checkpoint steps must be a positive integer")
    if save_every is not None and (type(save_every) is not int or save_every < 0):
        raise ValueError("Checkpoint save-every must be nonnegative")
    if type(retain_step_checkpoints) is not bool or type(allow_overwrite_output) is not bool:
        raise ValueError("Checkpoint retention and overwrite policy must be boolean")
    if type(min_free_gib) not in (float, int) or not math.isfinite(min_free_gib) or min_free_gib <= 0:
        raise ValueError("Checkpoint min-free-checkpoint-gib must be finite and positive")

    target = Path(output)
    if not target.name or target.name in (".", ".."):
        raise ValueError("Checkpoint output must name a file")
    if target.is_symlink() or target.is_dir():
        raise ValueError("Checkpoint output is a symlink or directory")
    if retain_step_checkpoints and allow_overwrite_output:
        raise ValueError("Checkpoint overwrite option is invalid with retained snapshots")
    if not retain_step_checkpoints and target.exists() and not allow_overwrite_output:
        raise ValueError("Checkpoint output already exists; refusing to overwrite")

    interval = 50 if save_every is None else save_every
    checked = 0
    if retain_step_checkpoints and target.parent.exists():
        if not target.parent.is_dir():
            raise ValueError("Checkpoint parent is not a directory")
        # Inspect existing names, not every future step: work budget may be very large.
        for candidate in target.parent.glob(f"{target.stem}-step*{target.suffix}"):
            name = candidate.name
            prefix, suffix = f"{target.stem}-step", target.suffix
            digits = name[len(prefix):len(name) - len(suffix) if suffix else len(name)]
            if len(digits) < 8 or not digits.isascii() or not digits.isdecimal():
                continue
            step = int(digits)
            if name != f"{target.stem}-step{step:08d}{target.suffix}":
                continue
            checked += 1
            if _planned_step(step, steps=steps, interval=interval):
                raise ValueError(f"Checkpoint scheduled snapshot already exists: {name}")

    anchor = target.parent
    while not anchor.exists():
        if anchor.is_symlink():
            raise ValueError("Checkpoint parent is a broken symlink")
        if anchor == anchor.parent:
            raise ValueError("Checkpoint storage filesystem is unavailable")
        anchor = anchor.parent
    if not anchor.is_dir():
        raise ValueError("Checkpoint parent is not a directory")
    try:
        usage = disk_usage(anchor)
        free_bytes = usage.free
    except (OSError, ValueError) as exc:
        raise ValueError("Cannot read checkpoint filesystem free space") from exc
    if type(free_bytes) is not int or free_bytes < 0:
        raise ValueError("Checkpoint filesystem returned invalid free space")
    required_bytes = math.ceil(min_free_gib * GIB)
    if free_bytes < required_bytes:
        raise ValueError("Insufficient free checkpoint disk space")

    return {
        "schema_version": 1,
        "kind": "aetherion-125m-checkpoint-destination-preflight",
        "result": "point_in_time_destination_check_passed",
        "output": str(target),
        "retained_step_snapshots": retain_step_checkpoints,
        "existing_candidate_files_examined": checked,
        "operator_approved_overwrite": allow_overwrite_output,
        "min_requested_free_bytes": required_bytes,
        "observed_free_bytes": free_bytes,
        "limitations": (
            "Point-in-time disk-space and filename check only. No disk reservation, "
            "write guarantee or collision-race protection; actual checkpoint bytes "
            "and filesystem capacity can change during training."
        ),
    }
