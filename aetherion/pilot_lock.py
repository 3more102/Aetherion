"""M37: Cooperative, crash-conservative exclusive lock for 125M pilot launches.

The directory is created atomically. A crashed launcher leaves it behind, and
an operator must investigate before removing it. Direct train.py and other
uncooperative writers are not protected.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from secrets import token_hex


@contextmanager
def pilot_execution_lock(output: str):
    """Prevent two guarded launchers from writing the same output concurrently.

    Only owns and removes locks created in this invocation. It never guesses
    whether an existing lock is stale and never clears it automatically.
    """
    if type(output) is not str or not output.strip():
        raise ValueError("Pilot lock output must be a nonempty file path")
    target = Path(output)
    if not target.name or target.name in (".", ".."):
        raise ValueError("Pilot lock output must name a file")

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir():
        raise ValueError("Pilot lock parent is not a directory")
    # Resolve the directory to make relative paths and directory aliases share
    # the same lock. Do not resolve the target: M36 rejects output symlinks.
    lock = parent.resolve(strict=True) / f".{target.name}.aetherion-pilot.lock"
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise ValueError(
            f"Pilot output already locked: {lock}. Investigate any interrupted "
            "training job before manually clearing the lock."
        ) from exc

    owner = lock / "owner.json"
    fingerprint = lock.stat(follow_symlinks=False)
    identity = (fingerprint.st_dev, fingerprint.st_ino)
    token = token_hex(16)
    info = {
        "schema_version": 1,
        "kind": "aetherion-125m-cooperative-pilot-execution-lock",
        "token": token,
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(parent.resolve(strict=True) / target.name),
    }
    try:
        with owner.open("x", encoding="utf-8") as handle:
            json.dump(info, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield {
            "schema_version": 1,
            "result": "exclusive_guarded_launcher_lock_acquired",
            "lock_directory": str(lock),
            "limitations": (
                "Cooperative lock for guarded pilot_train.py launchers only. "
                "An uncooperative writer can bypass it; sudden termination "
                "leaves a lock requiring manual investigation. No filesystem "
                "path is reserved against all possible races."
            ),
        }
    finally:
        # Never remove a path that no longer points to our original lock.
        try:
            current = lock.stat(follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None and (current.st_dev, current.st_ino) == identity:
            # Refuse to remove an altered owner record (possible manual
            # intervention, suspicious replacement or external interference).
            try:
                registered = json.loads(owner.read_text(encoding="utf-8"))
            except (FileNotFoundError, ValueError, UnicodeError):
                registered = None
            if isinstance(registered, dict) and registered.get("token") == token:
                owner.unlink()
                lock.rmdir()
