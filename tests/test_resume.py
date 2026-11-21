"""CPU regression coverage for deterministic Aetherion-Micro training resume."""
import os
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path, train_file, val_file, output, steps, *, resume=None, extra=()):
    command = [
        sys.executable, str(ROOT / "train.py"),
        "--data", str(train_file),
        "--val-data", str(val_file),
        "--device", "cpu",
        "--steps", str(steps),
        "--batch-size", "2",
        "--block-size", "8",
        "--d-model", "16",
        "--layers", "1",
        "--heads", "2",
        "--eval-every", "2",
        "--save-every", "2",
        "--output", str(output),
        *extra,
    ]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    return subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)


def _data(tmp_path):
    train_file, val_file = tmp_path / "train.txt", tmp_path / "val.txt"
    train_file.write_bytes(b"Licensed synthetic training sample, text A. " * 35)
    val_file.write_bytes(b"Independent synthetic validation sample B. " * 35)
    return train_file, val_file


def _assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_equal(x, y)
    else:
        assert a == b


def test_resume_matches_uninterrupted_training_exactly_on_cpu(tmp_path):
    train_file, val_file = _data(tmp_path)
    uninterrupted = tmp_path / "uninterrupted.pt"
    resumed = tmp_path / "resumed.pt"
    first = _run(tmp_path, train_file, val_file, uninterrupted, 6)
    assert first.returncode == 0, first.stderr
    partial = _run(tmp_path, train_file, val_file, resumed, 3)
    assert partial.returncode == 0, partial.stderr
    continuation = _run(tmp_path, train_file, val_file, resumed, 6, resume=resumed)
    assert continuation.returncode == 0, continuation.stderr

    a = torch.load(uninterrupted, weights_only=True, map_location="cpu")
    b = torch.load(resumed, weights_only=True, map_location="cpu")
    assert a["step"] == b["step"] == 6
    assert b["checkpoint_version"] == 1
    _assert_equal(a["state_dict"], b["state_dict"])
    _assert_equal(a["optimizer_state_dict"], b["optimizer_state_dict"])
    _assert_equal(a["torch_rng_state"], b["torch_rng_state"])
    assert a["data"] == b["data"]
    assert a["training"] == b["training"]


def test_resume_refuses_changed_data_or_training_settings(tmp_path):
    train_file, val_file = _data(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    result = _run(tmp_path, train_file, val_file, checkpoint, 2)
    assert result.returncode == 0, result.stderr

    train_file.write_bytes(train_file.read_bytes() + b" changed")
    bad_data = _run(tmp_path, train_file, val_file, checkpoint, 4, resume=checkpoint)
    assert bad_data.returncode != 0
    assert "data differs" in bad_data.stderr

    train_file.write_bytes(train_file.read_bytes()[:-8])
    bad_settings = _run(
        tmp_path, train_file, val_file, checkpoint, 4, resume=checkpoint,
        extra=("--batch-size", "3"),
    )
    assert bad_settings.returncode != 0
    assert "training differs" in bad_settings.stderr


def test_resume_refuses_inference_only_checkpoint(tmp_path):
    train_file, val_file = _data(tmp_path)
    legacy = tmp_path / "legacy.pt"
    torch.save({"step": 2, "state_dict": {}}, legacy)
    result = _run(tmp_path, train_file, val_file, tmp_path / "out.pt", 4, resume=legacy)
    assert result.returncode != 0
    assert "lacks resumable state" in result.stderr
