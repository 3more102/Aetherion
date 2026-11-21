"""Regression tests for exact CPU resume and data/config mismatch guards."""
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def train(tmp_path, output, steps, *extra):
    cmd = [sys.executable, str(ROOT / "train.py"),
           "--data", str(tmp_path / "train.txt"),
           "--val-data", str(tmp_path / "val.txt"),
           "--output", str(output), "--steps", str(steps),
           "--d-model", "32", "--layers", "1", "--heads", "4",
           "--block-size", "16", "--batch-size", "2", "--eval-every", "2",
           "--device", "cpu", *map(str, extra)]
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)


@pytest.fixture
def data(tmp_path):
    (tmp_path / "train.txt").write_text("Training data. " * 20)
    (tmp_path / "val.txt").write_text("Validation data. " * 20)
    return tmp_path


def test_resumed_training_matches_uninterrupted_cpu(data):
    full, split = data / "full.pt", data / "split.pt"
    result = train(data, full, 4)
    assert result.returncode == 0, result.stderr
    result = train(data, split, 3, "--save-every", 2)
    assert result.returncode == 0, result.stderr
    assert "step=2" in result.stdout
    result = train(data, split, 4, "--resume", split)
    assert result.returncode == 0, result.stderr
    assert "completed_steps=3" in result.stdout
    a = torch.load(full, weights_only=True)
    b = torch.load(split, weights_only=True)
    assert a["step"] == b["step"] == 4
    assert a["data"] == b["data"]
    assert a["training_state"]["optimizer"]["param_groups"] == b["training_state"]["optimizer"]["param_groups"]
    for k in a["state_dict"]:
        torch.testing.assert_close(a["state_dict"][k], b["state_dict"][k], rtol=0, atol=0)
    assert torch.equal(a["training_state"]["rng"]["train_batch"],
                       b["training_state"]["rng"]["train_batch"])


def test_resume_rejects_modified_data(data):
    path = data / "model.pt"
    assert train(data, path, 1).returncode == 0
    (data / "val.txt").write_text("Changed! " * 30)
    result = train(data, path, 2, "--resume", path)
    assert result.returncode != 0
    assert "dataset hashes differ" in result.stderr


def test_resume_rejects_incompatible_config_and_old_checkpoints(data):
    path = data / "model.pt"
    assert train(data, path, 1).returncode == 0
    result = train(data, path, 2, "--resume", path, "--lr", 0.001)
    assert result.returncode != 0
    assert "learning rate or device differs" in result.stderr
    old_path = data / "old.pt"
    old_payload = torch.load(path, weights_only=True)
    del old_payload["training_state"]
    torch.save(old_payload, old_path)
    result = train(data, path, 2, "--resume", old_path)
    assert result.returncode != 0
    assert "inference-only or too old" in result.stderr


def test_checkpoint_is_generation_compatible(data):
    path = data / "model.pt"
    assert train(data, path, 2).returncode == 0
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["checkpoint_version"] == 2
    assert checkpoint["config"]["vocab_size"] == 256
    assert "state_dict" in checkpoint and "training_state" in checkpoint
