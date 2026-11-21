"""Regression tests for exact Aetherion-Micro training resumption."""
from __future__ import annotations

import sys

import pytest
import torch

from train import main


def _args(train_data, val_data, checkpoint, steps, extra=()):
    return [
        "train.py", "--data", str(train_data), "--val-data", str(val_data),
        "--steps", str(steps), "--batch-size", "2", "--block-size", "8",
        "--d-model", "16", "--layers", "1", "--heads", "2",
        "--eval-every", "4", "--seed", "17", "--device", "cpu",
        "--output", str(checkpoint), *extra,
    ]


def _run(monkeypatch, train_data, val_data, checkpoint, steps, extra=()):
    monkeypatch.setattr(sys, "argv", _args(train_data, val_data, checkpoint, steps, extra))
    main()
    return torch.load(checkpoint, map_location="cpu", weights_only=True)


def _corpora(tmp_path):
    train_data = tmp_path / "train.txt"
    val_data = tmp_path / "validation.txt"
    train_data.write_bytes(b"Licensed test data, not useful for language training. " * 4)
    val_data.write_bytes(b"A separate validation fixture with enough UTF-8 bytes. " * 4)
    return train_data, val_data


def test_resume_exactly_matches_uninterrupted_cpu_training(monkeypatch, tmp_path):
    train_data, val_data = _corpora(tmp_path)
    continuous_path = tmp_path / "continuous.pt"
    resumed_path = tmp_path / "resumed.pt"

    continuous = _run(monkeypatch, train_data, val_data, continuous_path, 6)
    partial = _run(monkeypatch, train_data, val_data, resumed_path, 3)
    assert partial["step"] == 3
    resumed = _run(monkeypatch, train_data, val_data, resumed_path, 6,
                   ("--resume", str(resumed_path), "--save-every", "2"))

    assert resumed["checkpoint_version"] == 2
    assert continuous["step"] == resumed["step"] == 6
    assert resumed["data"]["explicit_validation_file"]
    assert not list(tmp_path.glob("*.tmp"))
    for key in continuous["state_dict"]:
        torch.testing.assert_close(continuous["state_dict"][key],
                                   resumed["state_dict"][key], atol=0, rtol=0)
    for key in ("cpu", "train_sampler"):
        torch.testing.assert_close(continuous["rng_state"][key],
                                   resumed["rng_state"][key], atol=0, rtol=0)
    assert continuous["optimizer_state_dict"]["param_groups"] == resumed["optimizer_state_dict"]["param_groups"]
    for idx in continuous["optimizer_state_dict"]["state"]:
        state_a = continuous["optimizer_state_dict"]["state"][idx]
        state_b = resumed["optimizer_state_dict"]["state"][idx]
        for key in state_a:
            torch.testing.assert_close(state_a[key], state_b[key], atol=0, rtol=0)


def test_resume_rejects_changed_training_data(monkeypatch, tmp_path):
    train_data, val_data = _corpora(tmp_path)
    checkpoint = tmp_path / "train.pt"
    _run(monkeypatch, train_data, val_data, checkpoint, 2)
    train_data.write_bytes(train_data.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="data hashes"):
        _run(monkeypatch, train_data, val_data, checkpoint, 4,
             ("--resume", str(checkpoint)))


def test_resume_rejects_changed_optimizer_hyperparameters(monkeypatch, tmp_path):
    train_data, val_data = _corpora(tmp_path)
    checkpoint = tmp_path / "train.pt"
    _run(monkeypatch, train_data, val_data, checkpoint, 2)
    with pytest.raises(ValueError, match="batch size, learning rate"):
        _run(monkeypatch, train_data, val_data, checkpoint, 4,
             ("--resume", str(checkpoint), "--lr", "0.001"))


def test_old_inference_only_checkpoint_cannot_resume(monkeypatch, tmp_path):
    train_data, val_data = _corpora(tmp_path)
    checkpoint = tmp_path / "train.pt"
    payload = _run(monkeypatch, train_data, val_data, checkpoint, 1)
    payload.pop("checkpoint_version")
    payload.pop("optimizer_state_dict")
    payload.pop("rng_state")
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="Old checkpoints still work for generation"):
        _run(monkeypatch, train_data, val_data, checkpoint, 3,
             ("--resume", str(checkpoint)))
