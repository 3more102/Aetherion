"""Regression tests for interrupted/resumed CPU training and resume guards."""
import sys

import pytest
import torch

from train import main


def run_train(monkeypatch, data, output, *, steps, extra=()):
    argv = [
        "train.py", "--data", str(data), "--steps", str(steps),
        "--batch-size", "2", "--block-size", "16", "--d-model", "32",
        "--layers", "1", "--heads", "4", "--eval-every", "10",
        "--save-every", "1", "--device", "cpu", "--output", str(output),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    return torch.load(output, map_location="cpu", weights_only=True)


def test_resumed_training_matches_uninterrupted_weights(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    data = tmp_path / "data.txt"
    data.write_bytes(b"synthetic bytes for a tiny training smoke test! " * 20)
    full_path, resume_path = tmp_path / "full.pt", tmp_path / "resume.pt"
    full = run_train(monkeypatch, data, full_path, steps=4)
    first = run_train(monkeypatch, data, resume_path, steps=2)
    assert first["step"] == 2
    resumed = run_train(monkeypatch, data, resume_path, steps=4,
                        extra=("--resume", str(resume_path)))
    assert resumed["step"] == full["step"] == 4
    assert resumed["data"] == full["data"]
    assert resumed["format_version"] == 2
    for name, tensor in full["state_dict"].items():
        assert torch.equal(tensor, resumed["state_dict"][name]), name
    assert torch.equal(full["train_generator_state"], resumed["train_generator_state"])
    # Check optimizer counters and momentum tensors, not just final model weights.
    for idx, state in full["optimizer_state_dict"]["state"].items():
        for key, value in state.items():
            other = resumed["optimizer_state_dict"]["state"][idx][key]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other


def test_resume_rejects_changed_dataset(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    data = tmp_path / "data.txt"
    data.write_bytes(b"allowed demo tokens " * 20)
    output = tmp_path / "ckpt.pt"
    run_train(monkeypatch, data, output, steps=1)
    data.write_bytes(b"different demo tokens " * 20)
    with pytest.raises(ValueError, match="dataset fingerprint"):
        run_train(monkeypatch, data, output, steps=2, extra=("--resume", str(output)))


def test_resume_rejects_nonresumable_checkpoint(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    data = tmp_path / "data.txt"
    data.write_bytes(b"allowed demo tokens " * 20)
    output = tmp_path / "old.pt"
    torch.save({"step": 1, "config": {}}, output)
    with pytest.raises(ValueError, match="format_version=2"):
        run_train(monkeypatch, data, output, steps=2, extra=("--resume", str(output)))
