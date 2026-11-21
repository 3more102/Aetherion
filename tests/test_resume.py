"""CPU checkpoint-resume correctness: uninterrupted and resumed runs must match."""
from __future__ import annotations

import sys

import pytest
import torch

import train


def _run(monkeypatch, dataset, output, steps, resume=None, batch_size=2):
    args = [
        "train.py", "--data", str(dataset), "--output", str(output),
        "--steps", str(steps), "--batch-size", str(batch_size),
        "--block-size", "8", "--layers", "1", "--heads", "4",
        "--d-model", "32", "--eval-every", "2",
        "--seed", "7", "--device", "cpu",
    ]
    if resume is not None:
        args += ["--resume", str(resume)]
    monkeypatch.setattr(sys, "argv", args)
    train.main()
    return torch.load(output, weights_only=True, map_location="cpu")


def test_interrupted_cpu_training_matches_uninterrupted(tmp_path, monkeypatch):
    corpus = tmp_path / "demo.txt"
    corpus.write_text("Aetherion artificial text example. " * 40, encoding="utf-8")
    full_path = tmp_path / "full.pt"
    partial_path = tmp_path / "partial.pt"
    resumed_path = tmp_path / "resumed.pt"

    full = _run(monkeypatch, corpus, full_path, steps=4)
    partial = _run(monkeypatch, corpus, partial_path, steps=2)
    assert partial["step"] == 2
    resumed = _run(monkeypatch, corpus, resumed_path, steps=4, resume=partial_path)

    assert full["step"] == resumed["step"] == 4
    assert full["config"] == resumed["config"]
    for key, weight in full["state_dict"].items():
        torch.testing.assert_close(weight, resumed["state_dict"][key], atol=0, rtol=0)
    for name in ("torch_cpu", "train_batches"):
        torch.testing.assert_close(full["rng_state"][name], resumed["rng_state"][name],
                                   atol=0, rtol=0)
    assert "optimizer_state_dict" in resumed


def test_resume_rejects_changed_data_or_batch_size(tmp_path, monkeypatch):
    corpus = tmp_path / "demo.txt"
    corpus.write_text("one two three four " * 100, encoding="utf-8")
    checkpoint = tmp_path / "partial.pt"
    _run(monkeypatch, corpus, checkpoint, steps=2)
    with pytest.raises(ValueError, match="batch size"):
        _run(monkeypatch, corpus, tmp_path / "other.pt", steps=4,
             resume=checkpoint, batch_size=3)

    corpus.write_text("changed data " * 100, encoding="utf-8")
    with pytest.raises(ValueError, match="data hash"):
        _run(monkeypatch, corpus, tmp_path / "other.pt", steps=4, resume=checkpoint)


def test_legacy_checkpoint_cannot_resume(tmp_path, monkeypatch):
    corpus = tmp_path / "demo.txt"
    corpus.write_text("train this toy text " * 100, encoding="utf-8")
    checkpoint = tmp_path / "partial.pt"
    result = _run(monkeypatch, corpus, checkpoint, steps=2)
    result.pop("optimizer_state_dict")
    result.pop("rng_state")
    torch.save(result, checkpoint)
    with pytest.raises(ValueError, match="legacy weights"):
        _run(monkeypatch, corpus, tmp_path / "other.pt", steps=4, resume=checkpoint)
