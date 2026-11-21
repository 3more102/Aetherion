"""Tests for deterministic held-out byte evaluation."""
from dataclasses import asdict
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aetherion import Aetherion, ModelConfig
from evaluate import evaluate_tokens, validation_tokens


class UniformModel(torch.nn.Module):
    """Always predicts a uniform 256-byte distribution."""

    def __init__(self, block_size: int = 4):
        super().__init__()
        self.cfg = SimpleNamespace(block_size=block_size)

    def forward(self, token_ids):
        b, t = token_ids.shape
        return torch.zeros(b, t, 256, device=token_ids.device), None


def test_validation_split_is_last_tenth():
    raw = bytes(range(100))
    val = validation_tokens(raw)
    assert val.tolist() == list(range(90, 100))
    with pytest.raises(ValueError):
        validation_tokens(b"short")


def test_evaluation_counts_each_target_once_across_windows():
    tokens = torch.arange(12)
    loss, count = evaluate_tokens(UniformModel(block_size=4), tokens)
    assert count == len(tokens) - 1
    assert loss == pytest.approx(math.log(256), abs=1e-5)


def test_cli_loads_checkpoint_and_reports_metadata(tmp_path):
    torch.manual_seed(1)
    cfg = ModelConfig(block_size=8, d_model=32, n_heads=4, n_layers=2)
    model = Aetherion(cfg)
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"config": asdict(cfg), "state_dict": model.state_dict(),
                "step": 5, "seed": 1}, checkpoint)
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(bytes(range(100)) * 2)

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "evaluate.py"),
         "--checkpoint", str(checkpoint), "--data", str(corpus), "--device", "cpu"],
        cwd=root, check=True, capture_output=True, text=True,
    )
    report = json.loads(result.stdout)
    assert report["checkpoint_step"] == 5
    assert report["heldout_bytes"] == 20
    assert report["evaluated_next_byte_targets"] == 19
    assert math.isfinite(report["cross_entropy_nats_per_byte"])
    assert report["perplexity_per_byte"] > 0
    assert len(report["checkpoint_sha256"]) == 64
    assert len(report["data_sha256"]) == 64
