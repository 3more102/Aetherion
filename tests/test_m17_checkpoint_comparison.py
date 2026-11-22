"""M17: checkpoint evidence must be reproducible and fail closed on mismatch."""
import json
import math
import os
import subprocess
import sys

import pytest
import torch

from compare_checkpoints import compare
from train import main as train_main


@pytest.fixture
def tiny_run(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.txt").write_text("Aetherion synthetic training sequence. " * 30)
    (data_dir / "validation.txt").write_text("Independent validation fixture text. " * 10)
    before, after = tmp_path / "before.pt", tmp_path / "after.pt"
    common = [
        "--data", str(data_dir / "train.txt"),
        "--val-data", str(data_dir / "validation.txt"),
        "--device", "cpu", "--batch-size", "1", "--block-size", "8",
        "--d-model", "16", "--heads", "2", "--layers", "1",
        "--schedule-steps", "3", "--eval-every", "1", "--save-every", "1",
    ]

    def train_to(path, steps, resume=None, seed="42"):
        argv = ["train.py", *common, "--steps", str(steps),
                "--output", str(path), "--seed", seed]
        if resume is not None:
            argv += ["--resume", str(resume)]
        monkeypatch.setattr(sys, "argv", argv)
        train_main()

    train_to(before, 1)
    train_to(after, 3, resume=before)
    return data_dir, before, after, train_to


def test_comparison_scores_full_heldout_validation(tiny_run):
    data_dir, before, after, _ = tiny_run
    report = compare(before=before, after=after, data_dir=data_dir, demo=True,
                     stride=3, device="cpu")
    assert report["mode"] == "synthetic_or_unreviewed_byte_demo"
    assert report["before"]["checkpoint_step"] == 1
    assert report["after"]["checkpoint_step"] == 3
    expected = len((data_dir / "validation.txt").read_bytes()) - 1
    assert report["before"]["evaluated_targets"] == expected
    assert report["after"]["evaluated_targets"] == expected
    assert math.isfinite(report["before"]["mean_nll"])
    assert math.isfinite(report["after"]["mean_nll"])
    assert report["perplexity_ratio"] == pytest.approx(
        report["after"]["perplexity"] / report["before"]["perplexity"])
    assert report["validation_nll_improved"] is (
        report["after"]["mean_nll"] < report["before"]["mean_nll"])
    assert len(report["before"]["checkpoint_sha256"]) == 64


def test_cli_reports_provenance_and_refuses_overwrite(tiny_run, tmp_path):
    data_dir, before, after, _ = tiny_run
    output = tmp_path / "report.json"
    cmd = [sys.executable, "compare_checkpoints.py", "--demo",
           "--data-dir", str(data_dir), "--before", str(before),
           "--after", str(after), "--stride", "3", "--output", str(output)]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    first = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
    assert json.loads(first.stdout) == json.loads(output.read_text())
    repeated = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert repeated.returncode != 0
    assert "File exists" in repeated.stderr


def test_rejects_tampering_and_nonprogress(tiny_run):
    data_dir, before, after, train_to = tiny_run
    with pytest.raises(ValueError, match="different files"):
        compare(before=before, after=before, data_dir=data_dir, demo=True)
    with pytest.raises(ValueError, match="step must exceed"):
        compare(before=after, after=before, data_dir=data_dir, demo=True)
    third = data_dir.parent / "different-seed.pt"
    train_to(third, 2, seed="99")
    with pytest.raises(ValueError, match="config, data, seed"):
        compare(before=before, after=third, data_dir=data_dir, demo=True)
    (data_dir / "validation.txt").write_text("tampered validation")
    with pytest.raises(ValueError, match="fingerprints differ"):
        compare(before=before, after=after, data_dir=data_dir, demo=True)


def test_production_requires_attestation_and_verified_bpe(tiny_run):
    data_dir, before, after, _ = tiny_run
    with pytest.raises(ValueError, match="Production requires"):
        compare(before=before, after=after, data_dir=data_dir)
    with pytest.raises(ValueError, match="--demo requires"):
        compare(before=before, after=after, data_dir=data_dir,
                demo=True, allow_licenses=("CC0-1.0",))
