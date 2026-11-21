"""Recovery and reproducibility tests for Aetherion Micro training."""
import os
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]


def run_training(args, *, ok=True):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, str(ROOT / "train.py"), *args],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    if ok:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
    return result


def test_resume_matches_uninterrupted_training(tmp_path):
    train = tmp_path / "train.txt"
    val = tmp_path / "validation.txt"
    train.write_bytes(b"hello reliable micro training " * 80)
    val.write_bytes(b"separate held out validation data " * 40)
    base = ["--data", str(train), "--val-data", str(val),
            "--batch-size", "2", "--block-size", "16", "--d-model", "32",
            "--layers", "2", "--heads", "4", "--seed", "7",
            "--eval-every", "2", "--device", "cpu"]
    full_path, resumed_path = tmp_path / "full.pt", tmp_path / "resumed.pt"
    run_training([*base, "--steps", "6", "--output", str(full_path)])
    run_training([*base, "--steps", "3", "--save-every", "2", "--output", str(resumed_path)])
    result = run_training([*base, "--steps", "6", "--resume", str(resumed_path),
                           "--output", str(resumed_path)])
    assert "completed_step=3" in result.stdout
    full = torch.load(full_path, map_location="cpu", weights_only=True)
    resumed = torch.load(resumed_path, map_location="cpu", weights_only=True)
    assert full["step"] == resumed["step"] == 6
    assert full["checkpoint_version"] == resumed["checkpoint_version"] == 2
    for name, expected in full["state_dict"].items():
        torch.testing.assert_close(expected, resumed["state_dict"][name], rtol=0, atol=0)
    for name in ("sampler_rng_state", "cpu_rng_state"):
        torch.testing.assert_close(full[name], resumed[name], rtol=0, atol=0)
    assert full["optimizer_state_dict"]["param_groups"] == resumed["optimizer_state_dict"]["param_groups"]
    for idx, params in full["optimizer_state_dict"]["state"].items():
        for key, value in params.items():
            actual = resumed["optimizer_state_dict"]["state"][idx][key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, actual, rtol=0, atol=0)
            else:
                assert value == actual


def test_resume_rejects_changed_dataset_and_parameters(tmp_path):
    train, val = tmp_path / "train.txt", tmp_path / "validation.txt"
    train.write_bytes(b"fresh train" * 40)
    val.write_bytes(b"distinct val" * 40)
    output = tmp_path / "model.pt"
    base = ["--data", str(train), "--val-data", str(val),
            "--batch-size", "2", "--block-size", "8", "--d-model", "16",
            "--layers", "1", "--heads", "2", "--device", "cpu"]
    run_training([*base, "--steps", "2", "--output", str(output)])
    changed = run_training([*base, "--steps", "4", "--batch-size", "3",
                            "--resume", str(output), "--output", str(output)], ok=False)
    assert "Training parameters do not match" in changed.stderr
    val.write_bytes(val.read_bytes() + b" changed")
    changed = run_training([*base, "--steps", "4", "--resume", str(output),
                            "--output", str(output)], ok=False)
    assert "Dataset hashes" in changed.stderr
    assert torch.load(output, weights_only=True)["step"] == 2
