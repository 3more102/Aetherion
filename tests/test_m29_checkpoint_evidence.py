"""M29: proof-oriented single-checkpoint validation tests using CPU fixtures."""
import json
import math
import subprocess
import sys

import pytest
import torch

import assess_checkpoint
from train import main as train_main


@pytest.fixture
def tiny_checkpoint(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "train.txt").write_text("Aetherion tiny synthetic train tokens. " * 24)
    (folder / "validation.txt").write_text("Separate synthetic validation fixture. " * 12)
    ckpt = tmp_path / "tiny.pt"
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--data", str(folder / "train.txt"),
        "--val-data", str(folder / "validation.txt"),
        "--output", str(ckpt), "--steps", "2", "--batch-size", "1",
        "--block-size", "8", "--d-model", "16", "--layers", "1",
        "--heads", "2", "--device", "cpu", "--eval-every", "1",
    ])
    train_main()
    return folder, ckpt


def demo(folder, ckpt, **kw):
    return assess_checkpoint.assess_checkpoint(
        checkpoint=ckpt, data_dir=folder, demo=True, device="cpu", **kw)


def test_demo_evidence_exact_coverage_and_checkpoint_hash(tiny_checkpoint):
    folder, ckpt = tiny_checkpoint
    result = demo(folder, ckpt, stride=3)
    assert result["kind"] == "aetherion-single-checkpoint-heldout-validation"
    assert result["mode"] == "synthetic_or_unreviewed_byte_demo"
    assert result["checkpoint"]["step"] == 2
    assert result["checkpoint"]["parameter_count"] > 0
    assert result["checkpoint"]["sha256"] == assess_checkpoint.file_sha256(ckpt)
    assert result["validation"]["evaluated_targets"] == (
        len((folder / "validation.txt").read_bytes()) - 1)
    assert result["validation"]["token_unit"] == "UTF8_byte_tokens"
    assert result["validation"]["target_coverage"] == "all_targets_exactly_once"
    assert math.isfinite(result["validation"]["mean_nll"])
    assert result["validation"]["perplexity"] == pytest.approx(
        math.exp(result["validation"]["mean_nll"]))
    assert result["dataset"]["lexical_screen_findings_count"] is None
    assert result == demo(folder, ckpt, stride=3)


def test_cli_exclusive_json_output_and_independent_evaluate_parity(tiny_checkpoint, tmp_path):
    folder, ckpt = tiny_checkpoint
    output = tmp_path / "evidence.json"
    cmd = [sys.executable, "assess_checkpoint.py", "--demo",
           "--checkpoint", str(ckpt), "--data-dir", str(folder),
           "--stride", "3", "--device", "cpu", "--output", str(output)]
    once = subprocess.run(cmd, capture_output=True, text=True, check=True)
    saved = json.loads(output.read_text())
    assert saved == json.loads(once.stdout)
    other = subprocess.run([
        sys.executable, "evaluate.py", "--checkpoint", str(ckpt),
        "--data", str(folder / "train.txt"), "--val-data", str(folder / "validation.txt"),
        "--stride", "3", "--device", "cpu",
    ], capture_output=True, text=True, check=True)
    assert saved["validation"]["mean_nll"] == pytest.approx(
        json.loads(other.stdout)["mean_nll"])
    repeated = subprocess.run(cmd, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "File exists" in repeated.stderr
    assert json.loads(output.read_text()) == saved


def test_reject_tampered_inputs_and_demo_approval_mix(tiny_checkpoint):
    folder, ckpt = tiny_checkpoint
    with pytest.raises(ValueError, match="requires byte tokens"):
        demo(folder, ckpt, tokenizer="dummy.json")
    with pytest.raises(ValueError, match="stride"):
        demo(folder, ckpt, stride=9)
    (folder / "validation.txt").write_text("Changed heldout content.")
    with pytest.raises(ValueError, match="fingerprints differ"):
        demo(folder, ckpt)


def test_reject_invalid_checkpoint_metadata_before_model_build(tiny_checkpoint, tmp_path):
    folder, ckpt = tiny_checkpoint
    bad = tmp_path / "bad.pt"
    original = torch.load(ckpt, weights_only=True, map_location="cpu")
    for name, value in [
        ("missing", {k: v for k, v in original.items() if k != "train_hparams"}),
        ("invalid-step", {**original, "step": True}),
        ("v1", {**original, "format_version": 1}),
    ]:
        torch.save(value, bad)
        with pytest.raises(ValueError):
            demo(folder, bad)


def test_production_requires_approval_before_checkpoint_access(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    missing_weights = tmp_path / "does-not-exist.pt"
    with pytest.raises(ValueError, match="Production requires"):
        assess_checkpoint.assess_checkpoint(
            checkpoint=missing_weights, data_dir=folder, demo=False,
            tokenizer="demo.json", allow_licenses=("CC0-1.0",))
    calls = []
    monkeypatch.setattr(assess_checkpoint, "audit_corpus",
                        lambda *args: calls.append("audit") or {"manifest_sha256": "dummy"})
    monkeypatch.setattr(assess_checkpoint, "require_review",
                        lambda *args: calls.append("review"))
    def block_scan(*args):
        calls.append("screen")
        raise ValueError("Cross-split lexical contamination found")
    monkeypatch.setattr(assess_checkpoint, "require_no_cross_split_findings", block_scan)
    monkeypatch.setattr(assess_checkpoint.torch, "load",
                        lambda *args, **kwargs: pytest.fail("Checkpoint loaded before screening"))
    with pytest.raises(ValueError, match="Cross-split lexical contamination"):
        assess_checkpoint.assess_checkpoint(
            checkpoint=missing_weights, data_dir=folder, demo=False,
            tokenizer="tokenizer.json", review="review.json",
            allow_licenses=("CC0-1.0",))
    assert calls == ["audit", "review", "screen"]
