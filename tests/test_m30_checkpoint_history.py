"""M30 CPU regression: atomic retained saves and comparable checkpoint history."""
import json
import math
import subprocess
import sys

import pytest
import torch

from audit_training_history import audit_history
from train import main as train_main


@pytest.fixture
def retained_run(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "train.txt").write_text("Synthetic training tokens for M30. " * 30)
    (folder / "validation.txt").write_text("Independent M30 synthetic heldout tokens. " * 12)
    base = tmp_path / "history.pt"

    def run(*, steps, resume=None, seed=42, output=None):
        argv = [
            "train.py", "--data", str(folder / "train.txt"),
            "--val-data", str(folder / "validation.txt"),
            "--output", str(output or base), "--steps", str(steps),
            "--save-every", "1", "--retain-step-checkpoints",
            "--batch-size", "1", "--block-size", "8",
            "--d-model", "16", "--layers", "1", "--heads", "2",
            "--eval-every", "1", "--seed", str(seed), "--device", "cpu",
        ]
        if resume:
            argv += ["--resume", str(resume)]
        monkeypatch.setattr(sys, "argv", argv)
        train_main()

    run(steps=3)
    snapshots = [tmp_path / f"history-step{n:08d}.pt" for n in (1, 2, 3)]
    return folder, base, snapshots, run


def test_retained_snapshots_are_independent_and_leave_legacy_output_unused(retained_run):
    folder, base, snapshots, _ = retained_run
    assert not base.exists()
    hashes = set()
    for i, path in enumerate(snapshots, 1):
        assert path.is_file()
        artifact = torch.load(path, weights_only=True, map_location="cpu")
        assert artifact["format_version"] == 2
        assert artifact["step"] == i
        assert artifact["data"]["explicit_validation_file"] is True
        hashes.add(path.read_bytes())
    assert len(hashes) == 3


def test_resume_saves_only_future_steps_and_existing_snapshots_never_overwritten(retained_run):
    folder, base, snapshots, run = retained_run
    first_bytes = snapshots[0].read_bytes()
    run(steps=4, resume=snapshots[-1])
    last = base.with_name("history-step00000004.pt")
    assert torch.load(last, weights_only=True, map_location="cpu")["step"] == 4
    assert snapshots[0].read_bytes() == first_bytes
    with pytest.raises(FileExistsError):
        run(steps=2)
    assert snapshots[0].read_bytes() == first_bytes
    assert not list(base.parent.glob(".history-step*.tmp"))


def test_history_remeasures_exact_target_coverage_and_deltas(retained_run):
    folder, _, snapshots, _ = retained_run
    report = audit_history(checkpoints=snapshots, data_dir=folder, demo=True,
                           stride=3, device="cpu")
    assert report["kind"] == "aetherion-validated-checkpoint-history"
    assert report["mode"] == "synthetic_or_unreviewed_byte_demo"
    history = report["history"]
    assert [h["step"] for h in history] == [1, 2, 3]
    assert len({h["checkpoint_sha256"] for h in history}) == 3
    expected_count = len((folder / "validation.txt").read_bytes()) - 1
    assert all(row["evaluated_targets"] == expected_count for row in history)
    assert all(math.isfinite(row["mean_nll"]) for row in history)
    assert history[0]["nll_delta_from_previous"] is None
    assert history[1]["nll_delta_from_previous"] == pytest.approx(
        history[1]["mean_nll"] - history[0]["mean_nll"])
    assert history[-1]["nll_delta_from_first"] == pytest.approx(
        history[-1]["mean_nll"] - history[0]["mean_nll"])
    assert report["summary"]["best_validation_step"] == min(
        history, key=lambda x: x["mean_nll"])["step"]
    assert report["summary"]["final_validation_improved"] is (
        history[-1]["mean_nll"] < history[0]["mean_nll"])


def test_history_refuses_reorders_duplicates_and_changed_metadata(retained_run, tmp_path):
    folder, _, snapshots, _ = retained_run
    with pytest.raises(ValueError, match="strictly increasing"):
        audit_history(checkpoints=[snapshots[2], snapshots[0]], data_dir=folder,
                      demo=True, device="cpu")
    with pytest.raises(ValueError, match="Repeated checkpoint"):
        audit_history(checkpoints=[snapshots[0], snapshots[0]], data_dir=folder,
                      demo=True, device="cpu")
    with pytest.raises(ValueError, match="2 to 32"):
        audit_history(checkpoints=[snapshots[0]], data_dir=folder, demo=True)
    mutated = torch.load(snapshots[2], weights_only=True, map_location="cpu")
    mutated["seed"] = 999
    altered = tmp_path / "different-seed.pt"
    torch.save(mutated, altered)
    with pytest.raises(ValueError, match="Checkpoint history differs"):
        audit_history(checkpoints=[snapshots[0], altered], data_dir=folder,
                      demo=True, device="cpu")
    (folder / "validation.txt").write_text("Tampered test validation")
    with pytest.raises(ValueError, match="fingerprints differ"):
        audit_history(checkpoints=snapshots, data_dir=folder, demo=True)


def test_history_cli_produces_exclusive_evidence_file(retained_run, tmp_path):
    folder, _, snapshots, _ = retained_run
    target = tmp_path / "history.json"
    cmd = [sys.executable, "audit_training_history.py", "--demo",
           "--data-dir", str(folder), "--stride", "3", "--device", "cpu",
           "--output", str(target)]
    for path in snapshots:
        cmd.extend(["--checkpoint", str(path)])
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stored = json.loads(target.read_text())
    assert stored == json.loads(result.stdout)
    assert stored["summary"]["checkpoint_count"] == 3
    repeat = subprocess.run(cmd, capture_output=True, text=True)
    assert repeat.returncode != 0
    assert "File exists" in repeat.stderr
    assert json.loads(target.read_text()) == stored


def test_history_production_requires_review_before_loading_weights(tmp_path):
    with pytest.raises(ValueError, match="Production requires"):
        audit_history(checkpoints=["missing1.pt", "missing2.pt"],
                      data_dir=str(tmp_path), demo=False)
