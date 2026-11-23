"""M31: SHA-linked retained snapshots, resume continuity, and fail-closed audit."""
from hashlib import sha256
import json
import subprocess
import sys

import pytest
import torch

from aetherion.checkpoint import validate_lineage
from audit_training_history import audit_history
from train import main as train_main


@pytest.fixture
def snapshot_run(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "train.txt").write_text("M31 synthetic train bytes. " * 20)
    (folder / "validation.txt").write_text("M31 distinct validation bytes. " * 15)

    def run(*, steps=2, resume=None, output=None, retain=True):
        base = output or tmp_path / "lineage.pt"
        args = [
            "train.py", "--data", str(folder / "train.txt"),
            "--val-data", str(folder / "validation.txt"),
            "--output", str(base), "--steps", str(steps),
            "--save-every", "1", "--batch-size", "1",
            "--block-size", "8", "--d-model", "16",
            "--layers", "1", "--heads", "2",
            "--eval-every", "1", "--device", "cpu",
        ]
        if retain:
            args.append("--retain-step-checkpoints")
        if resume is not None:
            args.extend(["--resume", str(resume)])
        monkeypatch.setattr(sys, "argv", args)
        train_main()
        return [base.with_name(f"{base.stem}-step{step:08d}{base.suffix}")
                for step in range(1, steps + 1)] if retain else [base]

    snapshots = run()
    return folder, snapshots, run


def test_saved_chain_is_rooted_and_auditable(snapshot_run):
    folder, snapshots, run = snapshot_run
    first, second = snapshots
    a = torch.load(first, weights_only=True, map_location="cpu")
    b = torch.load(second, weights_only=True, map_location="cpu")
    assert a["format_version"] == b["format_version"] == 2
    assert a["lineage"]["parent_sha256"] is None
    assert a["lineage"]["parent_step"] is None
    assert len(a["lineage"]["run_id"]) == 32
    assert b["lineage"]["run_id"] == a["lineage"]["run_id"]
    assert b["lineage"]["parent_sha256"] == sha256(first.read_bytes()).hexdigest()
    assert b["lineage"]["parent_step"] == 1
    result = audit_history(checkpoints=snapshots, data_dir=folder,
                           demo=True, stride=4, require_lineage=True)
    assert result["lineage_check"] == "validated"
    assert result["lineage_run_id"] == a["lineage"]["run_id"]
    assert [row["step"] for row in result["history"]] == [1, 2]

    next_snapshots = run(steps=3, resume=second)
    third = next_snapshots[-1]
    c = torch.load(third, weights_only=True, map_location="cpu")
    assert c["lineage"]["run_id"] == a["lineage"]["run_id"]
    assert c["lineage"]["parent_sha256"] == sha256(second.read_bytes()).hexdigest()
    assert c["lineage"]["parent_step"] == 2
    assert audit_history(checkpoints=[first, second, third], data_dir=folder,
                         demo=True, stride=4, require_lineage=True)["summary"]["checkpoint_count"] == 3


def test_modified_parent_hash_and_run_id_fail_closed(snapshot_run, tmp_path):
    folder, (first, second), _ = snapshot_run
    for key, value in [("parent_sha256", "0" * 64), ("run_id", "f" * 32)]:
        changed = torch.load(second, weights_only=True, map_location="cpu")
        changed["lineage"][key] = value
        altered = tmp_path / f"altered-{key}.pt"
        torch.save(changed, altered)
        with pytest.raises(ValueError, match="broken parent"):
            audit_history(checkpoints=[first, altered], data_dir=folder,
                          demo=True, stride=4, require_lineage=True)


def test_missing_or_truncated_lineage_is_rejected_only_when_requested(snapshot_run, tmp_path):
    folder, (first, second), run = snapshot_run
    assert audit_history(checkpoints=[first, second], data_dir=folder,
                         demo=True, stride=4)["lineage_check"] == "not_requested"
    bare = torch.load(first, weights_only=True, map_location="cpu")
    bare.pop("lineage")
    legacy = tmp_path / "legacy.pt"
    torch.save(bare, legacy)
    assert audit_history(checkpoints=[legacy, second], data_dir=folder,
                         demo=True, stride=4)["lineage_check"] == "not_requested"
    with pytest.raises(ValueError, match="lineage metadata"):
        audit_history(checkpoints=[legacy, second], data_dir=folder,
                      demo=True, stride=4, require_lineage=True)
    with pytest.raises(ValueError, match="lineage root"):
        audit_history(checkpoints=[second, second.with_name("missing.pt")],
                      data_dir=folder, demo=True, require_lineage=True)

    # Legacy checkpoints remain resumable, but a new chain must start at the
    # first saved step after legacy resume; there is no claimed earlier ancestry.
    new_base = tmp_path / "legacy-continuation.pt"
    run(steps=3, resume=legacy, output=new_base)
    new_root = tmp_path / "legacy-continuation-step00000002.pt"
    new_tip = tmp_path / "legacy-continuation-step00000003.pt"
    assert torch.load(new_root, weights_only=True)["lineage"]["parent_sha256"] is None
    assert audit_history(checkpoints=[new_root, new_tip], data_dir=folder,
                         demo=True, stride=4, require_lineage=True)["lineage_check"] == "validated"


def test_validation_rejects_malformed_lineage(snapshot_run, tmp_path):
    folder, (first, second), _ = snapshot_run
    for mutation in (
        {"version": True},
        {"run_id": "not-a-run-id"},
        {"parent_sha256": "not-a-sha"},
        {"parent_step": 9999},
    ):
        artifact = torch.load(second, weights_only=True, map_location="cpu")
        artifact["lineage"].update(mutation)
        forged = tmp_path / ("invalid-" + next(iter(mutation)) + ".pt")
        torch.save(artifact, forged)
        with pytest.raises(ValueError, match="lineage"):
            audit_history(checkpoints=[first, forged], data_dir=folder,
                          demo=True, stride=4, require_lineage=True)


def test_lineage_cli_accepts_rooted_chain_and_rejects_missing_snapshots(snapshot_run, tmp_path):
    folder, snapshots, run = snapshot_run
    cmd = [sys.executable, "audit_training_history.py", "--demo",
           "--data-dir", str(folder), "--stride", "4",
           "--require-lineage"]
    for path in snapshots:
        cmd += ["--checkpoint", str(path)]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["lineage_check"] == "validated"

    # A root-to-tip audit cannot skip the intermediate saved checkpoint.
    third = run(steps=3, resume=snapshots[1])[-1]
    failed = subprocess.run(cmd[:-2] + ["--checkpoint", str(third)],
                            capture_output=True, text=True)
    assert failed.returncode != 0
    assert "broken parent" in failed.stderr


def test_lineage_shape_validation_rejects_unknown_fields():
    lineage = {"version": 1, "run_id": "a" * 32,
               "parent_sha256": None, "parent_step": None}
    assert validate_lineage(lineage, step=1) == lineage
    with pytest.raises(ValueError, match="lineage fields"):
        validate_lineage({**lineage, "claimed_verified": True}, step=1)
