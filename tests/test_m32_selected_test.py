"""M32: selection/test separation, integrity gates and exact target coverage."""
from hashlib import sha256
import json
import subprocess
import sys

import pytest
import torch

from aetherion import Aetherion, ModelConfig
from aetherion.evaluation import evaluate_tokens
from assess_selected_test import evaluate_selected_test
from audit_training_history import audit_history
from train import main as train_main


@pytest.fixture
def m32_demo(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    folder = tmp_path / "data"
    folder.mkdir()
    train = ("M32 independent synthetic train sequence. " * 20).encode()
    val = ("M32 distinct validation only sequence! " * 12).encode()
    test = ("M32 sealed synthetic TEST-only sequence? " * 13).encode()
    (folder / "train.txt").write_bytes(train)
    (folder / "validation.txt").write_bytes(val)
    (folder / "test.txt").write_bytes(test)
    base = tmp_path / "tiny.pt"
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--data", str(folder / "train.txt"),
        "--val-data", str(folder / "validation.txt"),
        "--output", str(base), "--steps", "3", "--save-every", "1",
        "--retain-step-checkpoints",
        "--batch-size", "1", "--block-size", "8", "--d-model", "16",
        "--layers", "1", "--heads", "2", "--eval-every", "1", "--device", "cpu",
    ])
    train_main()
    paths = [tmp_path / f"tiny-step{n:08d}.pt" for n in (1, 2, 3)]
    return folder, paths, sha256(test).hexdigest()


def selected(folder, paths, digest, **kwargs):
    return evaluate_selected_test(
        checkpoints=paths, data_dir=str(folder), demo=True,
        expected_test_sha256=digest, require_lineage=True,
        stride=4, device="cpu", **kwargs,
    )


def test_select_by_validation_not_test_and_evaluate_exact_targets(m32_demo):
    folder, paths, digest = m32_demo
    report = selected(folder, paths, digest)
    audit = audit_history(checkpoints=paths, data_dir=folder, demo=True,
                          stride=4, require_lineage=True)
    winner = min(audit["history"], key=lambda e: e["mean_nll"])
    assert report["selection"]["candidate_count"] == 3
    assert report["selection"]["selected_step"] == winner["step"]
    assert report["selection"]["selected_checkpoint_sha256"] == winner["checkpoint_sha256"]
    assert report["selection"]["selected_validation_mean_nll"] == winner["mean_nll"]
    assert report["selection"]["lineage_check"] == "validated"
    assert report["test"]["data_sha256"] == digest
    assert report["test"]["evaluated_targets"] == len((folder / "test.txt").read_bytes()) - 1
    assert report["test"]["target_coverage"] == "all_targets_exactly_once"
    assert report["test"]["token_unit"] == "UTF8_byte_tokens"

    chosen_path = paths[winner["step"] - 1]
    artifact = torch.load(chosen_path, weights_only=True, map_location="cpu")
    model = Aetherion(ModelConfig(**artifact["config"]))
    model.load_state_dict(artifact["state_dict"])
    independent = evaluate_tokens(
        model, torch.tensor(list((folder / "test.txt").read_bytes()), dtype=torch.long),
        stride=4,
    )
    assert report["test"]["total_nll"] == pytest.approx(independent["total_nll"])
    assert report["test"]["mean_nll"] == pytest.approx(independent["mean_nll"])


def test_reject_unpinned_and_tampered_demo_test(m32_demo):
    folder, paths, digest = m32_demo
    with pytest.raises(ValueError, match="requires --expected-test-sha256"):
        evaluate_selected_test(checkpoints=paths, data_dir=folder, demo=True)
    with pytest.raises(ValueError, match="64 lowercase hex"):
        selected(folder, paths, "not-a-sha")
    (folder / "test.txt").write_bytes(b"Replaced heldout content, not the pinned test!")
    with pytest.raises(ValueError, match="operator-pinned"):
        selected(folder, paths, digest)


def test_reject_duplicate_and_too_short_test(m32_demo):
    folder, paths, digest = m32_demo
    test_file = folder / "test.txt"
    test_file.write_bytes((folder / "validation.txt").read_bytes())
    with pytest.raises(ValueError, match="duplicate train or validation"):
        selected(folder, paths, sha256(test_file.read_bytes()).hexdigest())
    test_file.write_bytes(b"x")
    with pytest.raises(ValueError, match="at least two tokens"):
        selected(folder, paths, sha256(test_file.read_bytes()).hexdigest())


def test_reject_checkpoint_mutation_between_selection_and_scoring(m32_demo, monkeypatch):
    import assess_selected_test as module

    folder, paths, digest = m32_demo
    verified = audit_history(checkpoints=paths, data_dir=folder, demo=True,
                             stride=4, require_lineage=True)
    index = min(range(3), key=lambda i: verified["history"][i]["mean_nll"])
    previous = module.audit_history

    def sabotage(*args, **kwargs):
        result = previous(*args, **kwargs)
        with paths[index].open("ab") as stream:
            stream.write(b"file modified after validation")
        return result

    monkeypatch.setattr(module, "audit_history", sabotage)
    with pytest.raises(ValueError, match="modified after validation"):
        selected(folder, paths, digest)


def test_cli_report_create_only_and_reproducible(m32_demo, tmp_path):
    folder, paths, digest = m32_demo
    output = tmp_path / "selected-test.json"
    cmd = [
        sys.executable, "assess_selected_test.py", "--demo",
        "--data-dir", str(folder), "--expected-test-sha256", digest,
        "--require-lineage", "--stride", "4", "--device", "cpu",
        "--output", str(output),
    ]
    for path in paths:
        cmd += ["--checkpoint", str(path)]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    report = json.loads(output.read_text())
    assert report == json.loads(completed.stdout)
    repeat = subprocess.run(cmd, capture_output=True, text=True)
    assert repeat.returncode != 0
    assert "File exists" in repeat.stderr
    assert json.loads(output.read_text()) == report


def test_production_requires_review_before_checkpoint_loading(tmp_path, monkeypatch):
    import assess_selected_test as module
    folder = tmp_path / "data"
    folder.mkdir()
    with pytest.raises(ValueError, match="Production requires"):
        module.evaluate_selected_test(
            checkpoints=[str(tmp_path / "absent1.pt"), str(tmp_path / "absent2.pt")],
            data_dir=str(folder), demo=False,
        )
