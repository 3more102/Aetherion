"""Checkpoint/resume is bitwise equivalent on the same CPU backend."""
from argparse import Namespace

import pytest
import torch

from aetherion.checkpoint import load_resume
from train import run_training


def options(tmp_path, *, steps, output, resume=None):
    return Namespace(data=str(tmp_path / "train.txt"),
                     val_data=str(tmp_path / "validation.txt"),
                     steps=steps, batch_size=2, block_size=12,
                     d_model=32, layers=1, heads=4,
                     lr=3e-4, eval_every=3, seed=42,
                     device="cpu", output=str(output),
                     resume=str(resume) if resume else None, save_every=0)


def make_data(tmp_path):
    (tmp_path / "train.txt").write_text("authorized training sentence. " * 25, encoding="utf-8")
    (tmp_path / "validation.txt").write_text("different held out sentence. " * 20, encoding="utf-8")


def test_pause_and_resume_matches_uninterrupted(tmp_path):
    torch.set_num_threads(1)
    make_data(tmp_path)
    full = tmp_path / "full.pt"
    partial = tmp_path / "partial.pt"
    resumed = tmp_path / "resumed.pt"
    run_training(options(tmp_path, steps=5, output=full))
    # Final-step evaluation at step 2 happens only in the partial run.
    # It must not perturb the shared training RNG.
    run_training(options(tmp_path, steps=2, output=partial))
    run_training(options(tmp_path, steps=5, output=resumed, resume=partial))
    a = torch.load(full, map_location="cpu", weights_only=True)
    b = torch.load(resumed, map_location="cpu", weights_only=True)
    assert a["step"] == b["step"] == 5
    assert a["data"] == b["data"]
    assert a["training"] == b["training"]
    assert a["optimizer_state"]["param_groups"] == b["optimizer_state"]["param_groups"]
    for key in a["state_dict"]:
        assert torch.equal(a["state_dict"][key], b["state_dict"][key]), key
    for param_id in a["optimizer_state"]["state"]:
        for key, value in a["optimizer_state"]["state"][param_id].items():
            assert torch.equal(value, b["optimizer_state"]["state"][param_id][key]), key
    assert torch.equal(a["rng_state"]["cpu"], b["rng_state"]["cpu"])


def test_rejects_data_mismatch_on_resume(tmp_path):
    torch.set_num_threads(1)
    make_data(tmp_path)
    initial = tmp_path / "checkpoint.pt"
    run_training(options(tmp_path, steps=1, output=initial))
    (tmp_path / "validation.txt").write_text("modified validation text " * 30, encoding="utf-8")
    with pytest.raises(ValueError, match="data identity"):
        run_training(options(tmp_path, steps=2, output=initial, resume=initial))


def test_rejects_changed_hyperparameters(tmp_path):
    torch.set_num_threads(1)
    make_data(tmp_path)
    initial = tmp_path / "checkpoint.pt"
    run_training(options(tmp_path, steps=1, output=initial))
    args = options(tmp_path, steps=2, output=initial, resume=initial)
    args.lr *= 2
    with pytest.raises(ValueError, match="Training settings"):
        run_training(args)


def test_legacy_checkpoint_is_inference_only(tmp_path):
    older = tmp_path / "old.pt"
    torch.save({"config": {"block_size": 8}, "state_dict": {}}, older)
    with pytest.raises(ValueError, match="legacy inference-only"):
        load_resume(older, config={}, data_hashes={}, settings={}, device="cpu")


def test_periodic_checkpoint(tmp_path):
    torch.set_num_threads(1)
    make_data(tmp_path)
    args = options(tmp_path, steps=3, output=tmp_path / "periodic.pt")
    args.save_every = 2
    run_training(args)
    saved = torch.load(args.output, weights_only=True)
    assert saved["step"] == 3
    assert saved["checkpoint_schema"] == 2
