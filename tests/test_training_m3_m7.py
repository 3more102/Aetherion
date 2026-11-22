import json
import math
import sys

import pytest
import torch

from aetherion.data import prepare
from aetherion.corpus_gate import audit_corpus, require_review
from aetherion.training import learning_rate, precision_context
from benchmark_training import benchmark
from train import main


def test_schedule():
    kw = dict(peak=1.0, warmup=2, horizon=6, min_ratio=0.1)
    assert learning_rate(1, **kw) == 0.5
    assert learning_rate(2, **kw) == 1.0
    assert learning_rate(4, **kw) == pytest.approx(0.55)
    assert learning_rate(6, **kw) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="exceeds"):
        learning_rate(7, **kw)
    assert learning_rate(50, peak=0.01) == 0.01


def test_accumulation_schedule_resume(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    data = tmp_path / "demo.txt"
    data.write_text("synthetic training corpus " * 30)
    def run(out, steps, resume=False):
        argv = ["train.py", "--data", str(data), "--output", str(out),
                "--steps", str(steps), "--schedule-steps", "4", "--warmup-steps", "1",
                "--accumulation", "2", "--batch-size", "1", "--d-model", "16", "--heads", "2",
                "--layers", "1", "--block-size", "8", "--device", "cpu"]
        if resume:
            argv += ["--resume", str(out)]
        monkeypatch.setattr(sys, "argv", argv)
        main()
        return torch.load(out, weights_only=True, map_location="cpu")
    full = run(tmp_path / "full.pt", 4)
    run(tmp_path / "resume.pt", 2)
    resumed = run(tmp_path / "resume.pt", 4, True)
    for key, value in full["state_dict"].items():
        assert torch.equal(value, resumed["state_dict"][key])
    assert torch.equal(full["train_generator_state"], resumed["train_generator_state"])
    assert full["optimizer_state_dict"]["param_groups"] == resumed["optimizer_state_dict"]["param_groups"]


def test_cpu_bf16_and_benchmark():
    torch.set_num_threads(1)
    with precision_context("cpu", "bf16"):
        assert (torch.ones(2, 2) @ torch.ones(2, 2)).dtype == torch.bfloat16
    report = benchmark(device="cpu", precision="bf16", context=8, steps=1, warmup=0)
    assert report["tokens"] == 8
    assert math.isfinite(report["last_loss"])
    assert report["tokens_per_second"] > 0
    assert report["cuda_peak_allocated_bytes"] is None


def test_corpus_review_and_tamper(tmp_path):
    raw = tmp_path / "source.jsonl"
    raw.write_text("".join(json.dumps({"id": str(i), "source": str(i), "license": "CC0-1.0",
                                       "authorized": True, "text": f"Synthetic document {i}"})+"\n"
                           for i in range(10)))
    dest = tmp_path / "prepared"
    prepare([raw], dest)
    report = audit_corpus(dest, ["CC0-1.0"])
    with pytest.raises(ValueError, match="allowlist"):
        audit_corpus(dest, ["MIT"])
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"manifest_sha256": report["manifest_sha256"],
                                  "allowed_licenses": ["CC0-1.0"], "reviewer": "test fixture",
                                  "rights_reviewed": True, "privacy_reviewed": True,
                                  "near_duplicates_reviewed": True, "contamination_reviewed": True}))
    require_review(report, review)
    review.write_text("{}")
    with pytest.raises(ValueError, match="attestation"):
        require_review(report, review)
    (dest / "test.txt").write_text("tampered")
    with pytest.raises(ValueError, match="Checksum"):
        audit_corpus(dest, ["CC0-1.0"])
