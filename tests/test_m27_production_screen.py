"""M27: 125M launch and production evaluation reject known split contamination.

These tests use synthetic fixtures and mock a 32k tokenizer; they do NOT
authorize a real corpus or create pretrained model weights.
"""
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

import pilot_train
from aetherion.contamination import require_no_cross_split_findings
from aetherion.corpus_gate import audit_corpus
from aetherion.data import prepare
from compare_checkpoints import compare


BASE = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi"


def prepared_fixture(tmp_path, *, near_duplicate=False):
    groups = [f"m27-group-{i}" for i in range(12)]
    ordered = sorted(groups, key=lambda g: sha256(f"42:{g}".encode()).hexdigest())
    # The first and tenth hashed groups fall in separate prepared splits.
    offenders = set((ordered[0], ordered[9])) if near_duplicate else set()
    records = []
    for i, group in enumerate(groups):
        body = (BASE + (" one" if group == ordered[0] else " two")) if group in offenders else (
            " ".join(f"distinct{i}_word{j}" for j in range(24)))
        records.append({"id": f"row{i}", "source": group, "group": group,
                        "license": "CC0-1.0", "authorized": True, "text": body})
    source = tmp_path / "synthetic.jsonl"
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    destination = tmp_path / "prepared"
    prepare([source], destination, seed=42)
    audit = audit_corpus(destination, ["CC0-1.0"])
    review_path = tmp_path / "synthetic-review.json"
    review_path.write_text(json.dumps({
        "manifest_sha256": audit["manifest_sha256"],
        "allowed_licenses": ["CC0-1.0"],
        "reviewer": "synthetic test fixture only",
        "rights_reviewed": True,
        "privacy_reviewed": True,
        "near_duplicates_reviewed": True,
        "contamination_reviewed": True,
    }), encoding="utf-8")
    return destination, review_path


def pilot_args(directory, review):
    return SimpleNamespace(
        data_dir=str(directory), review=str(review), tokenizer="dummy-tokenizer.json",
        allow_license=["CC0-1.0"], context=16, steps=2,
        schedule_steps=4, warmup_steps=0, batch_size=1, accumulation=1,
        precision="fp32", device="cpu", output="synthetic-steps.pt", resume=None,
    )


def test_clean_fixture_retains_explicit_dry_run_without_training(monkeypatch, tmp_path):
    directory, review = prepared_fixture(tmp_path)
    screen = require_no_cross_split_findings(directory, ["CC0-1.0"])
    assert screen["findings_count"] == 0
    called = []
    class TokenizerStub:
        def get_vocab_size(self):
            return 32000
    def tokenizer_stub(path):
        called.append(path)
        return TokenizerStub(), {"manifest_sha256": screen["manifest_sha256"]}
    monkeypatch.setattr(pilot_train, "load_verified_tokenizer", tokenizer_stub)
    cmd = pilot_train.pilot_command(pilot_args(directory, review))
    assert called == ["dummy-tokenizer.json"]
    assert "--block-size" in cmd and cmd[cmd.index("--block-size") + 1] == "16"
    assert "--steps" in cmd and cmd[cmd.index("--steps") + 1] == "2"
    assert "--output" in cmd and "synthetic-steps.pt" in cmd
    assert not (tmp_path / "synthetic-steps.pt").exists()


def test_self_attestation_cannot_override_detected_cross_split_leakage(monkeypatch, tmp_path):
    directory, review = prepared_fixture(tmp_path, near_duplicate=True)
    called = []
    monkeypatch.setattr(pilot_train, "load_verified_tokenizer",
                        lambda _: called.append(True))
    with pytest.raises(ValueError, match="Cross-split lexical contamination"):
        pilot_train.pilot_command(pilot_args(directory, review))
    assert called == [], "Tokenizer/model must not initialize after screening failure"


def test_production_checkpoint_comparison_rejects_before_opening_weights(tmp_path):
    directory, review = prepared_fixture(tmp_path, near_duplicate=True)
    # Nonexistent checkpoint paths are intentional; no checkpoint read is permitted.
    with pytest.raises(ValueError, match="Cross-split lexical contamination"):
        compare(before=tmp_path / "missing-before.pt",
                after=tmp_path / "missing-after.pt",
                data_dir=directory, tokenizer="dummy-tokenizer.json",
                review=review, allow_licenses=("CC0-1.0",), device="cpu")


def test_integrity_budget_and_license_failures_are_not_bypassed(tmp_path):
    directory, review = prepared_fixture(tmp_path)
    with pytest.raises(ValueError, match="License outside allowlist"):
        pilot_train.pilot_command(SimpleNamespace(
            **{**vars(pilot_args(directory, review)), "allow_license": ["forbidden"]}))
    from aetherion.contamination import screen_cross_split
    with pytest.raises(ValueError, match="Document scan budget"):
        screen_cross_split(directory, ["CC0-1.0"], max_documents=2)
    (directory / "validation.txt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        pilot_train.pilot_command(pilot_args(directory, review))
