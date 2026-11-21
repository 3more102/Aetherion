"""Regression coverage for byte compatibility and end-to-end BPE wiring."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from aetherion.encoding import load_bpe_splits, sha256_file
from generate import encode_prompt
from train import load_training_data


ROOT = Path(__file__).resolve().parents[1]


def test_byte_mode_is_backward_compatible_and_separate_val_is_enforced(tmp_path):
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "validation.txt"
    train_path.write_bytes(b"abcdefghijklmnopqrstuvwxyz" * 8)
    val_path.write_bytes(b"independent validation sample" * 7)
    args = SimpleNamespace(data=str(train_path), val_data=None, tokenizer=None)
    train, val, vocab, info = load_training_data(args)
    assert vocab == 256 and info["tokenizer"]["mode"] == "byte"
    assert len(train) + len(val) == len(train_path.read_bytes())
    assert encode_prompt("Hi", {"config": {"vocab_size": 256}}, None) == ([72, 105], None)
    with pytest.raises(ValueError, match="Byte checkpoint"):
        encode_prompt("Hi", {"config": {"vocab_size": 256}}, "incorrect.json")

    args.val_data = str(val_path)
    train, val, vocab, info = load_training_data(args)
    assert train.tolist() == list(train_path.read_bytes())
    assert val.tolist() == list(val_path.read_bytes())
    assert info["dataset"]["split_policy"] == "explicit-byte-files"
    args.val_data = str(train_path)
    with pytest.raises(ValueError, match="different files"):
        load_training_data(args)


def test_bpe_preparation_training_generation_and_hash_guard(tmp_path):
    pytest.importorskip("tokenizers")
    from aetherion.tokenizer import train_tokenizer

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    train_path = prepared / "train.jsonl"
    val_path = prepared / "validation.jsonl"
    train_path.write_text("".join(
        json.dumps({"text": f"Training example {i} مرحبا Hello world! " * 2},
                   ensure_ascii=False) + "\n"
        for i in range(70)
    ), encoding="utf-8")
    val_path.write_text("".join(
        json.dumps({"text": f"Validation-only sample {i} Bonjour مرحبا" * 2},
                   ensure_ascii=False) + "\n"
        for i in range(15)
    ), encoding="utf-8")
    (prepared / "manifest.json").write_text(json.dumps({
        "splits": {
            "train": {"sha256": sha256_file(train_path)},
            "validation": {"sha256": sha256_file(val_path)},
        }
    }), encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer.json"
    details = train_tokenizer(train_path, tokenizer_path, vocab_size=320,
                              min_frequency=1)
    assert Path(details["metadata_path"]).exists()
    train, val, metadata = load_bpe_splits(train_path, val_path, tokenizer_path)
    assert len(train) > 50 and len(val) > 50
    assert metadata["tokenizer"]["vocab_size"] == details["vocab_size"]
    assert metadata["dataset"]["train_sha256"] == sha256_file(train_path)

    checkpoint_path = tmp_path / "bpe.pt"
    run = subprocess.run([
        sys.executable, "train.py", "--data", str(train_path),
        "--val-data", str(val_path), "--tokenizer", str(tokenizer_path),
        "--steps", "1", "--batch-size", "2", "--block-size", "12",
        "--d-model", "32", "--layers", "1", "--heads", "4",
        "--device", "cpu", "--output", str(checkpoint_path),
    ], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["tokenizer"]["mode"] == "bpe"
    assert checkpoint["config"]["vocab_size"] == details["vocab_size"]

    generation = subprocess.run([
        sys.executable, "generate.py", "--checkpoint", str(checkpoint_path),
        "--tokenizer", str(tokenizer_path), "--prompt", "Hello",
        "--tokens", "2", "--device", "cpu",
    ], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert generation.returncode == 0, generation.stderr
    assert generation.stdout.startswith("Hello")

    # Appending insignificant JSON whitespace changes bytes but not validity.
    tokenizer_path.write_text(tokenizer_path.read_text(encoding="utf-8") + "\n",
                              encoding="utf-8")
    with pytest.raises(ValueError, match="checksum does not match checkpoint"):
        encode_prompt("Hello", checkpoint, str(tokenizer_path))


def test_bpe_validation_and_tokenizer_train_identity_rejected(tmp_path):
    pytest.importorskip("tokenizers")
    from aetherion.tokenizer import train_tokenizer

    folder = tmp_path / "prepared"
    folder.mkdir()
    train_path = folder / "train.jsonl"
    val_path = folder / "validation.jsonl"
    train_path.write_text('{"text": "training ABCDEF"}\n' * 10, encoding="utf-8")
    val_path.write_text('{"text": "validating XYZ"}\n' * 10, encoding="utf-8")
    manifest_path = folder / "manifest.json"
    manifest_path.write_text(json.dumps({
        "splits": {
            "train": {"sha256": sha256_file(train_path)},
            "validation": {"sha256": sha256_file(val_path)}
        }
    }), encoding="utf-8")
    tok = tmp_path / "tok.json"
    train_tokenizer(train_path, tok, vocab_size=280, min_frequency=1)
    val_path.write_text(val_path.read_text(encoding="utf-8") + "\n",
                        encoding="utf-8")
    with pytest.raises(ValueError, match="validation checksum"):
        load_bpe_splits(train_path, val_path, tok)

    manifest_path.write_text(json.dumps({
        "splits": {
            "train": {"sha256": sha256_file(train_path)},
            "validation": {"sha256": sha256_file(val_path)}
        }
    }), encoding="utf-8")
    train_path.write_text(train_path.read_text(encoding="utf-8") + "\n",
                          encoding="utf-8")
    manifest_path.write_text(json.dumps({
        "splits": {
            "train": {"sha256": sha256_file(train_path)},
            "validation": {"sha256": sha256_file(val_path)}
        }
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="fitted on this training split"):
        load_bpe_splits(train_path, val_path, tok)
