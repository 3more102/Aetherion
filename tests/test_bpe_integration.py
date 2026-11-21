"""End-to-end BPE training and inference regression tests (CPU, tiny model)."""
from __future__ import annotations

import json
import sys

import pytest
import torch

from aetherion.token_data import load_training_splits


def _write_split(path, texts):
    path.write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts),
        encoding="utf-8",
    )


def test_bpe_train_checkpoint_and_generate(tmp_path, monkeypatch, capsys):
    tokenizers = pytest.importorskip("tokenizers")
    from aetherion.tokenizer import train_tokenizer
    import train
    import generate

    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "validation.jsonl"
    _write_split(train_path, ["Hello world مرحبا بالعالم! " + str(i) for i in range(60)])
    _write_split(val_path, ["Different validation مثال آخر! " + str(i) for i in range(24)])
    tokenizer_path = tmp_path / "tokenizer.json"
    details = train_tokenizer(train_path, tokenizer_path, vocab_size=320, min_frequency=1)
    checkpoint_path = tmp_path / "micro.pt"

    monkeypatch.setattr(sys, "argv", [
        "train.py", "--data", str(train_path), "--val-data", str(val_path),
        "--tokenizer", str(tokenizer_path), "--steps", "2",
        "--batch-size", "2", "--block-size", "8", "--layers", "1",
        "--heads", "4", "--d-model", "32", "--eval-every", "1",
        "--device", "cpu", "--output", str(checkpoint_path),
    ])
    train.main()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["config"]["vocab_size"] == details["vocab_size"]
    assert checkpoint["tokenizer"]["mode"] == "bpe"
    assert checkpoint["tokenizer"]["sha256"] == details["sha256"]
    assert checkpoint["data"]["split_policy"] == "explicit-files"
    assert checkpoint["tokens_seen"] == 32
    assert checkpoint["step"] == 2

    monkeypatch.setattr(sys, "argv", [
        "generate.py", "--checkpoint", str(checkpoint_path),
        "--tokenizer", str(tokenizer_path), "--prompt", "مرحبا Hello",
        "--tokens", "0", "--device", "cpu",
    ])
    generate.main()
    assert capsys.readouterr().out.splitlines()[-1] == "مرحبا Hello"

    # A tokenizer with even a one-byte artifact change must not be used.
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_bytes(tokenizer_path.read_bytes() + b"\n")
    monkeypatch.setattr(sys, "argv", [
        "generate.py", "--checkpoint", str(checkpoint_path),
        "--tokenizer", str(tampered_path), "--prompt", "Hello", "--tokens", "1",
    ])
    with pytest.raises(ValueError, match="SHA-256"):
        generate.main()

    # Missing tokenizer is a CLI error, not an accidental byte-mode decode.
    monkeypatch.setattr(sys, "argv", [
        "generate.py", "--checkpoint", str(checkpoint_path), "--prompt", "Hello",
    ])
    with pytest.raises(SystemExit) as exc:
        generate.main()
    assert exc.value.code == 2


def test_bpe_cannot_derive_validation_from_train(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    from aetherion.tokenizer import train_tokenizer

    train_path = tmp_path / "train.jsonl"
    _write_split(train_path, ["sample"] * 30)
    tokenizer_path = tmp_path / "tokenizer.json"
    train_tokenizer(train_path, tokenizer_path, vocab_size=280, min_frequency=1)
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    with pytest.raises(ValueError, match="explicit --val-data"):
        load_training_splits(train_path, None, tokenizer)

    val_path = tmp_path / "validation.jsonl"
    val_path.write_bytes(train_path.read_bytes())
    with pytest.raises(ValueError, match="identical SHA-256"):
        load_training_splits(train_path, val_path, tokenizer)


def test_jsonl_loader_uses_document_text_not_metadata(tmp_path):
    from aetherion.token_data import load_document_tokens

    input_file = tmp_path / "train.jsonl"
    _write_split(input_file, ["Hi", "Yo"])
    ids = load_document_tokens(input_file).tolist()
    assert bytes(ids) == b"Hi\nYo\n"
