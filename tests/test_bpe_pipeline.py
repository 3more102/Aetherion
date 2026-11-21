"""End-to-end BPE training and inference with a tiny, synthetic corpus."""
import json
import sys

import pytest
import torch

from aetherion.encoding import encode_documents, load_bpe
from aetherion.tokenizer import train_tokenizer
from generate import main as generate_main
from train import main as train_main


def _make_splits(tmp_path):
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "validation.jsonl"
    train_path.write_text("".join(
        json.dumps({"text": ("Hello world! مرحبا بالعالم. " * 16) + str(i)}, ensure_ascii=False) + "\n"
        for i in range(8)
    ), encoding="utf-8")
    val_path.write_text("".join(
        json.dumps({"text": ("This is independent validation. أهلا وسهلا. " * 16) + str(i)}, ensure_ascii=False) + "\n"
        for i in range(4)
    ), encoding="utf-8")
    return train_path, val_path


def test_bpe_end_to_end_and_tokenizer_integrity(tmp_path, monkeypatch, capsys):
    pytest.importorskip("tokenizers")
    train_path, val_path = _make_splits(tmp_path)
    tokenizer_path = tmp_path / "tokenizer.json"
    train_tokenizer(train_path, tokenizer_path, vocab_size=320, min_frequency=1)

    tokenizer = load_bpe(tokenizer_path)
    ids = encode_documents(train_path, tokenizer)
    assert ids.numel() > 80
    assert (ids == tokenizer.token_to_id("<|eos|>")).sum().item() == 8

    checkpoint_path = tmp_path / "micro.pt"
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--data", str(train_path), "--val-data", str(val_path),
        "--tokenizer", str(tokenizer_path), "--steps", "2",
        "--batch-size", "2", "--block-size", "8",
        "--d-model", "32", "--layers", "1", "--heads", "4",
        "--eval-every", "1", "--device", "cpu", "--output", str(checkpoint_path)
    ])
    train_main()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["encoding"] == "bpe"
    assert checkpoint["config"]["vocab_size"] == tokenizer.get_vocab_size()
    assert checkpoint["data_sha256"] != checkpoint["val_data_sha256"]
    assert checkpoint["tokenizer_sha256"]

    monkeypatch.setattr(sys, "argv", [
        "generate.py", "--checkpoint", str(checkpoint_path),
        "--tokenizer", str(tokenizer_path), "--prompt", "Hello مرحبا",
        "--tokens", "0", "--device", "cpu"
    ])
    generate_main()
    assert capsys.readouterr().out.strip().endswith("Hello مرحبا")

    # Tampering with the tokenizer invalidates the checkpoint pairing.
    tokenizer_path.write_text(tokenizer_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        generate_main()


def test_bpe_requires_separate_validation_split(tmp_path, monkeypatch):
    pytest.importorskip("tokenizers")
    train_path, _ = _make_splits(tmp_path)
    tokenizer_path = tmp_path / "tokenizer.json"
    train_tokenizer(train_path, tokenizer_path, vocab_size=320, min_frequency=1)
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--data", str(train_path), "--tokenizer", str(tokenizer_path),
        "--steps", "1", "--block-size", "8"
    ])
    with pytest.raises(SystemExit):
        train_main()
