"""Coverage for the byte/BPE data and checkpoint integration."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from aetherion.token_stream import file_sha256, load_token_stream


class DummyEncoding:
    def __init__(self, ids):
        self.ids = ids


class DummyTokenizer:
    def token_to_id(self, value):
        return 7 if value == "<|eos|>" else None

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return DummyEncoding([len(text), 11])


def test_jsonl_only_loads_document_text_and_appends_eos(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps({"id": f"doc-{i}", "text": t, "license": "test-only"})
                              for i, t in enumerate(["hello", "مرحبا"])) + "\n", encoding="utf-8")
    assert load_token_stream(path, DummyTokenizer()) == [5, 11, 7, 5, 11, 7]
    with pytest.raises(ValueError, match="requires --tokenizer"):
        load_token_stream(path)
    assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_jsonl_rejects_bad_records_and_missing_eos(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"id":"x","text":42}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Missing text"):
        load_token_stream(path, DummyTokenizer())
    path.write_text('{"text":"hello"}\n', encoding="utf-8")

    class NoEos(DummyTokenizer):
        def token_to_id(self, value):
            return None

    with pytest.raises(ValueError, match="eos"):
        load_token_stream(path, NoEos())


def test_byte_training_and_generation_backward_compatible(tmp_path):
    root = Path(__file__).resolve().parents[1]
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world! " * 120, encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    run = subprocess.run([sys.executable, str(root / "train.py"), "--data", str(corpus),
                          "--steps", "1", "--batch-size", "1", "--block-size", "8",
                          "--d-model", "16", "--layers", "1", "--heads", "2",
                          "--output", str(checkpoint), "--device", "cpu"],
                         capture_output=True, text=True, cwd=root, check=True)
    assert "tokenizer=byte" in run.stdout
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["config"]["vocab_size"] == 256
    assert saved["tokenizer_sha256"] is None
    result = subprocess.run([sys.executable, str(root / "generate.py"),
                             "--checkpoint", str(checkpoint), "--prompt", "hi", "--tokens", "0",
                             "--device", "cpu"], capture_output=True, text=True, cwd=root, check=True)
    assert result.stdout.strip() == "hi"


def test_full_bpe_cli_if_available(tmp_path):
    pytest.importorskip("tokenizers")
    from aetherion.tokenizer import train_tokenizer
    root = Path(__file__).resolve().parents[1]
    train_set = tmp_path / "train.jsonl"
    val_set = tmp_path / "validation.jsonl"
    for path, token in ((train_set, "train"), (val_set, "heldout")):
        path.write_text("".join(json.dumps({"text": f"{token} مرحبا example {i}"}, ensure_ascii=False) + "\n"
                                for i in range(50)), encoding="utf-8")
    tok_file = tmp_path / "tokenizer.json"
    tok_info = train_tokenizer(train_set, tok_file, vocab_size=300, min_frequency=1)
    checkpoint = tmp_path / "model.pt"
    training = subprocess.run([sys.executable, str(root / "train.py"), "--data", str(train_set),
                               "--val-data", str(val_set), "--tokenizer", str(tok_file),
                               "--steps", "1", "--batch-size", "1", "--block-size", "8",
                               "--d-model", "16", "--layers", "1", "--heads", "2",
                               "--output", str(checkpoint), "--device", "cpu"],
                              capture_output=True, text=True, cwd=root, check=True)
    assert "tokenizer=bpe" in training.stdout
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["tokenizer_sha256"] == tok_info["sha256"]
    assert saved["config"]["vocab_size"] == tok_info["vocab_size"]
    result = subprocess.run([sys.executable, str(root / "generate.py"),
                             "--checkpoint", str(checkpoint), "--tokenizer", str(tok_file),
                             "--prompt", "مرحبا", "--tokens", "0", "--device", "cpu"],
                            capture_output=True, text=True, cwd=root, check=True)
    assert result.stdout.strip() == "مرحبا"
    missing = subprocess.run([sys.executable, str(root / "generate.py"),
                              "--checkpoint", str(checkpoint), "--tokens", "0", "--device", "cpu"],
                             capture_output=True, text=True, cwd=root)
    assert missing.returncode and "requires --tokenizer" in missing.stderr
    wrong = tmp_path / "wrong.json"
    wrong.write_text(tok_file.read_text(encoding="utf-8") + " ", encoding="utf-8")
    mismatch = subprocess.run([sys.executable, str(root / "generate.py"),
                               "--checkpoint", str(checkpoint), "--tokenizer", str(wrong),
                               "--tokens", "0", "--device", "cpu"],
                              capture_output=True, text=True, cwd=root)
    assert mismatch.returncode and "SHA-256 mismatch" in mismatch.stderr
