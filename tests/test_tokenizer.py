"""Regression tests for train-only BPE fitting and artifact integrity."""
import json

import pytest

from aetherion.data import prepare
from aetherion.tokenizer import train_tokenizer, verify_training_split


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "authorized.jsonl"
    texts = [
        "Hello, world!  two spaces.\nNext line.",
        "مرحبا بالعالم — العربية",
        "Emoji 🛰️🚀🙂 and 中文",
        "Café déjà vu and tabs\tinside",
    ]
    records = [
        {
            "id": f"doc-{i}",
            "source": f"source-{i}",
            "group": f"group-{i}",
            "license": "CC0-1.0",
            "authorized": True,
            "text": texts[i % len(texts)] + f" unique {i}",
        }
        for i in range(20)
    ]
    source.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    destination = tmp_path / "prepared"
    prepare([source], destination, seed=42)
    return destination


def test_bpe_roundtrip_reload_and_determinism(prepared, tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    a = tmp_path / "run-a" / "tokenizer.json"
    b = tmp_path / "run-b" / "tokenizer.json"
    first = train_tokenizer(prepared / "train.jsonl", a, vocab_size=400, min_frequency=1)
    second = train_tokenizer(prepared / "train.jsonl", b, vocab_size=400, min_frequency=1)
    assert a.read_bytes() == b.read_bytes()
    assert a.with_suffix(".meta.json").read_bytes() == b.with_suffix(".meta.json").read_bytes()
    assert first["train_sha256"] == second["train_sha256"]
    assert first["tokenizer_sha256"] == second["tokenizer_sha256"]
    assert first["vocab_size"] <= 400
    assert first["train_documents"] > 0
    tokenizer = tokenizers.Tokenizer.from_file(str(a))
    reloaded = tokenizers.Tokenizer.from_file(str(b))
    for text in (
        "Hello, world!  two spaces.\nNext line.",
        "مرحبا بالعالم",
        "中文测试 🛰️🚀🙂",
        "Café déjà vu\ttabs\n",
        "  leading and trailing  ",
    ):
        token_ids = tokenizer.encode(text).ids
        assert tokenizer.decode(token_ids) == text
        assert reloaded.encode(text).ids == token_ids
    with pytest.raises(FileExistsError, match="overwrite"):
        train_tokenizer(prepared / "train.jsonl", a)


def test_rejects_test_validation_and_tampered_data(prepared):
    with pytest.raises(ValueError, match="train.jsonl"):
        verify_training_split(prepared / "validation.jsonl")
    with pytest.raises(ValueError, match="train.jsonl"):
        verify_training_split(prepared / "test.jsonl")
    train = prepared / "train.jsonl"
    train.write_bytes(train.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_training_split(train)


def test_rejects_missing_manifest_and_invalid_config(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text('{"text":"example"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest.json"):
        verify_training_split(train)
    with pytest.raises(ValueError, match="259"):
        train_tokenizer(train, tmp_path / "tokenizer.json", vocab_size=258)
    with pytest.raises(ValueError, match="min_frequency"):
        train_tokenizer(train, tmp_path / "tokenizer.json", min_frequency=0)
