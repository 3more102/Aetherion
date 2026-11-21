"""Deterministic split, provenance, and tokenizer baseline tests."""
import json
from pathlib import Path

import pytest

from aetherion.data import prepare_corpus


def _rows(directory: Path, split: str):
    return [json.loads(line) for line in (directory / f"{split}.jsonl").read_text(
        encoding="utf-8").splitlines()]


def test_document_level_split_is_deterministic_and_deduplicated(tmp_path):
    rows = [{
        "id": f"doc-{i:03d}",
        "group_id": f"group-{i // 2}",
        "text": f"Sample {i}: مرحبا Hello world.",
        "source": "fixture",
        "license": "test-only",
        "language": "mixed",
    } for i in range(160)]
    rows.append({**rows[0], "id": "duplicate-content"})
    original = tmp_path / "input.jsonl"
    original.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                        encoding="utf-8")
    first, second = tmp_path / "one", tmp_path / "two"
    manifest = prepare_corpus(original, first, seed=41)
    reversed_input = tmp_path / "reversed.jsonl"
    reversed_input.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in reversed(rows)) + "\n",
        encoding="utf-8",
    )
    prepare_corpus(reversed_input, second, seed=41)
    assert manifest["exact_duplicates_excluded"] == 1
    assert manifest["input_documents"] == 161
    groups = {}
    texts = set()
    for name in ("train", "validation", "test"):
        assert (first / f"{name}.jsonl").read_bytes() == (second / f"{name}.jsonl").read_bytes()
        for row in _rows(first, name):
            assert row["text"] not in texts
            texts.add(row["text"])
            assert groups.setdefault(row["group_id"], name) == name
    assert len(texts) == 160
    assert (first / "manifest.json").is_file()


def test_missing_license_and_duplicate_ids_are_rejected(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text('{"id":"x","text":"hi","source":"x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="license"):
        prepare_corpus(source, tmp_path / "out")
    record = {"id": "x", "text": "hello", "source": "fixture", "license": "test-only"}
    source.write_text("\n".join(json.dumps(record) for _ in range(2)), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate document id"):
        prepare_corpus(source, tmp_path / "out")


def test_split_percentages_are_checked(tmp_path):
    with pytest.raises(ValueError, match="below 100"):
        prepare_corpus(tmp_path / "missing", tmp_path / "out",
                       validation_pct=60, test_pct=40)


def test_bpe_multilingual_roundtrip(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    from aetherion.tokenizer import train_tokenizer

    source = tmp_path / "train.jsonl"
    samples = ["Hello, world!", "مرحبا بالعالم", "Café déjà vu", "中文测试", "line  one"]
    source.write_text("".join(
        json.dumps({"text": s}, ensure_ascii=False) + "\n" for s in samples * 3
    ), encoding="utf-8")
    output = tmp_path / "tokenizer.json"
    details = train_tokenizer(source, output, vocab_size=320, min_frequency=1)
    tokenizer = tokenizers.Tokenizer.from_file(str(output))
    assert details["vocab_size"] <= 320
    for sample in samples:
        assert tokenizer.decode(tokenizer.encode(sample).ids) == sample
    with pytest.raises(ValueError, match="train.jsonl"):
        train_tokenizer(tmp_path / "validation.jsonl", output)
