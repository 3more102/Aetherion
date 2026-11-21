"""BPE tokenizer integrity, Unicode round-trip and data isolation tests."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("tokenizers")

from aetherion.data import prepare
from aetherion.tokenizer import AetherionTokenizer, SPECIAL_TOKENS, train_tokenizer


@pytest.fixture
def prepared(tmp_path):
    raw = tmp_path / "raw.jsonl"
    examples = (
        "Hello, Aetherion!\nNew line and   spaces.",
        "مرحبا بالعالم، كيف الحال؟",
        "عِلْمُ الذَّكَاءِ الاصْطِنَاعِيِّ.",
        "✨🤖🧠 Test emojis!",
        "Café naïve, résumé.",
        "中文测试与日本語テスト。",
        "Un texte en français.",
        "Testing tabs\tand\nnewlines!",
        "Dobrý den, světe.",
        "¡Hola, mundo! ¿Cómo estás?",
        "Another clearly separate source.",
        "Unique last source containing extra words.",
    )
    with raw.open("w", encoding="utf-8") as f:
        for i, sample in enumerate(examples):
            for j in range(2):
                f.write(json.dumps({
                    "id": f"{i}-{j}", "source": f"src-{i}",
                    "license": "CC0-1.0", "authorized": True,
                    "text": sample + f" Document {j} group {i}.",
                }, ensure_ascii=False) + "\n")
    data_dir = tmp_path / "data"
    prepare([raw], data_dir, seed=42)
    return data_dir


def test_roundtrip_reload_and_reproducible_artifacts(prepared, tmp_path):
    first = train_tokenizer(prepared, tmp_path / "one", vocab_size=320, min_frequency=1)
    second = train_tokenizer(prepared, tmp_path / "two", vocab_size=320, min_frequency=1)
    assert first.sha256 == second.sha256
    assert first.vocab_size == second.vocab_size
    assert first.vocab_size <= 320
    assert first.vocab_size >= 260
    for i, special in enumerate(SPECIAL_TOKENS):
        assert first.metadata["special_token_ids"][special] == i
    reloaded = AetherionTokenizer.load(tmp_path / "one")
    samples = [
        "Hello, world!",
        "مرحبا بالعالم! الذكاء الاصطناعي",
        "🧠🤖✨", " Café\twith spaces\nand newlines  ",
        "中文、日本語", "hello\r\nworld", "Mixed عربي English 🌍",
    ]
    for sample in samples:
        encoded = reloaded.encode(sample)
        assert encoded == second.encode(sample)
        assert reloaded.decode(encoded) == sample


def test_only_verified_training_split_and_tamper_detection(prepared, tmp_path):
    # Test/validation files are not used for fitting.
    (prepared / "validation.jsonl").write_text("not relevant to fitting", encoding="utf-8")
    (prepared / "test.jsonl").write_text("also not used for fitting", encoding="utf-8")
    trained = train_tokenizer(prepared, tmp_path / "bpe", vocab_size=300, min_frequency=1)
    assert trained.metadata["training_scope"].startswith("verified train.jsonl only")
    token_path = tmp_path / "bpe" / "tokenizer.json"
    token_path.write_bytes(token_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        AetherionTokenizer.load(tmp_path / "bpe")
    with pytest.raises(FileExistsError):
        train_tokenizer(prepared, tmp_path / "bpe", vocab_size=300)


def test_tampered_train_manifest_and_bad_config(prepared, tmp_path):
    with pytest.raises(ValueError, match="at least 260"):
        train_tokenizer(prepared, tmp_path / "impossible", vocab_size=10)
    with pytest.raises(ValueError, match="min_frequency"):
        train_tokenizer(prepared, tmp_path / "impossible", min_frequency=0)
    (prepared / "train.jsonl").write_text('{"text":"untrusted new data"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        train_tokenizer(prepared, tmp_path / "not-trained")


def test_micro_model_can_train_with_bpe(prepared, tmp_path):
    import subprocess
    import sys
    artifact_dir = tmp_path / "bpe"
    tok = train_tokenizer(prepared, artifact_dir, vocab_size=320, min_frequency=1)
    checkpoint = tmp_path / "ckpt.pt"
    result = subprocess.run(
        [sys.executable, "train.py",
         "--data", str(prepared / "train.txt"),
         "--val-data", str(prepared / "validation.txt"),
         "--tokenizer-dir", str(artifact_dir), "--steps", "2",
         "--batch-size", "2", "--d-model", "32", "--layers", "2",
         "--heads", "4", "--block-size", "16",
         "--output", str(checkpoint)],
        capture_output=True, text=True, check=True,
    )
    assert "saved_checkpoint" in result.stdout
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["config"]["vocab_size"] == tok.vocab_size
    assert saved["data"]["tokenizer_sha256"] == tok.sha256
    generated = subprocess.run(
        [sys.executable, "generate.py",
         "--checkpoint", str(checkpoint), "--tokenizer-dir", str(artifact_dir),
         "--prompt", "Hello", "--tokens", "2", "--device", "cpu"],
        capture_output=True, text=True, check=True,
    )
    assert generated.stdout.startswith("Hello")
