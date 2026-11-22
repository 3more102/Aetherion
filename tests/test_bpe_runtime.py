"""Regression coverage for existing tokenizer artifacts integrated with resumable Micro."""
import json
import subprocess
import sys

import pytest
import torch

pytest.importorskip("tokenizers")

from aetherion.data import prepare
from aetherion.tokenizer import train_tokenizer
from aetherion.tokenizer_runtime import load_verified_tokenizer
from train import load_bpe_data


@pytest.fixture
def bpe_project(tmp_path):
    raw = tmp_path / "authorized.jsonl"
    with raw.open("w", encoding="utf-8") as stream:
        for group in range(20):
            for copy in range(2):
                stream.write(json.dumps({
                    "id": f"{group}-{copy}", "source": f"source-{group}",
                    "license": "CC0-1.0", "authorized": True,
                    "text": (
                        f"Hello Aetherion! مرحبا 🤖. Training document {group} "
                        f"number {copy}. This sentence contains enough tokens "
                        "to test BPE training and validation safely."
                    ),
                }, ensure_ascii=False) + "\n")
    data = tmp_path / "prepared"
    prepare([raw], data)
    artifact = tmp_path / "tokens" / "tokenizer.json"
    metadata = train_tokenizer(data / "train.jsonl", artifact, vocab_size=400, min_frequency=1)
    return data, artifact, metadata, tmp_path


def test_bpe_matches_train_and_validation_dataset_manifest(bpe_project):
    data, artifact, metadata, _ = bpe_project
    tokenizer, verified = load_verified_tokenizer(artifact)
    assert tokenizer.get_vocab_size() == verified["vocab_size"]
    assert verified["tokenizer_sha256"] == metadata["tokenizer_sha256"]
    train, val, hashes, vocab = load_bpe_data(
        str(data / "train.txt"), str(data / "validation.txt"), str(artifact)
    )
    assert train.numel() > 10
    assert val.numel() > 10
    assert hashes["tokenizer_type"] == "bpe"
    assert hashes["tokenizer_sha256"] == verified["tokenizer_sha256"]
    assert vocab == tokenizer.get_vocab_size()


def test_bpe_checkpoint_resume_and_generation(bpe_project):
    data, artifact, _, tmp = bpe_project
    checkpoint = tmp / "bpe-micro.pt"
    shared = [
        sys.executable, "train.py",
        "--data", str(data / "train.txt"),
        "--val-data", str(data / "validation.txt"),
        "--tokenizer", str(artifact),
        "--steps", "2", "--batch-size", "2",
        "--d-model", "32", "--layers", "1", "--heads", "4",
        "--block-size", "8", "--eval-every", "1",
        "--save-every", "1", "--device", "cpu", "--output", str(checkpoint),
    ]
    first = subprocess.run(shared, capture_output=True, text=True, check=True)
    assert "saved_checkpoint" in first.stdout
    first_checkpoint = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert first_checkpoint["format_version"] == 2
    assert first_checkpoint["data"]["tokenizer_type"] == "bpe"
    assert first_checkpoint["config"]["vocab_size"] == first_checkpoint["data"]["tokenizer_vocab_size"]
    resumed = shared.copy()
    step_index = resumed.index("--steps") + 1
    resumed[step_index] = "3"
    resumed += ["--resume", str(checkpoint)]
    subprocess.run(resumed, capture_output=True, text=True, check=True)
    final_checkpoint = torch.load(checkpoint, weights_only=True, map_location="cpu")
    assert final_checkpoint["step"] == 3
    sample = subprocess.run([
        sys.executable, "generate.py",
        "--checkpoint", str(checkpoint), "--tokenizer", str(artifact),
        "--prompt", "Hello", "--tokens", "2", "--device", "cpu"
    ], capture_output=True, text=True, check=True)
    assert sample.stdout.startswith("Hello")


def test_reject_mismatched_tokenizer_and_corrupted_splits(bpe_project):
    data, artifact, _, tmp = bpe_project
    with pytest.raises(ValueError, match="requires --val-data"):
        load_bpe_data(str(data / "train.txt"), None, str(artifact))
    with pytest.raises(ValueError, match="prepared train.txt"):
        load_bpe_data(str(data / "test.txt"), str(data / "validation.txt"), str(artifact))
    broken = tmp / "copied-tokenizer.json"
    broken.write_bytes(artifact.read_bytes() + b"\n")
    broken.with_suffix(".meta.json").write_bytes(artifact.with_suffix(".meta.json").read_bytes())
    with pytest.raises(ValueError, match="SHA-256"):
        load_verified_tokenizer(broken)
    (data / "validation.txt").write_text("modified after preparation", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_bpe_data(str(data / "train.txt"), str(data / "validation.txt"), str(artifact))


def test_bpe_validation_evaluator(bpe_project):
    data, artifact, _, tmp = bpe_project
    checkpoint = tmp / "evaluation.pt"
    data_args = ["--data", str(data / "train.txt"), "--val-data",
                 str(data / "validation.txt"), "--tokenizer", str(artifact)]
    subprocess.run([sys.executable, "train.py", *data_args, "--steps", "1",
                    "--batch-size", "1", "--d-model", "16", "--layers", "1",
                    "--heads", "2", "--block-size", "8", "--device", "cpu",
                    "--output", str(checkpoint)], check=True, capture_output=True)
    completed = subprocess.run([sys.executable, "evaluate.py", *data_args,
                                "--checkpoint", str(checkpoint), "--device", "cpu"],
                               check=True, capture_output=True, text=True)
    report = json.loads(completed.stdout)
    _, val, _, _ = load_bpe_data(str(data / "train.txt"), str(data / "validation.txt"), str(artifact))
    assert report["evaluated_targets"] == val.numel() - 1
    assert report["tokenizer_type"] == "bpe"
    assert report["perplexity"] > 0
