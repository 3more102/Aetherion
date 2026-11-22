import json
import subprocess
import sys

import pytest
import torch

from aetherion import Aetherion, ModelConfig
from aetherion.instruct import encode_example, batch_examples, prompt_ids, load_splits
from aetherion.data import prepare
from aetherion.tokenizer import train_tokenizer
from aetherion.tokenizer_runtime import load_verified_tokenizer


def test_cache_prefill_chunks_and_overflow():
    torch.manual_seed(1)
    torch.set_num_threads(1)
    model = Aetherion(ModelConfig(vocab_size=32, block_size=8, d_model=16, n_layers=2, n_heads=2)).eval()
    ids = torch.arange(7).unsqueeze(0)
    full, _ = model(ids)
    a, cache = model.forward_cached(ids[:, :3])
    b, cache = model.forward_cached(ids[:, 3:5], cache)
    c, cache = model.forward_cached(ids[:, 5:], cache)
    torch.testing.assert_close(torch.cat((a, b, c), dim=1), full, atol=1e-6, rtol=1e-5)
    expected = ids
    for _ in range(5):
        logits, _ = model(expected[:, -8:])
        expected = torch.cat((expected, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    assert torch.equal(expected, model.generate_cached(ids, 5, temperature=0))
    assert torch.equal(ids, model.generate_cached(ids, 0))
    with pytest.raises(ValueError, match="exceeds"):
        model.forward_cached(ids[:, :2], cache)


@pytest.fixture
def tokenizer_artifact(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps({"id": str(i), "source": str(i), "license": "CC0-1.0",
                                      "authorized": True, "text": f"Document {i}: User: Hello Assistant: Hi. " * 6 + str(i)})+"\n"
                           for i in range(20)))
    data = tmp_path / "data"
    prepare([raw], data)
    artifact = tmp_path / "tokenizer.json"
    train_tokenizer(data / "train.jsonl", artifact, vocab_size=300, min_frequency=1)
    tokenizer, metadata = load_verified_tokenizer(artifact)
    return tokenizer, metadata, artifact


def test_sft_mask_and_split_guard(tokenizer_artifact, tmp_path):
    tok, _, _ = tokenizer_artifact
    x, y = encode_example(tok, "Hello", "Hi", 64)
    prefix = prompt_ids(tok, "Hello")
    assert torch.all(y[:len(prefix)-1] == -100)
    assert y[len(prefix)-1] != -100
    assert y[-1] == tok.token_to_id("<|eos|>")
    x2, y2 = encode_example(tok, "Hello there", "Hi", 64)
    bx, by = batch_examples([(x, y), (x2, y2)], tok.token_to_id("<|eos|>"), "cpu")
    assert bx.shape == by.shape
    assert torch.all(by[0, len(y):] == -100)
    with pytest.raises(ValueError, match="exceeds"):
        encode_example(tok, "Hello", "Hi", 2)
    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    row = {"instruction": "Hello", "response": "Hi", "source": "synthetic",
           "group": "same", "license": "CC0-1.0", "authorized": True}
    train.write_text(json.dumps(row))
    val.write_text(json.dumps({**row, "instruction": "Other"}))
    with pytest.raises(ValueError, match="cross-split"):
        load_splits(train, val, tok, 64)


def test_finetune_and_chat_cli(tokenizer_artifact, tmp_path):
    tok, meta, artifact = tokenizer_artifact
    model = Aetherion(ModelConfig(vocab_size=tok.get_vocab_size(), block_size=64,
                                  d_model=16, n_layers=1, n_heads=2))
    from dataclasses import asdict
    base, out = tmp_path / "base.pt", tmp_path / "sft.pt"
    torch.save({"config": asdict(model.cfg), "state_dict": model.state_dict(),
                "data": {"tokenizer_type": "bpe", "tokenizer_sha256": meta["tokenizer_sha256"]}}, base)
    train, val = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    row = {"instruction": "Hello", "response": "Hi", "source": "synthetic",
           "group": "train", "license": "CC0-1.0", "authorized": True}
    train.write_text(json.dumps(row))
    val.write_text(json.dumps({**row, "group":"validation", "instruction":"Good day"}))
    cmd = [sys.executable, "finetune.py", "--base", str(base), "--tokenizer", str(artifact),
           "--train", str(train), "--validation", str(val), "--output", str(out),
           "--device", "cpu", "--batch-size", "1"]
    subprocess.run(cmd + ["--steps", "1"], check=True, capture_output=True)
    subprocess.run(cmd + ["--steps", "2", "--resume", str(out)], check=True, capture_output=True)
    checkpoint = torch.load(out, map_location="cpu", weights_only=True)
    assert checkpoint["step"] == 2 and checkpoint["data"]["stage"] == "sft"
    subprocess.run([sys.executable, "chat.py", "--checkpoint", str(out), "--tokenizer",
                    str(artifact), "--prompt", "Hello", "--tokens", "2", "--device", "cpu"],
                   check=True, capture_output=True)
