import json
import math
import subprocess
import sys

import pytest
import torch

from aetherion import Aetherion, ModelConfig
from aetherion.evaluation import evaluate_tokens


def test_uniform_logits_and_exact_coverage():
    model = Aetherion(ModelConfig(vocab_size=16, block_size=4, d_model=16, n_layers=1, n_heads=2))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    tokens = torch.arange(13) % 16
    model.train()
    for stride in (1, 3, 4):
        result = evaluate_tokens(model, tokens, stride=stride)
        assert result["evaluated_targets"] == 12
        assert result["mean_nll"] == pytest.approx(math.log(16))
        assert result["perplexity"] == pytest.approx(16)
        assert model.training
    model.eval()
    evaluate_tokens(model, tokens)
    assert not model.training
    with pytest.raises(ValueError, match="stride"):
        evaluate_tokens(model, tokens, stride=5)


def test_matches_manual_window_scoring():
    torch.manual_seed(12)
    model = Aetherion(ModelConfig(vocab_size=16, block_size=4, d_model=16, n_layers=1, n_heads=2))
    model.eval()
    tokens = torch.arange(8) % 16
    expected = 0.0
    with torch.no_grad():
        for pos in range(1, len(tokens)):
            logits, _ = model(tokens[max(0, pos-4):pos].unsqueeze(0))
            expected -= logits[0, -1].log_softmax(-1)[tokens[pos]].item()
    result = evaluate_tokens(model, tokens, stride=1)
    assert result["evaluated_targets"] == len(tokens)-1
    assert result["total_nll"] == pytest.approx(expected)


def test_byte_checkpoint_cli_and_data_mismatch(tmp_path):
    train, val, ckpt = (tmp_path / name for name in ("train.txt", "validation.txt", "model.pt"))
    train.write_text("training text " * 5, encoding="utf-8")
    val.write_text("validation sample " * 3, encoding="utf-8")
    data_args = ["--data", str(train), "--val-data", str(val)]
    subprocess.run([sys.executable, "train.py", *data_args, "--steps", "1",
                    "--block-size", "4", "--batch-size", "1", "--d-model", "16",
                    "--layers", "1", "--heads", "2", "--device", "cpu",
                    "--output", str(ckpt)], check=True, capture_output=True)
    cmd = [sys.executable, "evaluate.py", *data_args, "--checkpoint", str(ckpt),
           "--device", "cpu", "--stride", "3"]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    report = json.loads(completed.stdout)
    assert report["evaluated_targets"] == len(val.read_bytes())-1
    assert report["checkpoint_step"] == 1
    assert report["tokenizer_type"] == "bytes"
    val.write_text("tampered data", encoding="utf-8")
    rejected = subprocess.run(cmd, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "fingerprints differ" in rejected.stderr
