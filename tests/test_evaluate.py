"""Regression coverage for the deterministic Micro evaluator."""
from dataclasses import asdict
import math

import pytest
import torch

from aetherion import Aetherion, ModelConfig
from evaluate import evaluate_checkpoint


def test_evaluation_is_deterministic_and_covers_validation(tmp_path):
    cfg = ModelConfig(block_size=8, d_model=32, n_heads=4, n_layers=2)
    model = Aetherion(cfg).eval()
    checkpoint = tmp_path / 'model.pt'
    torch.save({'config': asdict(cfg), 'state_dict': model.state_dict()}, checkpoint)
    data = tmp_path / 'data.txt'
    data.write_bytes(b'hello, world!\n' * 20)
    a = evaluate_checkpoint(checkpoint, data)
    b = evaluate_checkpoint(checkpoint, data)
    assert a == b
    assert a['evaluated_byte_tokens'] == len(data.read_bytes()) - int(0.9 * len(data.read_bytes())) - 1
    assert a['perplexity_per_byte'] == pytest.approx(math.exp(a['nll_nats_per_byte']))
    assert a['evaluated_chunks'] >= 1


def test_evaluation_batch_limit_and_short_validation(tmp_path):
    cfg = ModelConfig(block_size=8, d_model=32, n_heads=4, n_layers=2)
    model = Aetherion(cfg)
    checkpoint = tmp_path / 'model.pt'
    torch.save({'config': asdict(cfg), 'state_dict': model.state_dict()}, checkpoint)
    data = tmp_path / 'data.txt'
    data.write_bytes(b'x' * 150)
    result = evaluate_checkpoint(checkpoint, data, max_batches=1)
    assert result['evaluated_byte_tokens'] == 8
    with pytest.raises(ValueError, match='max_batches'):
        evaluate_checkpoint(checkpoint, data, max_batches=-1)
    data.write_bytes(b'too short')
    with pytest.raises(ValueError, match='Validation split'):
        evaluate_checkpoint(checkpoint, data)
