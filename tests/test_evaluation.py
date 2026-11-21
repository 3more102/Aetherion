"""Tests for deterministic held-out evaluation and state preservation."""
import math
from types import SimpleNamespace

import pytest
import torch

from aetherion.evaluation import evaluate_tokens


class UniformLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size=16, block_size=4):
        super().__init__()
        self.cfg = SimpleNamespace(block_size=block_size)
        self.vocab_size = vocab_size
        self.visited_lengths = []

    def forward(self, tokens):
        self.visited_lengths.append(tokens.size(1))
        b, t = tokens.shape
        return torch.zeros(b, t, self.vocab_size, device=tokens.device), None


def test_exact_target_count_and_uniform_perplexity():
    model = UniformLanguageModel()
    model.train()
    tokens = torch.arange(10, dtype=torch.long) % model.vocab_size
    metrics = evaluate_tokens(model, tokens, block_size=4)
    assert metrics["tokens"] == 9
    assert metrics["context_window"] == 4
    assert metrics["nll"] == pytest.approx(math.log(16), abs=1e-6)
    assert metrics["perplexity"] == pytest.approx(16.0, abs=1e-5)
    assert model.visited_lengths == [4, 4, 1]
    assert model.training  # restore original state


def test_eval_state_and_invalid_sequences():
    model = UniformLanguageModel().eval()
    with pytest.raises(ValueError, match="at least two"):
        evaluate_tokens(model, torch.tensor([1]), block_size=4)
    with pytest.raises(ValueError, match="block_size"):
        evaluate_tokens(model, torch.tensor([1, 2]), block_size=5)
    evaluate_tokens(model, torch.tensor([1, 2]), block_size=4)
    assert not model.training
