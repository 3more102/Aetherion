"""Correctness smoke tests for Aetherion Micro."""
from dataclasses import asdict
import torch
from aetherion import Aetherion, ModelConfig


def test_forward_backward_and_generate():
    torch.manual_seed(0)
    model = Aetherion(ModelConfig(block_size=16, d_model=32, n_heads=4, n_layers=2))
    x = torch.randint(0, 256, (2, 12))
    y = torch.randint(0, 256, (2, 12))
    logits, loss = model(x, y)
    assert logits.shape == (2, 12, 256)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert model.embed.weight.grad is not None
    out = model.generate(x[:, :4], max_new_tokens=2)
    assert out.shape == (2, 6)


def test_future_tokens_do_not_affect_past_logits():
    torch.manual_seed(0)
    model = Aetherion(ModelConfig(block_size=16, d_model=32, n_heads=4, n_layers=2)).eval()
    a = torch.randint(0, 256, (1, 12))
    b = a.clone()
    b[:, 8:] = torch.randint(0, 256, (1, 4))
    with torch.inference_mode():
        first, _ = model(a)
        second, _ = model(b)
    torch.testing.assert_close(first[:, :8, :], second[:, :8, :], atol=1e-5, rtol=1e-5)


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    cfg = ModelConfig(block_size=16, d_model=32, n_heads=4, n_layers=2)
    model = Aetherion(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    path = tmp_path / "model.pt"
    torch.save({"config": asdict(cfg), "state_dict": model.state_dict()}, path)
    loaded = torch.load(path, weights_only=True, map_location="cpu")
    restored = Aetherion(ModelConfig(**loaded["config"])).eval()
    restored.load_state_dict(loaded["state_dict"])
    with torch.inference_mode():
        original, _ = model(x)
        recreated, _ = restored(x)
    torch.testing.assert_close(original, recreated)
