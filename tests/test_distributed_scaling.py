import subprocess
import sys

import pytest
import torch

from aetherion import Aetherion, ModelConfig
from aetherion.presets import get_preset, parameter_breakdown, preset_report


def test_scaled_presets():
    counts = [parameter_breakdown(get_preset(name))["total"]
              for name in ("125m", "350m", "1.3b", "7b")]
    assert counts == sorted(counts)
    for actual, target in zip(counts, (125e6, 350e6, 1.3e9, 7e9)):
        assert actual == pytest.approx(target, rel=0.08)
    assert preset_report("7b")["trained"] is False
    assert preset_report("7b")["fp32_adamw_parameter_gradient_moments_gib"] > 90


def test_activation_checkpoint_equivalence():
    torch.set_num_threads(1)
    torch.manual_seed(21)
    cfg = ModelConfig(vocab_size=16, block_size=8, d_model=16, n_layers=2, n_heads=2)
    a, b = Aetherion(cfg), Aetherion(cfg)
    b.load_state_dict(a.state_dict())
    b.activation_checkpointing = True
    x = torch.arange(8).view(1, -1)
    y = (x+1) % 16
    for model in (a, b):
        model(x,y)[1].backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo required")
def test_two_rank_resume(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("synthetic text for distributed regression " * 60)
    full, resumed = tmp_path / "full.pt", tmp_path / "resumed.pt"
    shared = [sys.executable, "-m", "torch.distributed.run", "--standalone",
              "--nproc-per-node=2", "train_distributed.py", "--device", "cpu",
              "--data", str(data), "--context", "8", "--schedule-steps", "4",
              "--accumulation", "2", "--activation-checkpointing"]
    def run(output, steps, extra=()):
        cmd = shared + ["--output", str(output), "--steps", str(steps), *extra]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=150)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "tokens_per_update=32" in result.stdout
    run(full, 2)
    run(resumed, 1)
    run(resumed, 2, ("--resume", str(resumed)))
    a, b = (torch.load(path, map_location="cpu", weights_only=True) for path in (full, resumed))
    assert a["step"] == b["step"] == 2
    for key in a["state_dict"]:
        torch.testing.assert_close(a["state_dict"][key], b["state_dict"][key], rtol=1e-6, atol=1e-8)
    assert b["train_hparams"]["world_size"] == 2
