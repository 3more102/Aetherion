"""Regression tests for model architecture reports without heavy allocations."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from aetherion import Aetherion, ModelConfig
from aetherion.presets import get_preset, parameter_breakdown, preset_report


def test_125m_exact_parameter_count_without_model_allocation():
    cfg = get_preset("125m")
    assert (cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.vocab_size, cfg.block_size) == (
        14, 768, 12, 32_000, 2048
    )
    report = parameter_breakdown(cfg)
    assert report["swiglu_hidden"] == 2048
    assert report["total"] == 123_688_704
    assert sum(report[key] for key in ("embedding_tied_head", "attention", "feedforward", "rmsnorm")) == report["total"]
    assert preset_report("125m")["trained"] is False


@pytest.mark.parametrize("width,layers,vocab,heads", [(32, 2, 256, 4), (48, 1, 99, 6), (128, 4, 256, 4)])
def test_formula_matches_real_unique_parameters(width, layers, vocab, heads):
    cfg = ModelConfig(d_model=width, n_layers=layers, vocab_size=vocab, n_heads=heads, block_size=16)
    model = Aetherion(cfg)
    assert model.lm_head.weight is model.embed.weight
    assert parameter_breakdown(cfg)["total"] == sum(p.numel() for p in model.parameters())


def test_preset_config_is_fresh_and_invalid_names_are_rejected():
    first = get_preset("micro")
    first.block_size = 16
    assert get_preset("micro").block_size == 128
    with pytest.raises(ValueError, match="Unknown preset"):
        get_preset("missing")


def test_inspect_cli_json_has_no_training_claim():
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, str(root / "inspect_model.py"), "--preset", "125m", "--json"],
        cwd=root, capture_output=True, text=True, check=True,
    )
    payload = json.loads(out.stdout)
    assert payload["parameter_breakdown"]["total"] == 123_688_704
    assert payload["trained"] is False
    assert payload["fp32_weight_only_mib"] > 0
