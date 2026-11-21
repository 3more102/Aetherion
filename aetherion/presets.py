"""Named, inspectable Aetherion model configurations (not pretrained weights)."""
from __future__ import annotations

from aetherion.model import ModelConfig


PRESET_NAMES = ("micro", "aetherion-125m")


def get_preset(name: str) -> ModelConfig:
    """Create a fresh model config for the named architecture target."""
    if name == "micro":
        return ModelConfig(vocab_size=256, block_size=128, n_layers=4,
                           n_heads=4, d_model=128, dropout=0.0)
    if name == "aetherion-125m":
        return ModelConfig(vocab_size=32_000, block_size=2048, n_layers=14,
                           n_heads=12, d_model=768, dropout=0.0)
    raise ValueError(f"Unknown preset {name!r}; choose from {', '.join(PRESET_NAMES)}")
