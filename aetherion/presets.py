"""Auditable Aetherion architecture presets and exact parameter arithmetic.

These values describe untrained model architectures. Estimation never
instantiates the large 125M model or allocates its parameters.
"""
from dataclasses import asdict
import math
from .model import ModelConfig


PRESETS = {
    "micro": {
        "vocab_size": 256,
        "block_size": 128,
        "n_layers": 4,
        "n_heads": 4,
        "d_model": 128,
        "dropout": 0.0,
    },
    "125m": {
        "vocab_size": 32_000,
        "block_size": 2048,
        "n_layers": 14,
        "n_heads": 12,
        "d_model": 768,
        "dropout": 0.0,
    },
}


def get_preset(name: str) -> ModelConfig:
    """Return a fresh config, never the mutable preset dictionary."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose from {', '.join(sorted(PRESETS))}")
    return ModelConfig(**PRESETS[name])


def parameter_breakdown(cfg: ModelConfig) -> dict[str, int]:
    """Count unique trainable parameters for the Aetherion model.

    Embedding and LM-head weights are tied and counted once. The SwiGLU
    hidden size matches the model's 32-element rounding rule. RoPE
    has no trainable parameters. This is not a memory benchmark.
    """
    hidden = math.ceil((8 * cfg.d_model / 3) / 32) * 32
    embedding = cfg.vocab_size * cfg.d_model
    attention = cfg.n_layers * (4 * cfg.d_model * cfg.d_model)
    ffn = cfg.n_layers * (3 * cfg.d_model * hidden)
    norms = (2 * cfg.n_layers + 1) * cfg.d_model
    total = embedding + attention + ffn + norms
    return {
        "embedding_tied_head": embedding,
        "attention": attention,
        "feedforward": ffn,
        "rmsnorm": norms,
        "total": total,
        "swiglu_hidden": hidden,
    }


def preset_report(name: str) -> dict:
    """Serializable design report, with theoretical FP32 weights-only size."""
    cfg = get_preset(name)
    counts = parameter_breakdown(cfg)
    return {
        "preset": name,
        "config": asdict(cfg),
        "parameter_breakdown": counts,
        "fp32_weight_only_mib": round(counts["total"] * 4 / (1024**2), 2),
        "trained": False,
        "warning": "Architecture estimate only; no checkpoint, throughput or GPU-memory measurement.",
    }
