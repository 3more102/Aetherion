"""M19: ONNX two-graph KV-cached next-byte inference for Aetherion-Micro.

Prefill computes a right-padded fixed window and returns per-layer key/value
tensors. Decode consumes one token and updates one cache slot, avoiding the
quadratic full-window Transformer recomputation until a sliding-window rebuild.
This does NOT perform true continuous position-preserving eviction; for exact
compatibility with Aetherion's original sliding-window generator, rebuild at
the window boundary and reset relative RoPE positions.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path

import torch
from torch import nn

from .mobile_export import MobileNextLogits, ONNX_OPSET, load_mobile_checkpoint
from .presets import parameter_breakdown


def cache_names(n_layers: int, *, prefix: str = "") -> list[str]:
    return [
        f"{prefix}{kind}_{layer}"
        for layer in range(n_layers)
        for kind in ("key", "value")
    ]


def _rotate(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)


class MobileCachedPrefill(MobileNextLogits):
    """Inputs: tokens [1,W], last_position [1].

    Output: next_logits [1,256], then (key_i,value_i) [1,H,W,D] per layer.
    Cache positions past last_position are placeholders and are never attended
    to until overwritten by a real decode step.
    """

    def forward(self, tokens: torch.Tensor, last_position: torch.Tensor):
        x = self.model.embed(tokens)
        cached = []
        for block in self.model.blocks:
            residual = block.attn_norm(x)
            batch, length, width = residual.shape
            heads = block.attn.n_heads
            head_width = width // heads
            query, key, value = block.attn.qkv(residual).chunk(3, dim=-1)
            query = query.reshape(batch, length, heads, head_width).transpose(1, 2)
            key = key.reshape(batch, length, heads, head_width).transpose(1, 2)
            value = value.reshape(batch, length, heads, head_width).transpose(1, 2)
            query = query * self.rope_cos + _rotate(query) * self.rope_sin
            key = key * self.rope_cos + _rotate(key) * self.rope_sin

            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_width)
            scores = scores.masked_fill(~self.causal_mask, torch.finfo(scores.dtype).min)
            probability = torch.softmax(scores, dim=-1)
            attention = torch.matmul(probability, value)
            attention = attention.transpose(1, 2).contiguous().reshape(batch, length, width)
            x = x + block.attn.out(attention)
            x = x + block.ffn(block.ffn_norm(x))
            cached.extend((key, value))

        logits = self.model.lm_head(self.model.norm(x))
        positions = last_position.reshape(1, 1, 1).expand(1, 1, logits.size(-1))
        return (logits.gather(1, positions).squeeze(1), *cached)


class MobileCachedDecode(MobileNextLogits):
    """Inputs: token_ids int64 [1,1], past_length int64 [1], then per-layer cache.

    Requires 1 <= past_length < window. A dynamic ONNX Gather selects the
    correct RoPE phase, and every cache slot is masked by past_length.
    """

    def forward(self, token_ids: torch.Tensor, past_length: torch.Tensor, *past_cache):
        x = self.model.embed(token_ids)
        new_cache = []
        slot_index = torch.arange(self.window, device=token_ids.device).view(1, 1, self.window, 1)
        past_position = past_length.reshape(1, 1, 1, 1)
        write_slot = slot_index == past_position
        valid_slot = (slot_index <= past_position).transpose(-1, -2)
        cos = self.rope_cos.index_select(2, past_length.reshape(1))
        sin = self.rope_sin.index_select(2, past_length.reshape(1))

        for i, block in enumerate(self.model.blocks):
            residual = block.attn_norm(x)
            batch, length, width = residual.shape
            heads = block.attn.n_heads
            head_width = width // heads
            query, key, value = block.attn.qkv(residual).chunk(3, dim=-1)
            query = query.reshape(batch, length, heads, head_width).transpose(1, 2)
            key = key.reshape(batch, length, heads, head_width).transpose(1, 2)
            value = value.reshape(batch, length, heads, head_width).transpose(1, 2)
            query = query * cos + _rotate(query) * sin
            key = key * cos + _rotate(key) * sin

            updated_key = torch.where(write_slot, key, past_cache[2 * i])
            updated_value = torch.where(write_slot, value, past_cache[2 * i + 1])
            scores = torch.matmul(query, updated_key.transpose(-2, -1)) / math.sqrt(head_width)
            scores = scores.masked_fill(~valid_slot, torch.finfo(scores.dtype).min)
            probability = torch.softmax(scores, dim=-1)
            attended = torch.matmul(probability, updated_value)
            attended = attended.transpose(1, 2).contiguous().reshape(batch, length, width)
            x = x + block.attn.out(attended)
            x = x + block.ffn(block.ffn_norm(x))
            new_cache.extend((updated_key, updated_value))

        logits = self.model.lm_head(self.model.norm(x))
        return (logits[:, -1, :], *new_cache)


def export_mobile_cached(
    checkpoint: str | Path, output_dir: str | Path, *, window: int = 64
) -> dict:
    """Write prefill.onnx, decode.onnx, manifest.json (schema 2), no weights fallback."""
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    model, payload = load_mobile_checkpoint(checkpoint)
    if not 2 <= window <= model.cfg.block_size:
        raise ValueError("window must be between 2 and checkpoint block_size")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg
    cache_shape = (1, cfg.n_heads, window, cfg.d_model // cfg.n_heads)
    cache_inputs = tuple(torch.zeros(cache_shape, dtype=torch.float32)
                         for _ in range(2 * cfg.n_layers))
    names = cache_names(cfg.n_layers)
    outputs = ["next_logits", *names]
    prefill_path = output_dir / "prefill.onnx"
    decode_path = output_dir / "decode.onnx"
    common = dict(
        output_names=outputs, opset_version=ONNX_OPSET,
        dynamo=False, do_constant_folding=True,
    )
    torch.onnx.export(
        MobileCachedPrefill(model, window).eval(),
        (torch.zeros(1, window, dtype=torch.long), torch.tensor([0], dtype=torch.long)),
        str(prefill_path), input_names=["tokens", "last_position"], **common
    )
    torch.onnx.export(
        MobileCachedDecode(model, window).eval(),
        (torch.zeros(1, 1, dtype=torch.long), torch.tensor([1], dtype=torch.long), *cache_inputs),
        str(decode_path),
        input_names=["token_ids", "past_length", *cache_names(cfg.n_layers, prefix="past_")],
        **common,
    )
    manifest = {
        "schema_version": 2,
        "architecture": "aetherion-micro",
        "tokenizer_type": "utf8_bytes",
        "vocab_size": 256,
        "window": window,
        "cache_policy": "rebuild_relative_rope_at_window_boundary",
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "head_dim": cfg.d_model // cfg.n_heads,
        "model_config": asdict(cfg),
        "parameter_count": parameter_breakdown(cfg)["total"],
        "onnx_opset": ONNX_OPSET,
        "graphs": {
            "prefill": {"filename": prefill_path.name, "sha256": sha256(prefill_path.read_bytes()).hexdigest()},
            "decode": {"filename": decode_path.name, "sha256": sha256(decode_path.read_bytes()).hexdigest()},
        },
        "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_step": payload.get("step"),
        "status": "kv_cached_inference_only_not_language_quality_validated",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
