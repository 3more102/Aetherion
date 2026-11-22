"""Portable fixed-window ONNX inference for byte-token Aetherion-Micro.

ONNX is a separate inference artifact; no training code is modified.
A fixed input shape minimizes cross-browser operator/shape differences.
This exporter is for engineering validation, not a claim of useful weights.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path

import torch
from torch import nn

from .model import Aetherion, ModelConfig
from .presets import parameter_breakdown


MAX_MOBILE_PARAMETERS = 5_000_000
ONNX_OPSET = 17


class MobileNextLogits(nn.Module):
    """Export-friendly full-window decoder; outputs one selected token's logits.

    Inputs: tokens int64 [1, window], last_position int64 [1].
    Output: next_logits float32 [1, vocab_size].
    Future token slots are zero-filled; causality prevents them affecting output.
    """

    def __init__(self, model: Aetherion, window: int):
        super().__init__()
        if window < 2 or window > model.cfg.block_size:
            raise ValueError("window must be between 2 and the model's block_size")
        self.model = model.eval()
        self.window = window
        # Precompute constant-window RoPE before ONNX tracing. Dynamic arange/
        # repeat_interleave lowering can yield incompatible inferred shapes.
        head_width = model.cfg.d_model // model.cfg.n_heads
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, head_width, 2, dtype=torch.float32) / head_width)
        )
        phase = torch.outer(torch.arange(window, dtype=torch.float32), inv_freq)
        phase = torch.repeat_interleave(phase, 2, dim=-1)
        self.register_buffer(
            "rope_cos", phase.cos().reshape(1, 1, window, head_width), persistent=False
        )
        self.register_buffer(
            "rope_sin", phase.sin().reshape(1, 1, window, head_width), persistent=False
        )
        self.register_buffer(
            "causal_mask",
            torch.ones(window, window, dtype=torch.bool).tril().view(1, 1, window, window),
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor, last_position: torch.Tensor) -> torch.Tensor:
        x = self.model.embed(tokens)
        for block in self.model.blocks:
            residual = block.attn_norm(x)
            batch, length, width = residual.shape
            heads = block.attn.n_heads
            head_width = width // heads
            query, key, value = block.attn.qkv(residual).chunk(3, dim=-1)
            query = query.reshape(batch, length, heads, head_width).transpose(1, 2)
            key = key.reshape(batch, length, heads, head_width).transpose(1, 2)
            value = value.reshape(batch, length, heads, head_width).transpose(1, 2)
            rotated_q = torch.stack((-query[..., 1::2], query[..., 0::2]), dim=-1).flatten(-2)
            rotated_k = torch.stack((-key[..., 1::2], key[..., 0::2]), dim=-1).flatten(-2)
            query = query * self.rope_cos + rotated_q * self.rope_sin
            key = key * self.rope_cos + rotated_k * self.rope_sin
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_width)
            scores = scores.masked_fill(~self.causal_mask, torch.finfo(scores.dtype).min)
            probabilities = torch.softmax(scores, dim=-1)
            attended = torch.matmul(probabilities, value)
            attended = attended.transpose(1, 2).contiguous().reshape(batch, length, width)
            x = x + block.attn.out(attended)
            x = x + block.ffn(block.ffn_norm(x))
        logits = self.model.lm_head(self.model.norm(x))
        positions = last_position.reshape(1, 1, 1).expand(1, 1, logits.size(-1))
        return logits.gather(1, positions).squeeze(1)


def load_mobile_checkpoint(path: str | Path) -> tuple[Aetherion, dict]:
    """Accept only trusted, compatible byte-level checkpoints, not BPE or random fallback."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "config" not in payload or "state_dict" not in payload:
        raise ValueError("Not an Aetherion state_dict checkpoint")
    cfg = ModelConfig(**payload["config"])
    token_type = payload.get("data", {}).get("tokenizer_type", "bytes")
    if token_type != "bytes" or cfg.vocab_size != 256:
        raise ValueError("Mobile M17 supports only 256-ID UTF-8 byte tokenization; BPE is not yet supported")
    if parameter_breakdown(cfg)["total"] > MAX_MOBILE_PARAMETERS:
        raise ValueError("Checkpoint exceeds the M17 mobile parameter safety limit")
    model = Aetherion(cfg).float().eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def export_mobile_checkpoint(
    checkpoint: str | Path, output_dir: str | Path, *, window: int = 64
) -> dict:
    """Write model.onnx and verifiable manifest.json, with no fabricated pretrained state."""
    checkpoint, output_dir = Path(checkpoint), Path(output_dir)
    model, payload = load_mobile_checkpoint(checkpoint)
    if window > model.cfg.block_size or window < 2:
        raise ValueError("Requested window is outside the model's context limits")
    output_dir.mkdir(parents=True, exist_ok=True)
    module = MobileNextLogits(model, window).eval()
    example_tokens = torch.zeros((1, window), dtype=torch.long)
    example_pos = torch.tensor([0], dtype=torch.long)
    out_path = output_dir / "model.onnx"
    # The static legacy path avoids changing the ONNX input contract across
    # torch.export versions; parity must be verified separately using ORT.
    torch.onnx.export(
        module, (example_tokens, example_pos), str(out_path),
        input_names=["tokens", "last_position"],
        output_names=["next_logits"], opset_version=ONNX_OPSET,
        dynamo=False, do_constant_folding=True,
    )
    manifest = {
        "schema_version": 1,
        "architecture": "aetherion-micro",
        "tokenizer_type": "utf8_bytes",
        "vocab_size": 256,
        "window": window,
        "model_config": asdict(model.cfg),
        "parameter_count": parameter_breakdown(model.cfg)["total"],
        "onnx_opset": ONNX_OPSET,
        "onnx_sha256": sha256(out_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_step": payload.get("step"),
        "status": "inference_export_only_not_quality_validated",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
