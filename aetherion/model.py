"""Aetherion Micro: educational causal decoder-only Transformer."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 256
    block_size: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    dropout: float = 0.0

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("head dimension must be even for RoPE")
        if self.block_size < 2:
            raise ValueError("block_size must be >= 2")


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps).to(x.dtype) * self.weight


def apply_rope(x: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Rotary embeddings for tensor shape (batch, heads, time, head_dim)."""
    time, width = x.size(-2), x.size(-1)
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, width, 2, device=x.device, dtype=torch.float32) / width)
    )
    phase = torch.outer(torch.arange(offset, offset + time, device=x.device, dtype=torch.float32), inv_freq)
    phase = torch.repeat_interleave(phase, 2, dim=-1)
    cos = phase.cos()[None, None, :, :].to(x.dtype)
    sin = phase.sin()[None, None, :, :].to(x.dtype)
    rotated = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)
    return x * cos + rotated * sin


class CausalAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attend(x)[0]

    def attend(self, x, past=None):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
        k = k.view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
        v = v.view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
        offset = 0 if past is None else past[0].size(-2)
        q, k = apply_rope(q, offset), apply_rope(k, offset)
        mask = None
        if past is not None:
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)
            if t > 1:
                mask = torch.arange(k.size(-2), device=x.device)[None, :] <= (
                    offset + torch.arange(t, device=x.device)[:, None])
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=past is None,
            dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(y.transpose(1, 2).contiguous().view(b, t, c)), (k, v)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = math.ceil((8 * cfg.d_model / 3) / 32) * 32
        self.gate_up = nn.Linear(cfg.d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x)))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


    def cached(self, x, past=None):
        y, cache = self.attn.attend(self.attn_norm(x), past)
        x = x + self.dropout(y)
        return x + self.dropout(self.ffn(self.ffn_norm(x))), cache


class Aetherion(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None):
        if token_ids.size(1) > self.cfg.block_size:
            raise ValueError("Input exceeds configured context window")
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.inference_mode()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 80,
                 temperature: float = 0.8, top_k: int = 50):
        self.eval()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        for _ in range(max_new_tokens):
            context = prompt[:, -self.cfg.block_size:]
            logits, _ = self(context)
            scores = logits[:, -1, :] / temperature
            if 0 < top_k < scores.size(-1):
                cutoff = torch.topk(scores, top_k).values[:, -1, None]
                scores = scores.masked_fill(scores < cutoff, -float("inf"))
            next_token = torch.multinomial(F.softmax(scores, dim=-1), 1)
            prompt = torch.cat([prompt, next_token], dim=1)
        return prompt

    @torch.inference_mode()
    def forward_cached(self, token_ids, cache=None):
        """Inference-only KV cache; positions are relative to the current window."""
        if self.training:
            raise ValueError("KV caching requires eval mode")
        if token_ids.ndim != 2 or token_ids.size(1) == 0:
            raise ValueError("Expected nonempty [batch, time] tokens")
        if cache is not None and len(cache) != len(self.blocks):
            raise ValueError("Invalid cache layer count")
        past_len = 0 if cache is None else cache[0][0].size(-2)
        if past_len + token_ids.size(1) > self.cfg.block_size:
            raise ValueError("Cached context exceeds configured window")
        x = self.embed(token_ids)
        new_cache = []
        for i, block in enumerate(self.blocks):
            x, layer_cache = block.cached(x, None if cache is None else cache[i])
            new_cache.append(layer_cache)
        return self.lm_head(self.norm(x)), tuple(new_cache)

    @torch.inference_mode()
    def generate_cached(self, prompt, max_new_tokens=80, temperature=0.8,
                        top_k=50, eos_id=None):
        """Single-sequence decoding; rebuild cache when sliding window shifts."""
        if prompt.ndim != 2 or prompt.size(0) != 1 or prompt.size(1) == 0:
            raise ValueError("Expected one nonempty prompt")
        if max_new_tokens < 0 or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Invalid sampling options")
        self.eval()
        cache = None
        for _ in range(max_new_tokens):
            if cache is None or cache[0][0].size(-2) >= self.cfg.block_size:
                x = prompt[:, -self.cfg.block_size:]
                logits, cache = self.forward_cached(x)
            else:
                logits, cache = self.forward_cached(prompt[:, -1:], cache)
            scores = logits[:, -1, :]
            if temperature == 0:
                token = scores.argmax(dim=-1, keepdim=True)
            else:
                scores = scores / temperature
                if 0 < top_k < scores.size(-1):
                    cutoff = torch.topk(scores, top_k).values[:, -1, None]
                    scores = scores.masked_fill(scores < cutoff, -float("inf"))
                token = torch.multinomial(F.softmax(scores, dim=-1), 1)
            prompt = torch.cat((prompt, token), dim=1)
            if eos_id is not None and token.item() == eos_id:
                break
        return prompt
