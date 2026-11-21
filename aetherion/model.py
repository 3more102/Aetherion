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


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """Rotary embeddings for tensor shape (batch, heads, time, head_dim)."""
    time, width = x.size(-2), x.size(-1)
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, width, 2, device=x.device, dtype=torch.float32) / width)
    )
    phase = torch.outer(torch.arange(time, device=x.device, dtype=torch.float32), inv_freq)
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
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
        k = k.view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
        v = v.view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
        q, k = apply_rope(q), apply_rope(k)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(y.transpose(1, 2).contiguous().view(b, t, c))


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
