# Aetherion design notes

## System boundary

```text
Authorized text corpus
        │
        ▼
Data validator → deduplication → provenance manifest → document-level split
        │
        ▼
Tokenizer training → tokenizer artifact → packed training tokens
        │
        ▼
Model config → Decoder-only Transformer ← optimizer / schedule
                    │
                checkpoint
                    │
        held-out evaluation → model card → inference service
```

## Model computation

```text
token ids [B,T]
    │
    ▼
embedding [B,T,D]
    │
    ├── repeat L blocks ──────────────────────────────────────────┐
    │       RMSNorm → QKV projections → RoPE(Q,K) → causal SDPA   │
    │          → output projection → residual                    │
    │       RMSNorm → SwiGLU → residual                           │
    └─────────────────────────────────────────────────────────────┘
    │
    ▼
final RMSNorm → tied vocabulary head → logits [B,T,V]
    │
    ├─ training: cross entropy on shifted next-token targets
    └─ inference: temperature/top-k autoregressive sampling
```

## Two distinct configurations

**Micro (implemented as the first prototype):** 256 raw UTF-8 byte tokens, width 128, 4 layers, 4 attention heads, context 128. This is convenient for local verification but inefficient for actual language modeling.

**125M (future research preset):** 32,000-token BPE, width 768, 14 layers, 12 heads, SwiGLU hidden ~2048, context 2048, tied embedding. Rough count:
- embeddings: 32,000 × 768 = 24,576,000
- attention projection weights per block: 4 × 768² = 2,359,296
- SwiGLU weights per block: 3 × 768 × 2048 = 4,718,592
- 14 blocks weights: 99,090,432
- + norms: about 33k parameters
- **total ~123.7M trainable parameters**. Exact count must come from `sum(p.numel() for p in model.parameters())`.

## Attention implementation

Use `torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)`; RoPE applied to Q/K before attention. Enforce even head size and `d_model % n_heads == 0`. With a KV cache, position offsets and correct causal masking must be handled explicitly; Micro recomputes context and has no KV cache.

## Engineering requirements

- Unit-test attention causality, shapes, gradients and NaNs.
- Separate validation and test documents from training and tokenizer fitting to prevent leakage.
- Train with explicit seed, checkpoint metadata, resume support and logs.
- Prefer straightforward correctness-first implementations; profile before optimization.
- Model release requires reproducible evaluation and documented data provenance.

## Sources
- https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- https://huggingface.co/docs/tokenizers/main/quicktour
- https://huggingface.co/docs/course/chapter7/6
