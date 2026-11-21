# Aetherion 125M — architecture audit (not trained)

The 125M name is a **size class**, not an exact 125,000,000-parameter claim.
For the present Aetherion PyTorch architecture the proposed preset is:

| Property | Value |
| --- | ---: |
| Decoder layers | 14 |
| Model width | 768 |
| Attention heads | 12 |
| RoPE head width | 64 |
| SwiGLU intermediate width | 2,048 |
| Vocabulary | 32,000 BPE tokens (not yet trained/integrated) |
| Maximum context | 2,048 tokens |
| Attention | Causal scaled dot-product attention |
| Norm | RMSNorm |
| Embedding and output head | Tied weights |

Theoretical unique trainable parameter count:

| Component | Parameters |
| --- | ---: |
| Tied token embedding / output head | 24,576,000 |
| Attention projections over 14 layers | 33,030,144 |
| SwiGLU projections over 14 layers | 66,060,288 |
| RMSNorm scales | 22,272 |
| **Total** | **123,688,704** |

Reproduce the **architecture-only** report without allocating the 125M model:

    python inspect_model.py --preset 125m
    python inspect_model.py --preset 125m --json
    pytest -q tests/test_presets.py

The FP32 weight-only estimate (471.83 MiB) is **not** peak GPU/CPU
memory. Gradients, optimizer state, activations, attention workspaces, data
buffers, checkpointing and allocator overhead add memory. No training run,
BPE corpus, successful pretraining, GPU throughput or accuracy is implied.

## Required gates before pretraining

1. Merge and validate one data provenance/splitting pipeline and 32k BPE tokenizer;
   open M1 pull requests overlap and must be reconciled.
2. Integrate tokenizer IDs into training/generation with checksum verification;
   retain the byte-mode demo separately.
3. Implement optimizer/RNG-state checkpoint resume, accumulation and warmup/cosine
   schedule; run CPU resume-equivalence tests.
4. Benchmark the proposed configuration on identified hardware to record
   actual VRAM, effective tokens/second and compute cost.
5. Authorize a licensed corpus, document quality/contamination checks, and
   measure held-out next-token performance.

Only then consider long-running pretraining; instruction tuning follows
an evaluated base checkpoint.
