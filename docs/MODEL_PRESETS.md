# Aetherion model presets and parameter audit

## Actual architecture targets

| Preset | Vocabulary | Layers | Width | Heads | Context | Trainable parameters |
|---|---:|---:|---:|---:|---:|---:|
| `micro` | 256 | 4 | 128 | 4 | 128 | Inspect with CLI |
| `aetherion-125m` | 32,000 | 14 | 768 | 12 | 2,048 | **123,688,704** |

The 125M preset name is approximate: the model is a 123.69M-parameter decoder-only Transformer with tied input/output embeddings, RMSNorm, RoPE and SwiGLU. It is **not yet pretrained**, and these values do not measure quality or inference speed.

## Inspect safely without a GPU

```bash
python inspect_model.py --preset aetherion-125m --weight-dtype bf16
python inspect_model.py --preset micro --weight-dtype fp32
pytest -q tests/test_presets.py
```

The CLI builds the model on PyTorch's `meta` device, which stores shapes rather than allocating the full parameter arrays. The report includes exact trainable parameter counts, embedding weight tying, architecture configuration and **theoretical weight-only storage**.

At 123,688,704 parameters, BF16 weights alone require approximately **235.92 MiB** (247.38 MB). **This is not a GPU VRAM recommendation:** actual training memory also includes gradients, optimizer states, activations, attention temporaries, data batches, and fragmentation. It must be measured on the target hardware before any pretraining budget is approved.

## Derivation of 125M parameter target

- Tied token embedding: 32,000 × 768 = 24,576,000
- Each block's QKV + output projections: 4 × 768² = 2,359,296
- Each block's SwiGLU feedforward: 3 × 768 × 2,048 = 4,718,592
- Each block's RMSNorm weights: 2 × 768 = 1,536
- Blocks: 14 × (2,359,296 + 4,718,592 + 1,536) = 99,111,936
- Final RMSNorm: 768
- **Total = 123,688,704 trainable parameters**

The calculation counts tied embeddings only once. Automated tests check the exact model implementation using `sum(p.numel() for p in model.parameters())`.

## Required next feasibility gate

Before initiating significant training, benchmark actual target hardware: `tokens/sec`, peak memory, sequence length, batch size, precision, device configuration, checkpoint/resume, dataset size, data provenance, held-out validation loss and estimated total training spend. This parameter audit alone is not that benchmark.

Source: [PyTorch meta device](https://docs.pytorch.org/docs/stable/meta.html).
