# Aetherion — engineering execution plan

**Goal:** independently implement, train, evaluate, and eventually serve an open-source decoder-only language model. Frontier-model parity is not a supported or near-term claim. Progress is determined by measurable exit criteria, not calendar deadlines.

## Architecture and workstreams

```text
Licensed/authorized text
  -> provenance + rights manifest
  -> cleaning, deduplication, privacy filtering
  -> document-disjoint train/validation/test splits
  -> tokenizer trained only on training text
  -> packed token sequences and dataset checksums
  -> decoder-only Transformer (RoPE / RMSNorm / SwiGLU / causal SDPA)
  -> reproducible pretraining and checkpoints
  -> held-out evaluation, safety checks, model card
  -> inference API and chat client
```

## Phases and deliverables

| Phase | Indicative sprint | Deliverables | Exit criteria |
|---|---|---|---|
| M0: Micro | 1 | Verified forward/backward pass, sampling, checkpoint load, deterministic evaluation and CI | All tests pass; 5 train steps complete; checkpoint reloads and generates; JSON metrics captured |
| M1: Data + BPE | 2–3 | Licensed text manifest, duplicate detection, document-level splits, 32k BPE experiment | Document sources and licenses auditable; UTF-8/Arabic/English round-trips; zero train/test document overlap |
| M2: ~124M baseline | 4–6* | 14-layer, 768-width, 12-head configuration; 2048 context; training with grad accumulation, mixed precision, resumable state | Parameter count verified; small-data overfit check; measured throughput, memory, expense, and held-out NLL |
| M3: Instruct | 7–9* | Licensed supervised fine-tuning data, chat template, behavior and safety evaluation | Measured task improvements against base model and disclosed limitations |
| M4: Serve | 10–12* | REST inference service, basic chat UI, deployment notes | Contract tests, security review, latency/throughput data |
| M5: Scale study | Beyond | Distributed-training design and scaling-budget review | Feasibility based on GPU benchmarks, data availability and cost |

`*` Only planning targets: GPU access, training budget and dataset rights can materially change timing.

## Sprint 1 — work in order

1. Establish known-good Micro CI and validation metrics with checkpoint and data hashes.
2. Introduce tokenizer abstraction; implement a train-only BPE tokenizer and round-trip tests.
3. Implement a JSONL ingestion schema (`id`, `source`, `license`, `text`, `sha256`), deduplication and document-based splits.
4. Upgrade training with optimizer/scheduler resume, RNG state, token-based logging, gradient accumulation and warmup/cosine decay.
5. Add Aetherion-125M config, param-count and VRAM smoke tests before GPU training.
6. Publish an experiment report with reproducible commands, corpus disclosures, validation results and cost.

## Current vs proposed model

| Parameter | Micro implemented | 125M proposed, not trained |
|---|---:|---:|
| Token vocabulary | 256 byte IDs | 32,000 BPE IDs |
| Layers | 4 | 14 |
| Hidden width | 128 | 768 |
| Heads | 4 | 12 |
| Context | 128 | 2048 |
| Attention/MLP | RoPE, causal SDPA, SwiGLU | Same modules with scaled training pipeline |

The proposed configuration is roughly **123.7 million trainable parameters** with tied token embeddings. Verify the exact count in code.

## Evidence needed before claiming any release

- Code revision, environment, seed, hardware, runtime, training throughput and peak memory.
- Tokenizer, dataset checksums, data provenance/rights, exact splits and contamination audit.
- Checkpoint, model card, validation/test metrics, generated samples and failure cases.
- Security/privacy and harmful-use evaluation before public chat deployment.

**Important distinction:** Micro validation currently uses the last 10% *of bytes* from a demo file and reports next-byte NLL/perplexity. This is an engineering regression signal, not a publication-quality language-model benchmark. Production evaluation must use document-disjoint splits and disclose the tokenizer.

## Technical references

- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/quicktour)
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
