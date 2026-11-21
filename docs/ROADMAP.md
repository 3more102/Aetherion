# Aetherion engineering roadmap

**Purpose:** deliver a verifiable, original decoder-only language model, not a marketing claim of frontier equivalence.

## Milestones (gated, not promises)

### M0 — Reproducible Micro prototype
- [ ] PyTorch decoder-only LM with causal attention, RoPE, RMSNorm, SwiGLU and tied token embeddings.
- [ ] Byte-level UTF-8 tokenizer for the demo only.
- [ ] Training/validation split, gradient clipping, checkpoint save/load, seeded runs.
- [ ] Inference sampler with temperature/top-k.
- [ ] Automated smoke test (forward, backward, checkpoint, generation).
- [ ] README commands demonstrated on CPU.
**Exit gate:** install + tests pass; 5 training steps execute; saved checkpoint reloads and generates tokens. Training quality is explicitly not asserted.

### M1 — Data and tokenizer baseline
- [ ] Byte-level BPE tokenizer with 32k target vocabulary; publish tokenizer config.
- [ ] Licensed corpus ingestion with checksums, provenance and document IDs.
- [ ] Exact/near deduplication, Unicode normalization policy, PII filtering, train/validation/test document-level separation.
- [ ] Prepare packed token files with reproducible split manifests.
- [ ] Record language mix, token count, document count and quality checks.
**Exit gate:** reproducible tokenizer with round-trip tests and an auditable, contamination-checked dataset manifest. No training on copyrighted/private material without authorization.

### M2 — Aetherion-125M baseline
**Proposed architecture** (adjust only with documented experiment):
- decoder-only Transformer, 14 layers, width 768, 12 attention heads;
- RoPE, RMSNorm, SwiGLU, causal SDPA, tied embeddings;
- approximately 124M parameters with a 32k tokenizer;
- 2048 context, AdamW, gradient accumulation, warmup + cosine LR, mixed precision when supported.
- [ ] Implement model/config presets and parameter-count test.
- [ ] Deterministic small-overfit test plus attention causality test.
- [ ] Token throughput, VRAM and training-time benchmarks.
- [ ] Long-running pretraining with resumable checkpoints.
- [ ] Held-out evaluation: NLL/perplexity, repetition and data leakage checks.
**Exit gate:** reproducible training run, no validation leakage, published eval table and cost/compute report. Do not promote based only on lower train loss.

### M3 — Assistant behavior
- [ ] Curated instruction data and supervised fine-tuning.
- [ ] Chat template and turn-boundary tokens.
- [ ] Refusal/safety and instruction-following evaluations.
- [ ] Optional retrieval with citations and tool-calling under a sandboxed executor.
- [ ] Human/automated benchmarking against appropriate baselines.
**Exit gate:** measured instruction-following improvement and documented limitations. No "GPT-class" claims.

### M4 — Larger-scale feasibility
- [ ] Compute budget and data scaling analysis before choosing model size.
- [ ] FSDP/DeepSpeed or equivalent distributed training path.
- [ ] Optimized inference (KV cache, quantization, batching) and model serving.
- [ ] Evaluations across tasks, languages, security, bias and misuse risk.
**Exit gate:** budget approved and pilot scaling benchmark verified. 1B/7B+ targets depend on hardware and corpus availability.

## First sprint — issues to finish in order
1. Initialize source files and smoke tests.
2. Add GitHub Actions CI (install, lint, unit/smoke tests).
3. Add tokenizer interface and BPE experiment.
4. Add data manifest and split/contamination checks.
5. Add evaluation script and checkpoint metadata.
6. Benchmark 125M configuration before full training.

## Acceptance and experiment logging
For every run, record code commit SHA, seed, model config, tokenizer hash, dataset manifest hash, environment, optimizer/schedule, tokens processed, training loss, validation loss, samples, elapsed time, hardware, peak memory, and checkpoint hash.

**Important:** This project can produce a real Transformer; matching frontier systems requires extensive resources and is not an implied deliverable.
