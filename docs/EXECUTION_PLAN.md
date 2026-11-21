# Aetherion: eight-week engineering execution plan

**Baseline:** Aetherion-Micro is an educational decoder-only Transformer using byte tokens and random initial weights. It is **not** a pretrained general-purpose assistant. This is a prioritized sequence, **not a promised completion date**; actual progress depends on reviews, data rights, hardware and evidence from the exit gates.

## Scope and milestones

| Stage | Priority | Deliverable | Exit criteria |
|---|---|---|---|
| Week 1 — M0 + M1 data | P0 | Authorized JSONL ingestion, exact deduplication, source-grouped splits, checksums, separate validation CLI, CI | Local data tests + 2-step Micro training smoke; no group/exact-content overlap; stable hashes |
| Week 2 — M1 tokenizer | P0 | Byte-level BPE, 32k target vocabulary, reproducible tokenizer artifact | Unicode round-trip, reproducible tokenizer hash; fit on train data only |
| Week 3 — M2 trainer | P0 | Config presets, gradient accumulation, AdamW, warmup + cosine learning rate, mixed precision where supported, resumable optimizer/RNG state | Overfit-small-batch test; resume vs continuous tolerance documented |
| Week 4 — M2 profiling | P0 | Aetherion-125M preset (~124M target), parameter count, memory and throughput benchmarks | Verify device memory/batch/context feasibility before major training |
| Weeks 5–6 — M2 pretraining | Conditional P1 | Train small or 125M baseline depending on licensed corpus and compute budget | Held-out loss/perplexity, token count, checkpoint checksum, compute report |
| Week 7 — M3 instructions | Conditional P1 | Authorized supervised instruction tuning and chat template | Improved held-out instruction adherence plus safety regression checks |
| Week 8 — M3 serving | Conditional P2 | Bounded REST inference endpoint, health check, model card, example client | Correct input validation, deployment smoke test, measured latency, documented limitations |

If 125M is infeasible, train and evaluate a smaller model rather than promising a larger one.

## PR / issue order

1. **M1 data foundation** — this branch: provenance, split manifest, exact dedup, explicit validation data.
2. **M1 tokenizer** — trained BPE in `aetherion/tokenizer.py`, train-only fitting, serialization and UTF-8 tests.
3. **M2 trainer** — gradients/optimizer scheduling, checkpoint/resume, reproducibility, throughput metrics.
4. **M2 benchmark** — preset, parameter audit, single-device cost and memory profile.
5. **M3 assistant + API** — supervised instruction tuning, evaluation and model hosting after a verified base model.

## Planned module boundaries

- `aetherion/model.py`: decoder-only Transformer and configuration.
- `aetherion/data.py`: dataset validation and split manifests.
- `aetherion/tokenizer.py` (next): versioned encode/decode and tokenizer artifacts.
- `train.py`: current Micro CLI; modular, resumable trainer is upcoming.
- `eval.py` (next): held-out loss and evaluation report.
- `serve.py` (later): bounded inference API and health check.

## Run reporting policy

Record git commit SHA, model configuration, seed, dataset/tokenizer hashes, source licenses, tokens processed, training/validation loss, external task metrics where applicable, optimizer and scheduler, hardware, elapsed time, peak memory, weight/checkpoint SHA-256, and known evaluation limitations. Model cards precede public weights release.

## Risks and non-goals

- **Data:** No blind scraping or private/proprietary data without permission. Exact dedup ≠ near-dedup; PII/secret scanning and benchmark contamination analysis still required.
- **Compute:** 125M pretraining is more costly than Micro smoke tests. 1B+ and frontier parity remain out of scope until feasibility is established.
- **Evaluation:** Synthetic demo repetitions cannot measure useful intelligence. Checkpoints and samples are not proof of assistant quality.
- **Deployment:** External tools, persistent memory and network access require sandboxing, abuse controls and privacy reviews.

References: [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html); [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/main/training_from_memory).
