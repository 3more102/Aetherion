# Aetherion — build plan

**Baseline:** 21 November 2026. **Repository:** `3more102/Aetherion`. **Goal:** build and document an original decoder-only language-model stack, progressing from a working educational prototype to a measured small language model. This is **not** a roadmap for matching GPT-6, Gemini, or other frontier models with hobby-scale resources.

## 1. Product and research scope

- **Aetherion-Micro** — 256-byte UTF-8 vocabulary; verify architecture, gradient flow, reproducibility, evaluation, and checkpoint loading.
- **Aetherion-125M** — research-size decoder-only model with a separately trained ~32k BPE tokenizer and a controlled, licensed corpus. Size is an engineering target, **not** a trained or benchmarked release.
- **Aetherion-Instruct** — add instruction-following only after pretraining yields a competent, measured base checkpoint.
- **Aetherion-Serve** — API, batching, token streaming, and optional optimized inference; subsequent scaling requires a compute and data budget.

**Architecture baseline:** pre-norm Transformer with RoPE, RMSNorm, causal SDPA, SwiGLU, and tied embeddings. Keep this architecture stable until there is an evaluated reason to change it. See [ARCHITECTURE.md](ARCHITECTURE.md).

## 2. Measurable stage gates

| Gate | Must ship | Evidence required | Do not advance if… |
|---|---|---|---|
| G0 — Micro correctness | CPU train/generate loop, tests, CI, full held-out byte evaluation | CI green; finite backward gradients; no future-token leakage; checkpoint reload; deterministic evaluation reporting | Training/eval command fails or tests are flaky |
| G1 — Data foundation | BPE tokenizer, data loader, provenance manifest, dedup, document-level split | Unicode round-trip tests; train-only tokenizer fitting; reproducible checksums; zero known cross-split duplicate documents | Data rights or train/test isolation cannot be established |
| G2 — 125M pilot | Config preset, optimizer/scheduler, AMP, accumulation, resumable optimizer checkpoints | Exact parameter count; small-set overfit; GPU memory + tokens/sec + cost projection; restart parity | OOM, numerical instability, or unaffordable run |
| G3 — Baseline pretraining | Budgeted training and held-out evaluation | Training token count, loss curves, held-out NLL/perplexity with tokenizer identity, contamination report and reproducible run metadata | Only training loss improves or held-out data is contaminated |
| G4 — Instruction and release | SFT/chat format, inference, model/data cards, safety tests | Task-specific before/after evaluation, multilingual evaluation, privacy/safety review, documented limitations | Basic quality or safety thresholds are not met |

These are **acceptance gates**, not promised dates. Schedule depends on available GPU, training data, and observed throughput.

## 3. Work order — the next eight engineering PRs

1. **PR-01: held-out evaluation + smoke tests** — complete byte-level evaluation for Micro; report hashes, exact target count and per-byte metrics. *A byte-level demo score is not comparable to token-level large-model benchmarks.*
2. **PR-02: reproducible experiments** — save code revision, dataset hash, split manifest, tokenizer identity, full config, environment, step/token counters, and RNG states. Write JSONL training logs.
3. **PR-03: resumable training** — checkpoint optimizer/scheduler/scaler/RNG state; resume with a test that matches uninterrupted training in deterministic mode.
4. **PR-04: tokenizer interface** — byte tokenizer retained for Micro; add BPE train/encode/decode/save/load and multilingual round-trip tests. Freeze special tokens.
5. **PR-05: licensed corpus pipeline** — provenance and licensing manifest, exact/near dedup, normalization, secrets/PII screening, document-grouped splits, evaluation contamination checks.
6. **PR-06: 125M configuration and correctness** — add config preset; exact parameter test; RoPE/causal masking tests; tiny-set overfit test.
7. **PR-07: training performance pilot** — AMP where supported, accumulation, gradient clipping and LR schedule; log tokens/sec, max VRAM, loss, and energy/compute estimates.
8. **PR-08: evaluation and release discipline** — fixed held-out tasks, model/data cards, baseline comparisons, and an explicit go/no-go report.

For development sequencing, PR-02/03 can proceed in parallel with tokenizer work only if interfaces are agreed. **Training on a large dataset must wait for PR-05.**

## 4. Data and training rules

1. Start with a small, **authorized** corpus; store original source, date, permissions/license, checksum, transformations, and exclusion decisions.
2. Deduplicate and split at document or source-group level **before** learning a tokenizer. Fit tokenizer only on the training split.
3. Freeze train/validation/test manifests and keep test data inaccessible to optimization.
4. Run a short, bounded pilot before committing to full pretraining. Record peak memory, measured tokens/second, effective batch tokens, precision, device, seed, and cost.
5. Decide the final training-token budget **from measured throughput and data availability**, not from a parameter count alone.
6. Report train and validation metrics separately. Publish generated examples with the context of their checkpoint and sampling parameters.

See [DATA_AND_EVALUATION.md](DATA_AND_EVALUATION.md).

## 5. Resource and architecture decisions

**First run:** CPU is sufficient for Micro smoke tests. No GPU or large dataset is required to validate implementation.

**125M:** choose hardware only after the profiling pilot; report device model, VRAM, precision, context length, batch size, and tokens/second. Estimate duration as `planned_training_tokens / measured_tokens_per_second` and costs from actual provider rates. Do not announce a firm budget before measurement.

**Bigger models:** 1B+ is conditional on licensed data and approved compute. Consider adaptation of an openly licensed pretrained model as a *separate track* if the objective is a practical assistant rather than end-to-end pretraining research.

## 6. Definition of done for every PR

- Tests added or updated and passing in CI.
- Reproduction command documented in the PR.
- Input shapes, data sources, and any changes to model behavior are documented.
- Benchmark or quantitative result included where performance claims are made.
- No secrets, private training files, large model weights, or copyrighted unlicensed content committed.
- Failure cases, known limits, and follow-up issue recorded.

## 7. Current state and immediate next action

The repository contains the Transformer, train/generate scripts, Micro smoke tests, CI workflow, and engineering documents. **These files existing does not certify model quality.** The first incremental implementation is `evaluate.py` and `tests/test_evaluate.py`; merge it only after CI passes.

After that, start **PR-02: reproducible experiments**. Its acceptance criteria are: one structured run record per experiment; dataset/checkpoint hashes; code version; same-seed deterministic test for short CPU runs; explicit configuration and evaluation outputs. Follow with resumable stateful checkpoints.

**Release principle:** publish what is measured, including failures; do not label Aetherion as GPT/Gemini-class without independent evidence.
